
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_FILE = OUTPUT_DIR / "temporal_train_snapshot_2024-06-30.csv"
TEST_FILE = OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv"

TARGET = "target_future_90d_revenue"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# Separate output files: original results remain untouched.
VALIDATION_OUTPUT = (
    OUTPUT_DIR / "monthly_only_regression_validation_results.csv"
)
TEST_OUTPUT = OUTPUT_DIR / "monthly_only_regression_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "monthly_only_revenue_summary.txt"


# ============================================================
# 2. LOAD EXISTING SNAPSHOTS
# ============================================================

def load_snapshots():
    for path in [TRAIN_FILE, TEST_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required snapshot not found: {path}\n"
                "Run 05_temporal_validation.py first."
            )

    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)

    required = {
        "billing_period_at_cutoff",
        TARGET,
        "customer_id",
    }

    for name, df in [("training", train), ("test", test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name.capitalize()} snapshot is missing columns: "
                f"{sorted(missing)}"
            )

    return train, test


# ============================================================
# 3. FILTER TO MONTHLY-PLAN CUSTOMERS
# ============================================================

def keep_monthly_customers(df):
    period = (
        df["billing_period_at_cutoff"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return df.loc[period.eq("monthly")].copy()


# ============================================================
# 4. SELECT FEATURES
# ============================================================

EXCLUDED_FEATURES = {
    "customer_id",
    "subscription_id",
    "invoice_id",
    "payment_id",
    "event_id",
    "change_id",
    "signup_date",
    "start_date",
    "end_date",
    "invoice_date",
    "due_date",
    "payment_date",
    "churn_date",
    "change_date",
    "last_successful_payment_date",
    "snapshot_cutoff_date",
    "target_horizon_end_date",
    "status",
    "reason",
    TARGET,
    "target_churn_within_90d",
    "plan_id",
    "plan_id_at_cutoff",
    "plan_id_lookup",
}


def get_feature_columns(train, test):
    forbidden_terms = [
        "target_",
        "churn_date",
        "churn_reason",
        "end_date",
    ]

    columns = [
        column
        for column in train.columns
        if column in test.columns
        and column not in EXCLUDED_FEATURES
        and not any(term in column.lower() for term in forbidden_terms)
    ]

    # Remove any remaining date-like columns.
    columns = [
        column for column in columns
        if not (
            pd.api.types.is_datetime64_any_dtype(train[column])
            or pd.api.types.is_datetime64_any_dtype(test[column])
        )
    ]

    if not columns:
        raise ValueError("No usable common features were found.")

    return columns


# ============================================================
# 5. PREPROCESSING AND MODELS
# ============================================================

def build_pipeline(model, X):
    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not installed; skipping that model.")

    return models


def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


# ============================================================
# 6. VALIDATE, SELECT, AND TEST
# ============================================================

def main():
    print("=" * 68)
    print("MONTHLY-ONLY FUTURE 90-DAY REVENUE SENSITIVITY ANALYSIS")
    print("=" * 68)

    train_all, test_all = load_snapshots()

    train = keep_monthly_customers(train_all)
    test = keep_monthly_customers(test_all)

    print("\nSnapshot filtering:")
    print(f"Original training snapshot: {len(train_all):,}")
    print(f"Monthly training customers: {len(train):,}")
    print(f"Original test snapshot:     {len(test_all):,}")
    print(f"Monthly test customers:     {len(test):,}")

    if len(train) < 10 or len(test) < 2:
        raise ValueError(
            "Too few monthly-plan customers for this evaluation."
        )

    X_columns = get_feature_columns(train, test)

    X_train = train[X_columns].copy()
    X_test = test[X_columns].copy()

    X_test = X_test.reindex(columns=X_train.columns)

    y_train = pd.to_numeric(
        train[TARGET], errors="coerce"
    ).fillna(0)

    y_test = pd.to_numeric(
        test[TARGET], errors="coerce"
    ).fillna(0)

    if (y_train < 0).any() or (y_test < 0).any():
        raise ValueError(
            "Negative revenue targets found. Audit the targets first."
        )

    # Random validation holdout from the 2024 monthly snapshot.
    indices = np.arange(len(X_train))

    fit_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    X_fit = X_train.iloc[fit_indices].copy()
    X_val = X_train.iloc[val_indices].copy()
    y_fit = y_train.iloc[fit_indices].copy()
    y_val = y_train.iloc[val_indices].copy()

    print(f"\nFeatures used: {len(X_columns)}")
    print(f"Validation fit rows: {len(X_fit):,}")
    print(f"Validation rows:     {len(X_val):,}")
    print(f"Final test rows:     {len(X_test):,}")

    print("\nTarget distribution:")
    print(f"Training mean revenue: {y_train.mean():.2f}")
    print(f"Training zero targets: {(y_train == 0).sum():,}")
    print(f"Test mean revenue:     {y_test.mean():.2f}")
    print(f"Test zero targets:     {(y_test == 0).sum():,}")

    # Mean baseline, calculated using fit data only.
    baseline_val_predictions = np.full(len(y_val), y_fit.mean())
    baseline_val = calculate_metrics(y_val, baseline_val_predictions)

    print("\nValidation mean-baseline MAE:", f"{baseline_val['MAE']:.4f}")

    validation_rows = [{
        "model": "Mean Baseline",
        "validation_MAE": baseline_val["MAE"],
        "validation_RMSE": baseline_val["RMSE"],
        "validation_R2": baseline_val["R2"],
    }]

    models = get_models()
    successful_models = {}

    # Select the model based on validation MAE only.
    for name, model in models.items():
        try:
            pipeline = build_pipeline(model, X_fit)
            pipeline.fit(X_fit, y_fit)

            predictions = np.maximum(pipeline.predict(X_val), 0)
            metrics = calculate_metrics(y_val, predictions)

            validation_rows.append({
                "model": name,
                "validation_MAE": metrics["MAE"],
                "validation_RMSE": metrics["RMSE"],
                "validation_R2": metrics["R2"],
            })

            successful_models[name] = model

            print(
                f"{name}: MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}"
            )

        except Exception as exc:
            print(f"{name} failed: {exc}")

    validation_results = pd.DataFrame(validation_rows)
    validation_results = validation_results.sort_values(
        "validation_MAE"
    ).reset_index(drop=True)

    validation_results.to_csv(VALIDATION_OUTPUT, index=False)

    best_name = validation_results.iloc[0]["model"]
    print(f"\nSelected using validation MAE: {best_name}")

    # Evaluate selected approach on the 2025 monthly test snapshot.
    if best_name == "Mean Baseline":
        test_predictions = np.full(len(y_test), y_train.mean())
    else:
        final_pipeline = build_pipeline(
            successful_models[best_name], X_train
        )
        final_pipeline.fit(X_train, y_train)
        test_predictions = np.maximum(
            final_pipeline.predict(X_test), 0
        )

    test_metrics = calculate_metrics(y_test, test_predictions)

    test_results = pd.DataFrame([{
        "selected_model": best_name,
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_R2": test_metrics["R2"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_zero_target_pct": 100 * (y_train == 0).mean(),
        "test_zero_target_pct": 100 * (y_test == 0).mean(),
        "train_mean_revenue": y_train.mean(),
        "test_mean_revenue": y_test.mean(),
    }])

    test_results.to_csv(TEST_OUTPUT, index=False)

    print("\n" + "=" * 68)
    print("FINAL MONTHLY-ONLY TEST RESULTS")
    print("=" * 68)
    print(f"Selected model: {best_name}")
    print(f"MAE:  {test_metrics['MAE']:.4f}")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"R2:   {test_metrics['R2']:.4f}")

    print("\nSaved files:")
    print(VALIDATION_OUTPUT)
    print(TEST_OUTPUT)

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as file:
        file.write("MONTHLY-ONLY REVENUE SENSITIVITY ANALYSIS\n")
        file.write("=" * 60 + "\n")
        file.write("Training cutoff: 2024-06-30\n")
        file.write("Training target horizon ends: 2024-09-28\n")
        file.write("Test cutoff: 2025-06-30\n")
        file.write("Test target horizon ends: 2025-09-28\n\n")
        file.write(f"Original training rows: {len(train_all)}\n")
        file.write(f"Monthly training rows: {len(train)}\n")
        file.write(f"Original test rows: {len(test_all)}\n")
        file.write(f"Monthly test rows: {len(test)}\n\n")
        file.write(
            "Target: successful-payment revenue recorded during the "
            "90 days after each cutoff. This is a short-term revenue "
            "proxy, not true customer lifetime value or profit.\n\n"
        )
        file.write(
            "Only customers classified as monthly at the relevant "
            "snapshot cutoff are included. Validation is a random "
            "holdout from the 2024 monthly snapshot. The 2025 monthly "
            "snapshot is reserved for final evaluation.\n\n"
        )
        file.write(
            "The original dataset's annual-payment records are absent "
            "from 2025. This monthly-only analysis checks sensitivity "
            "to that limitation but does not establish real-world "
            "generalization. The dataset appears synthetic.\n\n"
        )
        file.write("VALIDATION RESULTS\n")
        file.write(validation_results.to_string(index=False) + "\n\n")
        file.write("FINAL TEST RESULTS\n")
        file.write(test_results.to_string(index=False) + "\n")

    print("Summary:", SUMMARY_OUTPUT)
    print("\nAnalysis complete. Original results were not overwritten.")


if __name__ == "__main__":
    main()
