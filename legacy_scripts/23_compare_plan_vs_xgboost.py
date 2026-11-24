
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = OUTPUT_DIR / "temporal_train_snapshot_2024-06-30.csv"
TEST_PATH = OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv"

RESULT_PATH = OUTPUT_DIR / "plan_vs_xgboost_comparison.csv"

RANDOM_STATE = 42

# Raw identifiers, target fields, and unavailable-at-cutoff data
EXCLUDED_COLUMNS = {
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
    "plan_id",
    "plan_id_at_cutoff",
    "plan_id_lookup",
    "target_future_90d_churn",
    "target_churn_90d",
}

PLAN_FEATURE_TERMS = (
    "plan_id",
    "plan_name",
    "plan_price",
    "billing_period",
)


# ============================================================
# 2. LOAD SNAPSHOTS
# ============================================================

def find_revenue_target(df):
    candidates = [
        col for col in df.columns
        if "target" in col.lower()
        and "revenue" in col.lower()
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely identify the future revenue target. "
            f"Matching columns: {candidates}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return candidates[0]


def load_snapshots():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target_col = find_revenue_target(train)

    if target_col not in test.columns:
        raise ValueError(
            f"Target column {target_col!r} is missing from test data."
        )

    required = ["billing_period_at_cutoff", "plan_price_at_cutoff"]

    for col in required:
        if col not in train.columns or col not in test.columns:
            raise ValueError(
                f"Required column {col!r} is missing. "
                "Check the saved temporal snapshots."
            )

    # This comparison is restricted to monthly plans, matching
    # the scope of the previous monthly-only evaluation.
    train = train[
        train["billing_period_at_cutoff"]
        .astype(str).str.strip().str.lower().eq("monthly")
    ].copy()

    test = test[
        test["billing_period_at_cutoff"]
        .astype(str).str.strip().str.lower().eq("monthly")
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "No monthly-plan rows found in the snapshots. "
            "Check billing_period_at_cutoff values."
        )

    train[target_col] = pd.to_numeric(
        train[target_col], errors="coerce"
    )
    test[target_col] = pd.to_numeric(
        test[target_col], errors="coerce"
    )

    train = train.dropna(subset=[target_col]).copy()
    test = test.dropna(subset=[target_col]).copy()

    print("Revenue target:", target_col)
    print("Monthly training rows:", len(train))
    print("Monthly test rows:", len(test))
    print("Train target mean:", round(train[target_col].mean(), 4))
    print("Test target mean:", round(test[target_col].mean(), 4))

    return train, test, target_col


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

def get_feature_columns(train, test, target_col, remove_plan_features):
    columns = [
        col for col in train.columns
        if col in test.columns
        and col not in EXCLUDED_COLUMNS
        and col != target_col
        and not col.lower().startswith("target_")
        and "churn_date" not in col.lower()
        and "churn_reason" not in col.lower()
        and "end_date" not in col.lower()
        and not pd.api.types.is_datetime64_any_dtype(train[col])
    ]

    # Exclude object/date fields that encode raw timestamps.
    columns = [
        col for col in columns
        if not any(
            term in col.lower()
            for term in (
                "date",
                "timestamp",
            )
        )
    ]

    if remove_plan_features:
        columns = [
            col for col in columns
            if not any(
                term in col.lower()
                for term in PLAN_FEATURE_TERMS
            )
        ]

    if not columns:
        raise ValueError("No usable features remain.")

    return columns


def build_model_pipeline(X, model):
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
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
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def make_xgb():
    return XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# 4. EVALUATION
# ============================================================

def calculate_metrics(name, actual, predicted):
    predicted = np.maximum(np.asarray(predicted), 0)

    return {
        "model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "test_rows": len(actual),
    }


def main():
    train, test, target_col = load_snapshots()

    y_train = train[target_col].astype(float)
    y_test = test[target_col].astype(float)

    # Baseline: three times the plan's monthly price.
    baseline_predictions = (
        pd.to_numeric(
            test["plan_price_at_cutoff"], errors="coerce"
        )
        .fillna(0)
        .to_numpy()
        * 3
    )

    results = [
        calculate_metrics(
            "Three-month plan-price baseline",
            y_test,
            baseline_predictions,
        )
    ]

    # Full model: all eligible features used by this script.
    full_columns = get_feature_columns(
        train, test, target_col,
        remove_plan_features=False,
    )

    # Reduced model: remove plan-related fields, including price.
    reduced_columns = get_feature_columns(
        train, test, target_col,
        remove_plan_features=True,
    )

    print("\nFull-model feature count:", len(full_columns))
    print("Reduced-model feature count:", len(reduced_columns))

    for model_name, feature_columns in [
        ("Full XGBoost", full_columns),
        ("Reduced XGBoost (no plan features)", reduced_columns),
    ]:
        X_train = train[feature_columns].copy()
        X_test = test[feature_columns].copy()

        # Align feature columns and order.
        X_test = X_test.reindex(columns=X_train.columns)

        pipeline = build_model_pipeline(X_train, make_xgb())
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        results.append(
            calculate_metrics(model_name, y_test, predictions)
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "MAE", ascending=True
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULT_PATH, index=False)

    print("\n" + "=" * 68)
    print("MONTHLY-PLAN FUTURE 90-DAY REVENUE COMPARISON")
    print("=" * 68)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved comparison to:")
    print(RESULT_PATH)

    print(
        "\nInterpretation: lower MAE and RMSE are better. "
        "Higher R2 is better. Compare all models on these same test rows."
    )


if __name__ == "__main__":
    main()
