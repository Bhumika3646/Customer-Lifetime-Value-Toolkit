
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = OUTPUT_DIR / "temporal_train_snapshot_2024-06-30.csv"
TEST_PATH = OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv"

VALIDATION_OUTPUT = OUTPUT_DIR / "hybrid_validation_results.csv"
TEST_OUTPUT = OUTPUT_DIR / "hybrid_test_results.csv"

RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    "customer_id", "subscription_id", "invoice_id", "payment_id",
    "event_id", "change_id", "signup_date", "start_date", "end_date",
    "invoice_date", "due_date", "payment_date", "churn_date",
    "change_date", "last_successful_payment_date",
    "snapshot_cutoff_date", "target_horizon_end_date",
    "status", "reason", "plan_id", "plan_id_at_cutoff",
    "plan_id_lookup", "target_future_90d_churn",
    "target_churn_90d",
}


def find_target(df):
    matches = [
        col for col in df.columns
        if "target" in col.lower() and "revenue" in col.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not identify revenue target: {matches}")
    return matches[0]


def prepare_monthly(df, target):
    df = df[
        df["billing_period_at_cutoff"].astype(str).str.lower().str.strip()
        == "monthly"
    ].copy()

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    return df


def make_model(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def predictions_for(model, X, df):
    xgb_predictions = np.maximum(model.predict(X), 0)

    price = pd.to_numeric(
        df["plan_price_at_cutoff"], errors="coerce"
    ).fillna(0).to_numpy()

    baseline_predictions = price * 3

    plan_names = df["plan_name_at_cutoff"].fillna("Unknown").astype(str).to_numpy()

    # Hybrid strategy fixed before evaluation:
    # Free = zero; Basic = XGBoost; Pro/Premium = plan-price baseline.
    hybrid = xgb_predictions.copy()

    for i, plan in enumerate(plan_names):
        if plan.strip().lower() == "free":
            hybrid[i] = 0
        elif plan.strip().lower() in {"pro", "premium"}:
            hybrid[i] = baseline_predictions[i]
        # Basic and any other plan retain the XGBoost prediction.

    return {
        "XGBoost": xgb_predictions,
        "Plan-price baseline": baseline_predictions,
        "Hybrid": hybrid,
    }


def metric_rows(df, predictions, target, split_name):
    rows = []

    groups = [("ALL PLANS", df.index)] + [
        (str(plan), group.index)
        for plan, group in df.groupby("plan_name_at_cutoff", dropna=False)
    ]

    for plan, indices in groups:
        group = df.loc[indices]
        actual = group[target].to_numpy(dtype=float)

        for model_name, all_predictions in predictions.items():
            predicted = all_predictions[group.index.map(df.index.get_loc)]

            rows.append({
                "split": split_name,
                "plan": plan,
                "model": model_name,
                "customers": len(group),
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "R2": (
                    r2_score(actual, predicted)
                    if len(actual) >= 2 and np.unique(actual).size >= 2
                    else np.nan
                ),
            })

    return rows


def main():
    train_path = TRAIN_PATH
    test_path = TEST_PATH

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Temporal snapshots not found. Run 05_temporal_validation.py first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target = find_target(train)

    for col in ["billing_period_at_cutoff", "plan_name_at_cutoff",
                "plan_price_at_cutoff", target]:
        if col not in test.columns:
            raise ValueError(f"Missing required test column: {col}")

    train = prepare_monthly(train, target)
    test = prepare_monthly(test, target)

    features = [
        col for col in train.columns
        if col in test.columns
        and col not in EXCLUDED_COLUMNS
        and col != target
        and not col.lower().startswith("target_")
        and "churn_date" not in col.lower()
        and "churn_reason" not in col.lower()
        and "end_date" not in col.lower()
        and "date" not in col.lower()
        and "timestamp" not in col.lower()
        and not pd.api.types.is_datetime64_any_dtype(train[col])
    ]

    if not features:
        raise ValueError("No usable features were found.")

    # -------------------------------------------------------
    # VALIDATION: hold out part of the 2024 training snapshot.
    # -------------------------------------------------------
    train_part, val_part = train_test_split(
        train,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    X_train = train_part[features].copy()
    X_val = val_part[features].copy()
    y_train = train_part[target].astype(float)
    y_val = val_part[target].astype(float)

    validation_model = make_model(X_train)
    validation_model.fit(X_train, y_train)

    val_predictions = predictions_for(
        validation_model, X_val, val_part
    )

    validation_rows = metric_rows(
        val_part, val_predictions, target, "Validation"
    )

    validation_results = pd.DataFrame(validation_rows)
    validation_results.to_csv(VALIDATION_OUTPUT, index=False)

    # -------------------------------------------------------
    # FINAL TEST: fit on all 2024 monthly training rows.
    # -------------------------------------------------------
    X_full_train = train[features].copy()
    y_full_train = train[target].astype(float)
    X_test = test[features].copy().reindex(columns=features)

    final_model = make_model(X_full_train)
    final_model.fit(X_full_train, y_full_train)

    test_predictions = predictions_for(final_model, X_test, test)

    test_rows = metric_rows(
        test, test_predictions, target, "2025 Test"
    )

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(TEST_OUTPUT, index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFeatures used:", len(features))
    print("Training rows:", len(train))
    print("Validation rows:", len(val_part))
    print("2025 test rows:", len(test))

    print("\n=== VALIDATION RESULTS ===")
    print(
        validation_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n=== 2025 TEST RESULTS ===")
    print(
        test_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved validation results:", VALIDATION_OUTPUT)
    print("Saved test results:", TEST_OUTPUT)


if __name__ == "__main__":
    main()
