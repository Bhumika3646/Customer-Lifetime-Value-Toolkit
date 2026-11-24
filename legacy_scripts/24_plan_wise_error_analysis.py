
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = OUTPUT_DIR / "temporal_train_snapshot_2024-06-30.csv"
TEST_PATH = OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv"

METRICS_PATH = OUTPUT_DIR / "plan_wise_error_analysis.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "plan_wise_predictions.csv"

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


def find_revenue_target(df):
    matches = [
        col for col in df.columns
        if "target" in col.lower() and "revenue" in col.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not identify revenue target: {matches}")
    return matches[0]


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


def make_pipeline(X):
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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", make_xgb()),
    ])


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2 and np.unique(actual).size >= 2
            else np.nan
        ),
    }


def main():
    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Temporal snapshots are missing. Run 05_temporal_validation.py first."
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target = find_revenue_target(train)

    required = ["billing_period_at_cutoff", "plan_name_at_cutoff",
                "plan_price_at_cutoff"]

    for col in required + [target]:
        if col not in test.columns:
            raise ValueError(f"Required column missing from test snapshot: {col}")

    # Match the monthly-only evaluation used previously.
    train = train[
        train["billing_period_at_cutoff"].astype(str).str.lower().str.strip()
        == "monthly"
    ].copy()

    test = test[
        test["billing_period_at_cutoff"].astype(str).str.lower().str.strip()
        == "monthly"
    ].copy()

    train[target] = pd.to_numeric(train[target], errors="coerce")
    test[target] = pd.to_numeric(test[target], errors="coerce")

    train = train.dropna(subset=[target]).copy()
    test = test.dropna(subset=[target]).copy()

    y_train = train[target].astype(float)
    y_test = test[target].astype(float)

    # --------------------------------------------------------
    # 3. BUILD FEATURES
    # --------------------------------------------------------
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
        raise ValueError("No usable model features found.")

    X_train = train[features].copy()
    X_test = test[features].copy().reindex(columns=X_train.columns)

    # --------------------------------------------------------
    # 4. TRAIN FULL XGBOOST ON TRAINING SNAPSHOT
    # --------------------------------------------------------
    model = make_pipeline(X_train)
    model.fit(X_train, y_train)

    full_predictions = np.maximum(model.predict(X_test), 0)

    # Simple baseline: three times the monthly plan price.
    baseline_predictions = (
        pd.to_numeric(test["plan_price_at_cutoff"], errors="coerce")
        .fillna(0)
        .to_numpy() * 3
    )

    # --------------------------------------------------------
    # 5. SAVE CUSTOMER-LEVEL PREDICTIONS
    # --------------------------------------------------------
    output = pd.DataFrame({
        "customer_id": test["customer_id"].values,
        "plan_name": test["plan_name_at_cutoff"].fillna("Unknown").values,
        "actual_future_90d_revenue": y_test.values,
        "full_xgboost_prediction": full_predictions,
        "three_month_plan_price_prediction": baseline_predictions,
    })

    output["full_absolute_error"] = (
        output["actual_future_90d_revenue"]
        - output["full_xgboost_prediction"]
    ).abs()

    output["baseline_absolute_error"] = (
        output["actual_future_90d_revenue"]
        - output["three_month_plan_price_prediction"]
    ).abs()

    # --------------------------------------------------------
    # 6. METRICS FOR EACH PLAN
    # --------------------------------------------------------
    rows = []

    for plan_name, group in output.groupby("plan_name", dropna=False):
        actual = group["actual_future_90d_revenue"]

        for model_name, prediction_column in [
            ("Full XGBoost", "full_xgboost_prediction"),
            ("Three-month plan-price baseline",
             "three_month_plan_price_prediction"),
        ]:
            metrics = calculate_metrics(
                actual,
                group[prediction_column],
            )

            rows.append({
                "plan_name": plan_name,
                "model": model_name,
                "customers": len(group),
                "actual_mean_revenue": actual.mean(),
                "actual_zero_revenue_count": int((actual == 0).sum()),
                **metrics,
            })

    metrics_df = pd.DataFrame(rows)

    # Save outputs without modifying the original snapshots.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    output.to_csv(PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 78)
    print("PLAN-WISE FUTURE 90-DAY REVENUE ERROR ANALYSIS")
    print("=" * 78)
    print("Training rows:", len(train))
    print("Test rows:", len(test))
    print("Features used:", len(features))
    print("\nMetrics by plan:")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved metrics:", METRICS_PATH)
    print("Saved customer predictions:", PREDICTIONS_PATH)

    print(
        "\nInterpretation: compare MAE and RMSE within each plan. "
        "Lower values indicate smaller prediction errors. "
        "R2 may be undefined when actual revenue has no variation."
    )


if __name__ == "__main__":
    main()
