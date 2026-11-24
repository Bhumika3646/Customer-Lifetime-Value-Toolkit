
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

TRAIN_FILE = OUT / "temporal_train_snapshot_2024-06-30.csv"
TEST_FILE = OUT / "temporal_test_snapshot_2025-06-30.csv"

TARGET = "target_future_90d_revenue"

RESULT_FILE = OUT / "revenue_baseline_comparison.csv"


# --------------------------------------------------
# 2. Load monthly-plan customers
# --------------------------------------------------

def load_monthly_data(path, label):
    df = pd.read_csv(path, low_memory=False)

    required = [
        TARGET,
        "billing_period_at_cutoff",
        "plan_price_at_cutoff",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )

    period = (
        df["billing_period_at_cutoff"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df = df.loc[period.eq("monthly")].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["plan_price_at_cutoff"] = pd.to_numeric(
        df["plan_price_at_cutoff"], errors="coerce"
    )

    df = df.dropna(subset=[TARGET, "plan_price_at_cutoff"])
    df = df[df[TARGET] >= 0].copy()

    print(f"{label}: {len(df):,} monthly-plan customers")

    return df


# --------------------------------------------------
# 3. Define the evaluation metrics
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


# --------------------------------------------------
# 4. Load training and test snapshots
# --------------------------------------------------

train = load_monthly_data(TRAIN_FILE, "Training snapshot")
test = load_monthly_data(TEST_FILE, "Test snapshot")

print("\nTraining target mean:", round(train[TARGET].mean(), 2))
print("Test target mean:", round(test[TARGET].mean(), 2))

y_train = train[TARGET].astype(float)
y_test = test[TARGET].astype(float)


# --------------------------------------------------
# 5. Historical-revenue baseline
# --------------------------------------------------

# Use the historical 90-day revenue feature if available.
historical_candidates = [
    "revenue_90d",
    "historical_successful_revenue_90d",
    "historical_successful_revenue",
]

historical_column = next(
    (col for col in historical_candidates if col in train.columns),
    None,
)

results = []

if historical_column is not None:
    historical_prediction = pd.to_numeric(
        test[historical_column], errors="coerce"
    ).fillna(0).clip(lower=0)

    metrics = calculate_metrics(y_test, historical_prediction)

    results.append({
        "Model": f"Historical revenue baseline ({historical_column})",
        **metrics,
    })

    print(f"\nHistorical baseline uses: {historical_column}")
else:
    print(
        "\nWARNING: No historical revenue column was found. "
        "Historical baseline will be skipped."
    )


# --------------------------------------------------
# 6. Three-month billing baseline
# --------------------------------------------------

billing_prediction = (
    3 * test["plan_price_at_cutoff"]
).clip(lower=0)

metrics = calculate_metrics(y_test, billing_prediction)

results.append({
    "Model": "Three-month plan-price baseline",
    **metrics,
})


# --------------------------------------------------
# 7. Prepare machine-learning features
# --------------------------------------------------

# Exclude targets, identifiers and raw date columns.
# Keep engineered historical features available at the cutoff.
excluded_exact = {
    TARGET,
    "target_churn_within_90d",
    "customer_id",
    "subscription_id",
    "invoice_id",
    "payment_id",
    "event_id",
    "change_id",
    "snapshot_date",
    "cutoff_date",
    # Future/outcome fields are never valid prediction-time features.
    "status",
    "subscription_status",
    "reason",
    "churn_reason",
    "end_date",
    "churn_date",
    "last_successful_payment_date",
}

feature_columns = []

for col in train.columns:
    name = col.lower()

    if col in excluded_exact:
        continue

    if name.startswith("target_"):
        continue

    if name.endswith("_date") or name.endswith("_datetime"):
        continue

    if col not in test.columns:
        continue

    feature_columns.append(col)

if not feature_columns:
    raise ValueError("No usable model features were found.")

X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()

# Convert date-like values to numeric only if they are not
# excluded above; categorical features are handled separately.
numeric_columns = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = [
    col for col in feature_columns
    if col not in numeric_columns
]

# Remove infinite numeric values.
for col in numeric_columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers = []

if numeric_columns:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

preprocessor = ColumnTransformer(transformers=transformers)


# --------------------------------------------------
# 8. Train XGBoost on 2024 and test on 2025
# --------------------------------------------------

xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )),
])

xgb_model.fit(X_train, y_train)

xgb_prediction = xgb_model.predict(X_test)
xgb_prediction = np.clip(xgb_prediction, 0, None)

metrics = calculate_metrics(y_test, xgb_prediction)

results.append({
    "Model": "XGBoost",
    **metrics,
})

print(f"\nNumber of model features: {len(feature_columns)}")


# --------------------------------------------------
# 9. Compare results and save to a new file
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE", ascending=True
).reset_index(drop=True)

OUT.mkdir(parents=True, exist_ok=True)
results_df.to_csv(RESULT_FILE, index=False)

print("\n" + "=" * 70)
print("REVENUE BASELINE COMPARISON — MONTHLY PLANS")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

print("\nHow to read the metrics:")
print("- Lower MAE is better.")
print("- Lower RMSE is better.")
print("- Higher R2 is better; negative R2 is possible.")
print("- All approaches were evaluated on the same test customers.")

print(f"\nResults saved to:\n{RESULT_FILE}")
print("\nOriginal snapshots and result files were not modified.")
