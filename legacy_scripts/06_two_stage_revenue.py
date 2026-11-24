
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_FILE = OUTPUT_DIR / "temporal_train_snapshot_2024-06-30.csv"
TEST_FILE = OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv"

TARGET = "target_future_90d_revenue"
ID_COLUMN = "customer_id"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD TEMPORAL TRAIN AND TEST DATA
# --------------------------------------------------

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Testing file not found: {TEST_FILE}")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("=" * 65)
print("TWO-STAGE FUTURE REVENUE PREDICTION")
print("=" * 65)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' must exist in both snapshot files."
    )

# --------------------------------------------------
# 3. SELECT ONLY APPROVED PREDICTOR FEATURES
# --------------------------------------------------
# These features describe customer/subscription details and
# historical payment behaviour available at the cutoff date.
#
# Customer IDs, raw dates, subscription status, churn outcomes,
# and future-revenue targets are NOT model predictors.

FEATURES = [
    "company_size",
    "industry",
    "acquisition_channel",
    "country",
    "plan_name_at_cutoff",
    "billing_period_at_cutoff",
    "plan_price_at_cutoff",
    "tenure_days",
    "subscription_age_days",
    "historical_successful_payment_count",
    "historical_successful_revenue",
    "payment_count_30d",
    "revenue_30d",
    "payment_count_60d",
    "revenue_60d",
    "payment_count_90d",
    "revenue_90d",
    "days_since_last_successful_payment",
]

missing_train = [c for c in FEATURES + [TARGET] if c not in train_df.columns]
missing_test = [c for c in FEATURES + [TARGET] if c not in test_df.columns]

if missing_train or missing_test:
    raise ValueError(
        f"Missing training columns: {missing_train}\n"
        f"Missing testing columns: {missing_test}"
    )

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = pd.to_numeric(train_df[TARGET], errors="coerce")
y_test = pd.to_numeric(test_df[TARGET], errors="coerce")

# Keep only rows with valid target values.
train_valid = y_train.notna()
test_valid = y_test.notna()

X_train = X_train.loc[train_valid].reset_index(drop=True)
y_train = y_train.loc[train_valid].reset_index(drop=True)

X_test = X_test.loc[test_valid].reset_index(drop=True)
y_test = y_test.loc[test_valid].reset_index(drop=True)

train_ids = (
    train_df.loc[train_valid, ID_COLUMN].reset_index(drop=True)
    if ID_COLUMN in train_df.columns
    else pd.Series(np.arange(len(X_train)), name=ID_COLUMN)
)

test_ids = (
    test_df.loc[test_valid, ID_COLUMN].reset_index(drop=True)
    if ID_COLUMN in test_df.columns
    else pd.Series(np.arange(len(X_test)), name=ID_COLUMN)
)

# Future revenue should be non-negative for this target.
if (y_train < 0).any() or (y_test < 0).any():
    raise ValueError(
        "Negative future revenue values were found. "
        "Check target construction before continuing."
    )

# Stage 1 target: 1 means positive future revenue; 0 means zero.
y_train_positive = (y_train > 0).astype(int)
y_test_positive = (y_test > 0).astype(int)

if y_train_positive.nunique() < 2:
    raise ValueError(
        "Training data must contain both zero and positive future revenue."
    )

if int(y_train_positive.sum()) < 2:
    raise ValueError(
        "Not enough positive-revenue training rows for Stage 2."
    )

print("\nTraining revenue summary:")
print(f"Customers with positive future revenue: {y_train_positive.sum():,}")
print(f"Customers with zero future revenue   : {(y_train_positive == 0).sum():,}")
print(f"Training zero-revenue percentage     : {(y_train == 0).mean() * 100:.2f}%")
print(f"Testing zero-revenue percentage      : {(y_test == 0).mean() * 100:.2f}%")

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

NUMERIC_FEATURES = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

CATEGORICAL_FEATURES = [
    c for c in FEATURES if c not in NUMERIC_FEATURES
]

numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessor, NUMERIC_FEATURES),
        ("categorical", categorical_preprocessor, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# --------------------------------------------------
# 5. STAGE 1: PREDICT WHETHER FUTURE REVENUE IS POSITIVE
# --------------------------------------------------

print("\n" + "-" * 65)
print("STAGE 1: POSITIVE-REVENUE CLASSIFICATION")
print("-" * 65)

stage1_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

stage1_model.fit(X_train, y_train_positive)

positive_probability = stage1_model.predict_proba(X_test)[:, 1]

# Use the default 0.50 probability threshold.
# Do not tune this threshold on the final test data.
predicted_positive = (positive_probability >= 0.50).astype(int)

stage1_accuracy = accuracy_score(y_test_positive, predicted_positive)
stage1_precision = precision_score(
    y_test_positive, predicted_positive, zero_division=0
)
stage1_recall = recall_score(
    y_test_positive, predicted_positive, zero_division=0
)
stage1_f1 = f1_score(
    y_test_positive, predicted_positive, zero_division=0
)

if y_test_positive.nunique() == 2:
    stage1_auc = roc_auc_score(y_test_positive, positive_probability)
else:
    stage1_auc = np.nan

print(f"Accuracy : {stage1_accuracy:.4f}")
print(f"Precision: {stage1_precision:.4f}")
print(f"Recall   : {stage1_recall:.4f}")
print(f"F1-score : {stage1_f1:.4f}")
print(f"ROC-AUC  : {stage1_auc:.4f}")

# --------------------------------------------------
# 6. STAGE 2: PREDICT THE AMOUNT OF POSITIVE REVENUE
# --------------------------------------------------
# Train Stage 2 only on training customers whose future
# revenue was positive. Test data is never used for fitting.

print("\n" + "-" * 65)
print("STAGE 2: REVENUE AMOUNT REGRESSION")
print("-" * 65)

positive_train_mask = y_train > 0

X_train_positive = X_train.loc[positive_train_mask].copy()
y_train_positive_amount = y_train.loc[positive_train_mask].copy()

stage2_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        ),
    ]
)

stage2_model.fit(X_train_positive, y_train_positive_amount)

# Predict the possible revenue amount for every test customer.
# Stage 1 decides whether to keep that amount or set it to zero.
stage2_amount = stage2_model.predict(X_test)
stage2_amount = np.maximum(stage2_amount, 0)

combined_prediction = np.where(
    predicted_positive == 1,
    stage2_amount,
    0.0,
)

# --------------------------------------------------
# 7. COMPARE AGAINST BASELINES AND A SINGLE-STAGE MODEL
# --------------------------------------------------

# Baseline A: predict zero future revenue for every customer.
zero_prediction = np.zeros(len(y_test))

# Baseline B / comparison: train one regression model on all
# training customers, including those with zero future revenue.
single_stage_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        ),
    ]
)

single_stage_model.fit(X_train, y_train)
single_stage_prediction = np.maximum(
    single_stage_model.predict(X_test), 0
)

def regression_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

results = []

for model_name, predictions in [
    ("Zero Revenue Baseline", zero_prediction),
    ("Single-Stage Gradient Boosting", single_stage_prediction),
    ("Two-Stage Revenue Model", combined_prediction),
]:
    metrics = regression_metrics(y_test, predictions)
    results.append(
        {
            "model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        }
    )

results_df = pd.DataFrame(results).sort_values("MAE")

print("\n" + "=" * 65)
print("TEST-SET REVENUE PREDICTION COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------
# 8. SAVE RESULTS AND CUSTOMER-LEVEL PREDICTIONS
# --------------------------------------------------

results_file = OUTPUT_DIR / "two_stage_revenue_results.csv"
predictions_file = OUTPUT_DIR / "two_stage_revenue_predictions.csv"
summary_file = OUTPUT_DIR / "two_stage_revenue_summary.txt"

results_df.to_csv(results_file, index=False)

predictions_df = pd.DataFrame(
    {
        ID_COLUMN: test_ids,
        "actual_future_90d_revenue": y_test,
        "actual_positive_revenue": y_test_positive,
        "predicted_positive_probability": positive_probability,
        "predicted_positive_revenue": predicted_positive,
        "stage2_predicted_amount": stage2_amount,
        "two_stage_predicted_revenue": combined_prediction,
        "single_stage_predicted_revenue": single_stage_prediction,
    }
)

predictions_df.to_csv(predictions_file, index=False)

summary_lines = [
    "TWO-STAGE FUTURE REVENUE PREDICTION SUMMARY",
    "=" * 50,
    f"Training customers: {len(X_train)}",
    f"Testing customers: {len(X_test)}",
    f"Training positive-revenue customers: {int(y_train_positive.sum())}",
    f"Testing positive-revenue customers: {int(y_test_positive.sum())}",
    f"Stage 1 accuracy: {stage1_accuracy:.4f}",
    f"Stage 1 precision: {stage1_precision:.4f}",
    f"Stage 1 recall: {stage1_recall:.4f}",
    f"Stage 1 F1-score: {stage1_f1:.4f}",
    f"Stage 1 ROC-AUC: {stage1_auc:.4f}",
    "",
    "Revenue regression comparison:",
    results_df.to_string(index=False),
    "",
    "Important interpretation:",
    "The target is successful-payment revenue during the next 90 days.",
    "It is a short-term revenue proxy, not true lifetime value or profit.",
    "The dataset is synthetic; results should be reported with that limitation.",
    "The temporal test set contains customers also present in training.",
    "Therefore, this evaluates a later period, not unseen-customer generalization.",
]

summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

print("\nFiles saved:")
print(results_file)
print(predictions_file)
print(summary_file)

print("\nTwo-stage model completed successfully.")
