
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"

train_file = OUT / "temporal_train_snapshot_2024-06-30.csv"
test_file = OUT / "temporal_test_snapshot_2025-06-30.csv"
pred_file = OUT / "two_stage_revenue_predictions.csv"

for path in [train_file, test_file, pred_file]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
pred = pd.read_csv(pred_file)

target = "target_future_90d_revenue"

# Explicit list of features permitted for modelling.
features = [
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

print("=" * 65)
print("TWO-STAGE MODEL AUDIT")
print("=" * 65)

print("\n1. Snapshot columns")
print("Train shape:", train.shape)
print("Test shape :", test.shape)

unexpected = [
    c for c in features
    if c not in train.columns or c not in test.columns
]
print("Expected features missing from either snapshot:", unexpected)

print("\n2. Check for suspicious feature names")
suspicious_terms = [
    "target", "churn_date", "churn_reason",
    "end_date", "future", "outcome"
]

suspicious = [
    c for c in train.columns
    if any(term in c.lower() for term in suspicious_terms)
]

print("Potentially sensitive/outcome-related columns:")
for c in suspicious:
    print(" -", c)

print("\n3. Historical revenue versus target")
if target not in train.columns or target not in test.columns:
    raise ValueError(f"Missing target column: {target}")

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n{name}")
    print("Target summary:")
    print(df[target].describe().to_string())

    for feature in [
        "revenue_30d", "revenue_60d", "revenue_90d",
        "historical_successful_revenue"
    ]:
        if feature in df.columns:
            corr = df[[feature, target]].corr().iloc[0, 1]
            print(f"Correlation {feature} vs future target: {corr:.4f}")

print("\n4. Validate prediction file")
actual_col = "actual_future_90d_revenue"
prob_col = "predicted_positive_probability"
flag_col = "predicted_positive_revenue"
stage2_col = "stage2_predicted_amount"
combined_col = "two_stage_predicted_revenue"

required = [
    actual_col, prob_col, flag_col, stage2_col, combined_col
]
missing = [c for c in required if c not in pred.columns]

if missing:
    raise ValueError(f"Missing prediction columns: {missing}")

print("Prediction rows:", len(pred))
print("Prediction probability range:",
      pred[prob_col].min(), "to", pred[prob_col].max())

print("Predicted positive-revenue share:",
      f"{pred[flag_col].mean():.2%}")

print("Actual positive-revenue share:",
      f"{(pred[actual_col] > 0).mean():.2%}")

print("Stage 2 amount summary:")
print(pred[stage2_col].describe().to_string())

print("Combined prediction summary:")
print(pred[combined_col].describe().to_string())

# Confirm that Stage 1's zero predictions produce zero combined revenue.
wrong_zero_predictions = (
    (pred[flag_col] == 0)
    & (pred[combined_col].abs() > 1e-9)
).sum()

print("Rows where Stage 1 predicts zero but combined prediction is nonzero:",
      int(wrong_zero_predictions))


# Compare saved actual targets with the temporal test snapshot.
print("\n5. Match predictions with test snapshot")

if "customer_id" in pred.columns and "customer_id" in test.columns:
    actual_from_test = test[["customer_id", target]].copy()
    actual_from_test = actual_from_test.rename(
        columns={target: "snapshot_actual_revenue"}
    )

    joined = pred.merge(
        actual_from_test,
        on="customer_id",
        how="left",
        validate="one_to_one"
    )

    unmatched = joined["snapshot_actual_revenue"].isna().sum()
    print("Prediction rows without matching test customer:", int(unmatched))

    comparison = (
        joined[actual_col] - joined["snapshot_actual_revenue"]
    ).abs()

    valid_comparison = comparison.dropna()

    if not valid_comparison.empty:
        print("Maximum actual-target mismatch:",
              float(valid_comparison.max()))
        print("Rows with actual-target mismatch:",
              int((valid_comparison > 1e-9).sum()))
    else:
        print("No matching actual targets available for comparison.")
else:
    print("Cannot compare: customer_id is missing from a file.")

print("\nAudit finished. No files were modified.")
