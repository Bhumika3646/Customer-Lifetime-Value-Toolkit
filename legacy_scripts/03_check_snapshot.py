
import pandas as pd

file_path = "outputs/customer_snapshot_2025-06-30.csv"
df = pd.read_csv(file_path)

print("=" * 55)
print("SNAPSHOT DIAGNOSTIC")
print("=" * 55)

columns = [
    "successful_payments_30d",
    "revenue_30d",
    "successful_payments_60d",
    "revenue_60d",
    "successful_payments_90d",
    "revenue_90d",
]

for col in columns:
    print(f"\n{col}")
    print("Missing:", df[col].isna().sum())
    print("Non-missing:", df[col].notna().sum())

print("\nMissingness patterns:")
print(
    df[columns]
    .isna()
    .value_counts()
    .to_string()
)

print("\nCustomers with missing 30-day payment count:")
print(
    df.loc[
        df["successful_payments_30d"].isna(),
        [
            "customer_id",
            "historical_successful_payment_count",
            "historical_successful_revenue",
            "days_since_last_successful_payment",
            "target_future_90d_revenue",
            "target_churn_within_90d",
        ],
    ].head(10).to_string(index=False)
)

print("\nTarget distribution:")
print(
    df["target_churn_within_90d"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nRevenue target summary:")
print(
    df["target_future_90d_revenue"]
    .describe()
    .to_string()
)

print("\nDiagnostic completed. No files were modified.")
