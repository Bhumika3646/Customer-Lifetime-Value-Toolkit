
from pathlib import Path
import pandas as pd

ROOT = Path(".")
OUTPUT = ROOT / "outputs"
RAW = ROOT / "raw_data"

test_path = OUTPUT / "temporal_test_snapshot_2025-06-30.csv"

# The payments.csv file has no header.
payments = pd.read_csv(
    RAW / "payments.csv",
    header=None,
    names=[
        "payment_id",
        "subscription_id",
        "payment_date",
        "amount",
        "status",
        "extra_blank",
    ],
)

test = pd.read_csv(test_path)

payments["subscription_id"] = payments["subscription_id"].astype(str).str.strip()
test["subscription_id"] = test["subscription_id"].astype(str).str.strip()

payments["payment_date"] = pd.to_datetime(
    payments["payment_date"], errors="coerce"
)
payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce")

cutoff = pd.Timestamp("2025-06-30")
horizon_end = pd.Timestamp("2025-09-28")

annual = test[
    test["billing_period_at_cutoff"].astype(str).str.lower() == "annual"
].copy()

annual_ids = set(annual["subscription_id"])

future = payments[
    payments["subscription_id"].isin(annual_ids)
    & (payments["payment_date"] > cutoff)
    & (payments["payment_date"] <= horizon_end)
].copy()

print("=" * 65)
print("ANNUAL PAYMENT RECORD CHECK")
print("=" * 65)
print(f"Annual customers in test snapshot: {len(annual):,}")
print(f"Annual customers with any payment in target window: "
      f"{future['subscription_id'].nunique():,}")

print("\nPayment statuses in target window:")
if future.empty:
    print("No payment records found in this window.")
else:
    print(future["status"].value_counts(dropna=False).to_string())

print("\nPayments by month:")
if future.empty:
    print("No payments to summarize.")
else:
    future["month"] = future["payment_date"].dt.to_period("M").astype(str)
    print(
        future.groupby("month")
        .agg(
            payment_records=("payment_id", "count"),
            subscriptions=("subscription_id", "nunique"),
            amount=("amount", "sum"),
        )
        .round(2)
        .to_string()
    )

print("\nAll recorded annual-customer payments by year:")
annual_history = payments[
    payments["subscription_id"].isin(annual_ids)
    & payments["payment_date"].notna()
].copy()

if annual_history.empty:
    print("No dated payment records found.")
else:
    annual_history["year"] = annual_history["payment_date"].dt.year
    print(
        annual_history.groupby("year")
        .agg(
            payment_records=("payment_id", "count"),
            subscriptions=("subscription_id", "nunique"),
            amount=("amount", "sum"),
        )
        .round(2)
        .to_string()
    )

print("\nCheck complete. No data was modified.")
