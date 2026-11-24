
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"

# Dataset columns, matching 05_temporal_validation.py
subscriptions = pd.read_csv(
    RAW_DIR / "subscriptions.csv",
    header=None,
    names=[
        "subscription_id", "customer_id", "start_date",
        "end_date", "status", "plan_id",
    ],
    dtype={"subscription_id": "string", "customer_id": "string",
          "plan_id": "string"},
)

payments = pd.read_csv(
    RAW_DIR / "payments.csv",
    header=None,
    names=[
        "payment_id", "subscription_id", "payment_date",
        "amount", "status", "extra_blank",
    ],
    dtype={"subscription_id": "string"},
)

plans = pd.read_csv(
    RAW_DIR / "subscription_plans.csv",
    header=None,
    names=["plan_id", "plan_name", "billing_period", "price"],
    dtype={"plan_id": "string"},
)

payments["payment_date"] = pd.to_datetime(
    payments["payment_date"], errors="coerce"
)
subscriptions["start_date"] = pd.to_datetime(
    subscriptions["start_date"], errors="coerce"
)
payments["amount"] = pd.to_numeric(
    payments["amount"], errors="coerce"
)
payments["status"] = (
    payments["status"].astype("string").str.strip().str.lower()
)

successful_statuses = {
    "success", "successful", "succeeded", "paid", "completed"
}
successful = payments[
    payments["status"].isin(successful_statuses)
    & payments["payment_date"].notna()
].copy()

# Attach each subscription's stored plan.
sub_plan = subscriptions[
    ["subscription_id", "customer_id", "plan_id", "start_date"]
].merge(
    plans,
    on="plan_id",
    how="left",
    validate="many_to_one",
)

successful = successful.merge(
    sub_plan,
    on="subscription_id",
    how="inner",
    validate="many_to_one",
)

windows = [
    ("TRAIN", pd.Timestamp("2024-06-30"), pd.Timestamp("2024-09-28")),
    ("TEST", pd.Timestamp("2025-06-30"), pd.Timestamp("2025-09-28")),
]

print("=" * 70)
print("RAW SUCCESSFUL PAYMENTS BY BILLING PERIOD")
print("=" * 70)
print("All successful payments:", len(successful))
print("Payment date range:",
      successful["payment_date"].min(),
      "to", successful["payment_date"].max())

for label, cutoff, horizon_end in windows:
    window = successful[
        (successful["payment_date"] > cutoff)
        & (successful["payment_date"] <= horizon_end)
    ].copy()

    print(f"\n{label}: {cutoff.date()} through {horizon_end.date()}")
    print("Successful payment rows:", len(window))

    summary = (
        window.groupby(
            ["billing_period", "plan_name", "price"],
            dropna=False,
        )
        .agg(
            payment_rows=("payment_id", "count"),
            subscriptions=("subscription_id", "nunique"),
            revenue=("amount", "sum"),
        )
        .reset_index()
        .sort_values(["billing_period", "plan_name"])
    )

    if summary.empty:
        print("No successful payments in this window.")
    else:
        print(summary.to_string(index=False))

print("\nAudit complete. No files were modified.")
