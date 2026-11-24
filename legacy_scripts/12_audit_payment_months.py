
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"

payments = pd.read_csv(
    RAW_DIR / "payments.csv",
    header=None,
    names=[
        "payment_id", "subscription_id", "payment_date",
        "amount", "status", "extra_blank",
    ],
)

payments["payment_date"] = pd.to_datetime(
    payments["payment_date"], errors="coerce"
)
payments["status"] = (
    payments["status"].astype("string").str.strip().str.lower()
)

successful = payments[
    payments["status"].isin(
        {"success", "successful", "succeeded", "paid", "completed"}
    )
    & payments["payment_date"].notna()
].copy()

successful["month"] = successful["payment_date"].dt.to_period("M")

monthly_summary = (
    successful.groupby("month")
    .agg(
        payment_rows=("payment_id", "count"),
        subscriptions=("subscription_id", "nunique"),
        total_revenue=("amount", "sum"),
    )
    .reset_index()
)

print("Successful payments by month:")
print(monthly_summary.to_string(index=False))

print("\nAudit complete. No files were modified.")
