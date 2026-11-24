
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"

invoices = pd.read_csv(
    RAW_DIR / "invoices.csv",
    header=None,
    names=[
        "invoice_id", "subscription_id", "invoice_date",
        "due_date", "amount", "status"
    ]
)

subscriptions = pd.read_csv(
    RAW_DIR / "subscriptions.csv",
    header=None,
    names=[
        "subscription_id", "customer_id", "start_date",
        "end_date", "status", "plan_id"
    ]
)

plans = pd.read_csv(
    RAW_DIR / "subscription_plans.csv",
    header=None,
    names=["plan_id", "plan_name", "billing_period", "price"]
)

data = invoices.merge(
    subscriptions[["subscription_id", "plan_id"]],
    on="subscription_id",
    how="left"
).merge(
    plans[["plan_id", "plan_name", "billing_period"]],
    on="plan_id",
    how="left"
)

data["invoice_date"] = pd.to_datetime(
    data["invoice_date"], errors="coerce"
)
data["month"] = data["invoice_date"].dt.to_period("M").astype(str)

print("Invoice records by month and billing period:")
summary = data.groupby(
    ["month", "billing_period"], dropna=False
).agg(
    invoice_rows=("invoice_id", "count"),
    subscriptions=("subscription_id", "nunique"),
    total_amount=("amount", "sum")
).reset_index()

print(summary.to_string(index=False))

print("\nInvoice statuses:")
print(data.groupby(["billing_period", "status"], dropna=False).size().to_string())

print("\nAudit complete. No files were modified.")
