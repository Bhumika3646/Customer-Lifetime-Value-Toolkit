
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"

def load_csv(filename, columns):
    path = RAW_DIR / filename
    df = pd.read_csv(path, header=None, names=columns)
    return df

subscriptions = load_csv(
    "subscriptions.csv",
    ["subscription_id", "customer_id", "start_date",
     "end_date", "status", "plan_id"]
)

invoices = load_csv(
    "invoices.csv",
    ["invoice_id", "subscription_id", "invoice_date",
     "due_date", "amount", "status"]
)

plans = load_csv(
    "subscription_plans.csv",
    ["plan_id", "plan_name", "billing_period", "price"]
)

# Attach each subscription's plan and each invoice's date.
data = invoices.merge(
    subscriptions[["subscription_id", "plan_id"]],
    on="subscription_id",
    how="left"
)

data = data.merge(
    plans[["plan_id", "plan_name", "billing_period"]],
    on="plan_id",
    how="left"
)

data["invoice_date"] = pd.to_datetime(
    data["invoice_date"], errors="coerce"
)

data["year"] = data["invoice_date"].dt.year

annual = data[
    data["billing_period"].astype(str).str.lower().str.contains(
        "annual|year", na=False
    )
].copy()

print("Annual invoices by year:")
print(
    annual.groupby("year").agg(
        invoice_rows=("invoice_id", "count"),
        subscriptions=("subscription_id", "nunique"),
        total_amount=("amount", "sum")
    ).to_string()
)

print("\nAnnual invoice records in 2025:")
print(
    annual[annual["year"] == 2025][
        ["invoice_date", "plan_name", "amount", "status"]
    ].head(20).to_string(index=False)
)

print("\nAudit complete. No files were modified.")
