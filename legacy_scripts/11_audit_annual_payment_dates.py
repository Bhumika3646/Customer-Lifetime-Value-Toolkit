
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"

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
payments["status"] = (
    payments["status"].astype("string").str.strip().str.lower()
)

successful = payments[
    payments["status"].isin(
        {"success", "successful", "succeeded", "paid", "completed"}
    )
    & payments["payment_date"].notna()
]

annual_ids = set(
    subscriptions.merge(
        plans,
        on="plan_id",
        how="left",
        validate="many_to_one",
    )
    .loc[
        lambda df: df["billing_period"].astype("string").str.lower()
        == "annual",
        "subscription_id",
    ]
    .dropna()
)

annual_payments = successful[
    successful["subscription_id"].isin(annual_ids)
].copy()

print("Annual subscriptions:", len(annual_ids))
print("Successful annual payment rows:", len(annual_payments))

print("\nAnnual payment dates by year:")
print(
    annual_payments.assign(
        year=annual_payments["payment_date"].dt.year
    )
    .groupby("year")
    .agg(
        payment_rows=("payment_id", "count"),
        subscriptions=("subscription_id", "nunique"),
    )
    .to_string()
)

print("\nLatest annual payment dates:")
print(
    annual_payments.sort_values("payment_date")[
        ["subscription_id", "payment_date", "status"]
    ].tail(15).to_string(index=False)
)

print("\nAnnual payments in each target window:")
for label, start, end in [
    ("TRAIN", "2024-06-30", "2024-09-28"),
    ("TEST", "2025-06-30", "2025-09-28"),
]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    window = annual_payments[
        (annual_payments["payment_date"] > start)
        & (annual_payments["payment_date"] <= end)
    ]
    print(
        f"{label}: {len(window)} payment rows, "
        f"{window['subscription_id'].nunique()} subscriptions"
    )

print("\nAudit complete. No files were modified.")
