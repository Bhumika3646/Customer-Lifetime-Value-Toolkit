
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

plans = pd.read_csv(
    RAW_DIR / "subscription_plans.csv",
    header=None,
    names=["plan_id", "plan_name", "billing_period", "price"],
    dtype={"plan_id": "string"},
)

changes = pd.read_csv(
    RAW_DIR / "plan_chnages.csv",
    header=None,
    names=[
        "change_id", "subscription_id", "old_plan_id",
        "new_plan_id", "change_date",
    ],
    dtype={"subscription_id": "string"},
)

subscriptions["start_date"] = pd.to_datetime(
    subscriptions["start_date"], errors="coerce"
)
changes["change_date"] = pd.to_datetime(
    changes["change_date"], errors="coerce"
)

# Keep subscriptions that started by each cutoff.
for cutoff in [pd.Timestamp("2024-06-30"), pd.Timestamp("2025-06-30")]:
    eligible = subscriptions[
        subscriptions["start_date"].notna()
        & (subscriptions["start_date"] <= cutoff)
    ].copy()

    # Reconstruct plan as of cutoff, matching the current project logic.
    relevant = changes[
        changes["change_date"].isna()
        | (changes["change_date"] <= cutoff)
    ].copy()

    latest_changes = (
        relevant.sort_values("change_date")
        .drop_duplicates("subscription_id", keep="last")
    )

    plan_map = latest_changes.set_index(
        "subscription_id"
    )["new_plan_id"]

    eligible["plan_id_at_cutoff"] = (
        eligible["subscription_id"].map(plan_map)
    )

    # If no change occurred by cutoff, use the stored plan.
    eligible["plan_id_at_cutoff"] = (
        eligible["plan_id_at_cutoff"].fillna(eligible["plan_id"])
    )

    eligible = eligible.merge(
        plans,
        left_on="plan_id_at_cutoff",
        right_on="plan_id",
        how="left",
        validate="many_to_one",
    )

    print(f"\nPLAN DISTRIBUTION AT {cutoff.date()}")
    print("Eligible subscriptions:", len(eligible))

    print(
        eligible.groupby(
            ["billing_period", "plan_name"],
            dropna=False,
        ).size().to_string()
    )

print("\nAudit complete. No files were modified.")
