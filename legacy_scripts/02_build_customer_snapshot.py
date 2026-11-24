"""
Step 2: Build a leakage-aware 90-day customer snapshot.

Run in the same folder as:
  - archive (8).zip
  - 02_build_customer_snapshot.py

Command:
  python 02_build_customer_snapshot.py

This is a baseline snapshot, not the final model. Column names are inferred
from the headerless CSVs and must be checked against the Kaggle documentation.
"""

from pathlib import Path
import zipfile
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "archive (8).zip"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CUTOFF = pd.Timestamp("2025-06-30")
HORIZON_DAYS = 90
HORIZON_END = CUTOFF + pd.Timedelta(days=HORIZON_DAYS)

SCHEMAS = {
    "customers.csv": ["customer_id", "signup_date", "company_size", "industry", "acquisition_channel", "country"],
    "subscriptions.csv": ["subscription_id", "customer_id", "start_date", "end_date", "status", "plan_id"],
    "invoices.csv": ["invoice_id", "subscription_id", "invoice_date", "due_date", "amount", "status"],
    "payments.csv": ["payment_id", "subscription_id", "payment_date", "amount", "status", "extra_blank"],
    "subscription_plans.csv": ["plan_id", "plan_name", "billing_period", "price"],
    "Churn_events.csv": ["event_id", "customer_id", "churn_date", "reason"],
    "plan_chnages.csv": ["change_id", "subscription_id", "old_plan_id", "new_plan_id", "change_date"],
}
DATE_COLUMNS = {
    "customers.csv": ["signup_date"],
    "subscriptions.csv": ["start_date", "end_date"],
    "invoices.csv": ["invoice_date", "due_date"],
    "payments.csv": ["payment_date"],
    "Churn_events.csv": ["churn_date"],
    "plan_chnages.csv": ["change_date"],
}

def load_tables():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Cannot find {ZIP_PATH.name} beside this script.")
    tables = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = set(zf.namelist())
        for filename, columns in SCHEMAS.items():
            if filename not in names:
                raise FileNotFoundError(f"ZIP does not contain {filename}")
            df = pd.read_csv(zf.open(filename), header=None, names=columns)
            for col in DATE_COLUMNS.get(filename, []):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            tables[filename] = df
    return tables

def main():
    t = load_tables()
    customers = t["customers.csv"].copy()
    subs = t["subscriptions.csv"].copy()
    payments = t["payments.csv"].copy()
    plans = t["subscription_plans.csv"].copy()
    churn = t["Churn_events.csv"].copy()
    changes = t["plan_chnages.csv"].copy()

    # Keep customers who had signed up and had a subscription by the cutoff.
    base = customers.merge(
        subs[["subscription_id", "customer_id", "start_date", "end_date", "status", "plan_id"]],
        on="customer_id", how="inner", validate="one_to_one"
    )
    base = base[(base["signup_date"] <= CUTOFF) & (base["start_date"] <= CUTOFF)].copy()

    # Determine whether a churn event had already happened by the cutoff.
    churn_dates = churn.groupby("customer_id", as_index=False)["churn_date"].min()
    base = base.merge(churn_dates, on="customer_id", how="left", validate="one_to_one")
    base["churned_before_or_at_cutoff"] = base["churn_date"].notna() & (base["churn_date"] <= CUTOFF)
    base = base[~base["churned_before_or_at_cutoff"]].copy()

    # Reconstruct the plan at cutoff where a plan change happened after cutoff.
    # The final subscriptions table may reflect a later plan than the one at cutoff.
    plan_at_cutoff = {}
    for row in base[["subscription_id", "plan_id"]].itertuples(index=False):
        plan_at_cutoff[row.subscription_id] = row.plan_id
    future_changes = changes[changes["change_date"] > CUTOFF].sort_values("change_date")
    for row in future_changes.itertuples(index=False):
        if row.subscription_id in plan_at_cutoff:
            plan_at_cutoff[row.subscription_id] = row.old_plan_id
    base["plan_id_at_cutoff"] = base["subscription_id"].map(plan_at_cutoff)
    base = base.merge(
        plans.rename(columns={"plan_id": "plan_id_at_cutoff"}),
        on="plan_id_at_cutoff", how="left", validate="many_to_one"
    )

    # Historical successful payments only; failed payments are not counted as revenue.
    successful = payments[payments["status"].astype(str).str.lower().eq("successful")].copy()
    historical = successful[successful["payment_date"] <= CUTOFF].copy()
    for days in (30, 60, 90):
        lower = CUTOFF - pd.Timedelta(days=days)
        w = historical[(historical["payment_date"] > lower) & (historical["payment_date"] <= CUTOFF)]
        agg = w.groupby("subscription_id").agg(
            **{f"successful_payments_{days}d": ("payment_id", "count"),
               f"revenue_{days}d": ("amount", "sum")}
        ).reset_index()
        base = base.merge(agg, left_on="subscription_id", right_on="subscription_id", how="left", validate="one_to_one")

    hist_agg = historical.groupby("subscription_id").agg(
        historical_successful_payment_count=("payment_id", "count"),
        historical_successful_revenue=("amount", "sum"),
        last_successful_payment_date=("payment_date", "max"),
    ).reset_index()
    base = base.merge(hist_agg, on="subscription_id", how="left", validate="one_to_one")
    base["days_since_last_successful_payment"] = (CUTOFF - base["last_successful_payment_date"]).dt.days
    base["days_since_last_successful_payment"] = base["days_since_last_successful_payment"].fillna(999)
    base["historical_successful_payment_count"] = base["historical_successful_payment_count"].fillna(0).astype(int)
    base["historical_successful_revenue"] = base["historical_successful_revenue"].fillna(0.0)

    # Historical plan-change count. No change events after cutoff are used as features.
    past_changes = changes[changes["change_date"] <= CUTOFF]
    change_counts = past_changes.groupby("subscription_id").size().rename("plan_changes_before_cutoff")
    base = base.merge(change_counts, on="subscription_id", how="left", validate="one_to_one")
    base["plan_changes_before_cutoff"] = base["plan_changes_before_cutoff"].fillna(0).astype(int)

    # Targets use only the future 90-day period after the snapshot cutoff.
    future_payments = successful[
        (successful["payment_date"] > CUTOFF) & (successful["payment_date"] <= HORIZON_END)
    ]
    future_revenue = future_payments.groupby("subscription_id")["amount"].sum().rename("target_future_90d_revenue")
    base = base.merge(future_revenue, on="subscription_id", how="left", validate="one_to_one")
    base["target_future_90d_revenue"] = base["target_future_90d_revenue"].fillna(0.0)

    base["target_churn_within_90d"] = (
        base["churn_date"].notna()
        & (base["churn_date"] > CUTOFF)
        & (base["churn_date"] <= HORIZON_END)
    ).astype(int)

    # Remove fields that reveal final/future outcomes from the model table.
    # Keep IDs for tracing only; do not include customer_id/subscription_id as model features.
    drop_cols = [
        "end_date", "status", "churn_date", "reason", "event_id",
        "churned_before_or_at_cutoff", "last_successful_payment_date",
        "extra_blank", "plan_id", "plan_id_at_cutoff"
    ]
    # reason/event_id are not normally in base; dropping with errors='ignore' is intentional.
    snapshot = base.drop(columns=drop_cols, errors="ignore")
    snapshot.insert(0, "snapshot_cutoff_date", CUTOFF.date().isoformat())
    snapshot["target_horizon_end_date"] = HORIZON_END.date().isoformat()

    output_path = OUTPUT_DIR / "customer_snapshot_2025-06-30.csv"
    snapshot.to_csv(output_path, index=False)

    summary_path = OUTPUT_DIR / "snapshot_summary.txt"
    summary = [
        "CUSTOMER SNAPSHOT SUMMARY",
        "=" * 35,
        f"Snapshot cutoff: {CUTOFF.date()}",
        f"Future target window: {(CUTOFF + pd.Timedelta(days=1)).date()} through {HORIZON_END.date()} (inclusive)",
        f"Eligible customers at cutoff (not already churned): {len(snapshot):,}",
        f"Customers with churn label 1 in next 90 days: {int(snapshot['target_churn_within_90d'].sum()):,}",
        f"Customers with churn label 0 in next 90 days: {int((snapshot['target_churn_within_90d'] == 0).sum()):,}",
        f"Mean future 90-day successful-payment revenue: {snapshot['target_future_90d_revenue'].mean():.2f}",
        f"Median future 90-day successful-payment revenue: {snapshot['target_future_90d_revenue'].median():.2f}",
        f"Share with zero future 90-day successful-payment revenue: {(snapshot['target_future_90d_revenue'] == 0).mean():.2%}",
        "",
        "Important limitations:",
        "- Targets are based on successful payments only, not invoice status.",
        "- This is one fixed-cutoff baseline snapshot; later work should test additional cutoffs if the available timeline permits.",
        "- Features must exclude customer_id/subscription_id, target columns, and all post-cutoff information.",
        "- Column names are inferred from headerless CSVs and must be verified against Kaggle documentation.",
        "- Dataset is synthetic; performance does not establish real-world validity.",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"\nSaved snapshot: {output_path}")
    print(f"Saved summary:  {summary_path}")
    print("\nNext: review the summary before training models.")

if __name__ == "__main__":
    main()
