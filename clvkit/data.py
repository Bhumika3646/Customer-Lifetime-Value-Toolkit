"""Loading, auditing and assembling the raw subscription tables."""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_tables(raw_dir=C.RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the headerless CSVs, assign column names, parse dates, normalise text."""
    tables = {}
    for key, (filename, columns) in C.SCHEMAS.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Put the raw CSVs in {raw_dir}.")
        df = pd.read_csv(path, header=None, names=columns)
        for col in C.DATE_COLUMNS.get(key, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
        tables[key] = df
    tables["payments"] = tables["payments"].drop(columns=["extra_blank"])
    return tables


def build_subscribers(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per subscription with static attributes and the FINAL plan.

    Columns ``end_date``, ``status`` and ``plan_id_final`` describe the whole
    observed history. They are used ONLY to construct survival labels after
    censoring at a cutoff; they must never be used as model features.
    """
    subs = tables["subscriptions"].rename(columns={"plan_id": "plan_id_final",
                                                   "status": "final_status"})
    out = subs.merge(tables["customers"], on="customer_id", how="left",
                     validate="one_to_one")
    out["churn_date"] = out["end_date"].where(out["final_status"] == "churned")
    return out


def plan_at(subs: pd.DataFrame, changes: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Plan id a subscription held on ``cutoff`` (undoes plan changes made after it)."""
    plan = subs.set_index("subscription_id")["plan_id_final"].copy()
    later = changes[changes["change_date"] > cutoff].sort_values("change_date")
    earliest = later.groupby("subscription_id")["old_plan_id"].first()
    plan.loc[earliest.index] = earliest.values
    return plan


def eligible_at(subs: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Subscriptions that exist and have not churned on or before ``cutoff``."""
    started = subs["start_date"] <= cutoff
    already_churned = subs["churn_date"].notna() & (subs["churn_date"] <= cutoff)
    return subs[started & ~already_churned].copy()


def audit_tables(tables: dict[str, pd.DataFrame]) -> dict:
    """Integrity checks printed in the report. Returns plain python values."""
    cust, subs, pay = tables["customers"], tables["subscriptions"], tables["payments"]
    inv, plans, churn = tables["invoices"], tables["plans"], tables["churn"]
    sub = build_subscribers(tables).merge(plans, left_on="plan_id_final",
                                          right_on="plan_id", how="left")

    paid_pay = pay[pay["status"] == "successful"]
    last_any = pay.groupby("subscription_id")["payment_date"].max()
    churned_monthly_paid = sub[(sub["final_status"] == "churned")
                               & (sub["billing_period"] == "monthly")
                               & (sub["price"] > 0)].set_index("subscription_id")
    lag = (churned_monthly_paid["end_date"] - last_any.reindex(churned_monthly_paid.index)).dt.days
    annual_pay_by_year = (pay.merge(sub[["subscription_id", "billing_period"]], on="subscription_id")
                          .query("billing_period == 'annual'")
                          .groupby(pay["payment_date"].dt.year).size().to_dict())

    return {
        "rows": {k: int(len(v)) for k, v in tables.items()},
        "one_subscription_per_customer": bool(subs["customer_id"].is_unique),
        "all_subscription_customers_exist": bool(subs["customer_id"].isin(cust["customer_id"]).all()),
        "all_payment_subscriptions_exist": bool(pay["subscription_id"].isin(subs["subscription_id"]).all()),
        "churn_events_match_churned_subscriptions": bool(
            set(churn["customer_id"]) == set(subs.loc[subs["status"] == "churned", "customer_id"])),
        "end_date_equals_churn_event_date": bool(
            subs[subs["status"] == "churned"].merge(churn, on="customer_id")
            .eval("end_date == churn_date").all()),
        "invoice_amount_equals_payment_amount": bool(
            (inv["amount"].values == pay["amount"].values).all()) if len(inv) == len(pay) else False,
        "invoices_all_marked_paid": bool((inv["status"] == "paid").all()),
        "payment_failure_rate": float((pay["status"] == "failed").mean()),
        "failed_payments_on_paid_invoices": int((pay["status"] == "failed").sum()),
        "share_free_plan": float((sub["price"] == 0).mean()),
        "share_churned": float((sub["final_status"] == "churned").mean()),
        "last_payment_date": str(pay["payment_date"].max().date()),
        "annual_payments_by_year": {int(k): int(v) for k, v in annual_pay_by_year.items()},
        "days_between_last_payment_and_churn_date_mean": float(lag.mean()),
        "days_between_last_payment_and_churn_date_min": float(lag.min()),
        "successful_payment_rows": int(len(paid_pay)),
        "is_synthetic_warning": ("Dataset appears synthetic (near-uniform categorical fields, "
                                 "plan changes only Free->Basic). Findings do not establish "
                                 "real-world validity."),
    }
