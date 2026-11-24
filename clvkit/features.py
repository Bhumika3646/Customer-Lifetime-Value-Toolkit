"""Leakage-safe snapshot builder: features from <= cutoff, targets from > cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import build_subscribers, eligible_at, plan_at

CATEGORICAL = ["company_size", "industry", "acquisition_channel", "country",
               "plan_name", "billing_period"]
NUMERIC = [
    "plan_price", "tenure_days", "plan_changes_before_cutoff",
    "hist_success_count", "hist_success_revenue", "hist_failed_count",
    "pay_count_30d", "revenue_30d", "pay_count_60d", "revenue_60d",
    "pay_count_90d", "revenue_90d",
    "pay_count_prev30d", "revenue_prev30d", "pay_count_days61_90", "revenue_days61_90",
    "failed_count_30d", "failed_count_60d", "failed_count_90d",
    "failed_count_prev30d", "failed_count_days61_90",
    "revenue_delta_30d_vs_prev30d", "revenue_trend_ratio_30d_vs_prev30d",
    "revenue_delta_30d_vs_days61_90", "payment_count_delta_30d_vs_prev30d",
    "payment_count_trend_ratio_30d_vs_prev30d", "failed_count_delta_30d_vs_prev30d",
    "days_since_last_success", "has_billing_record",
]
FEATURE_COLUMNS = CATEGORICAL + NUMERIC
TARGET_REVENUE = "target_revenue"
TARGET_CHURN = "target_churn"

# Columns that describe the future or the final outcome. A guard test asserts
# none of these ever reach a model.
FORBIDDEN_FEATURES = {
    "status", "final_status", "end_date", "churn_date", "reason", "plan_id_final",
    "target_revenue", "target_churn", "subscription_id", "customer_id",
}


def assert_no_leakage(feature_columns) -> None:
    bad = FORBIDDEN_FEATURES.intersection(feature_columns)
    if bad:
        raise ValueError(f"Leakage guard: forbidden columns used as features: {sorted(bad)}")


def _window(succ: pd.DataFrame, cutoff: pd.Timestamp, days: int, prefix: str) -> pd.DataFrame:
    w = succ[(succ["payment_date"] > cutoff - pd.Timedelta(days=days))
             & (succ["payment_date"] <= cutoff)]
    g = w.groupby("subscription_id")["amount"].agg(["count", "sum"])
    g.columns = [f"pay_count_{prefix}", f"revenue_{prefix}"]
    return g


def build_snapshot(tables: dict, cutoff, horizon_days: int) -> pd.DataFrame:
    """One row per customer active at ``cutoff``.

    Features use only rows dated <= cutoff. Targets (revenue and churn in the
    following ``horizon_days``) come from rows dated > cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)

    subs = build_subscribers(tables)
    snap = eligible_at(subs, cutoff)
    snap["plan_id_cutoff"] = snap["subscription_id"].map(plan_at(subs, tables["changes"], cutoff))
    snap = snap.merge(tables["plans"].rename(columns={"plan_id": "plan_id_cutoff",
                                                      "price": "plan_price"}),
                      on="plan_id_cutoff", how="left", validate="many_to_one")
    snap["tenure_days"] = (cutoff - snap["start_date"]).dt.days

    pay = tables["payments"]
    succ = pay[pay["status"] == "successful"]
    fail = pay[pay["status"] == "failed"]
    hist_s = succ[succ["payment_date"] <= cutoff]
    hist_f = fail[fail["payment_date"] <= cutoff]

    agg = hist_s.groupby("subscription_id").agg(
        hist_success_count=("amount", "count"),
        hist_success_revenue=("amount", "sum"),
        last_success=("payment_date", "max"))
    snap = snap.merge(agg, on="subscription_id", how="left")
    snap = snap.merge(hist_f.groupby("subscription_id").size().rename("hist_failed_count"),
                      on="subscription_id", how="left")
    # Rolling historical windows. All rows are restricted to payment_date <= cutoff.
    for days, tag in ((30, "30d"), (60, "60d"), (90, "90d")):
        snap = snap.merge(_window(hist_s, cutoff, days, tag), on="subscription_id", how="left")
        fail_w = fail[(fail["payment_date"] > cutoff - pd.Timedelta(days=days))
                      & (fail["payment_date"] <= cutoff)]
        snap = snap.merge(
            fail_w.groupby("subscription_id").size().rename(f"failed_count_{tag}"),
            on="subscription_id", how="left")
    snap["days_since_last_success"] = (cutoff - snap["last_success"]).dt.days  # NaN if never
    ch = tables["changes"]
    snap = snap.merge(ch[ch["change_date"] <= cutoff].groupby("subscription_id").size()
                      .rename("plan_changes_before_cutoff"), on="subscription_id", how="left")
    # Fill base window counts/revenue before deriving deltas. These are observed-history
    # features only; subtracting nested windows creates non-overlapping 31-60d and 61-90d
    # periods without referencing future rows.
    base_window_cols = [
        "hist_success_count", "hist_success_revenue", "hist_failed_count",
        "pay_count_30d", "revenue_30d", "pay_count_60d", "revenue_60d",
        "pay_count_90d", "revenue_90d", "failed_count_30d", "failed_count_60d", "failed_count_90d",
        "plan_changes_before_cutoff",
    ]
    for col in base_window_cols:
        if col in snap.columns:
            snap[col] = snap[col].fillna(0)

    snap["pay_count_prev30d"] = (snap["pay_count_60d"] - snap["pay_count_30d"]).clip(lower=0)
    snap["revenue_prev30d"] = (snap["revenue_60d"] - snap["revenue_30d"]).clip(lower=0)
    snap["pay_count_days61_90"] = (snap["pay_count_90d"] - snap["pay_count_60d"]).clip(lower=0)
    snap["revenue_days61_90"] = (snap["revenue_90d"] - snap["revenue_60d"]).clip(lower=0)
    snap["failed_count_prev30d"] = (snap["failed_count_60d"] - snap["failed_count_30d"]).clip(lower=0)
    snap["failed_count_days61_90"] = (snap["failed_count_90d"] - snap["failed_count_60d"]).clip(lower=0)

    snap["revenue_delta_30d_vs_prev30d"] = snap["revenue_30d"] - snap["revenue_prev30d"]
    snap["revenue_trend_ratio_30d_vs_prev30d"] = (
        (snap["revenue_30d"] - snap["revenue_prev30d"])
        / (snap["revenue_30d"] + snap["revenue_prev30d"] + 1.0)
    )
    snap["revenue_delta_30d_vs_days61_90"] = snap["revenue_30d"] - snap["revenue_days61_90"]
    snap["payment_count_delta_30d_vs_prev30d"] = snap["pay_count_30d"] - snap["pay_count_prev30d"]
    snap["payment_count_trend_ratio_30d_vs_prev30d"] = (
        (snap["pay_count_30d"] - snap["pay_count_prev30d"])
        / (snap["pay_count_30d"] + snap["pay_count_prev30d"] + 1.0)
    )
    snap["failed_count_delta_30d_vs_prev30d"] = snap["failed_count_30d"] - snap["failed_count_prev30d"]

    # Fill the remaining numeric features while retaining NaN for days since last success,
    # plan price, and tenure so the comparison pipeline can impute those values consistently.
    fill0 = [c for c in NUMERIC if c in snap.columns and c not in (
        "days_since_last_success", "plan_price", "tenure_days", "has_billing_record")]
    snap[fill0] = snap[fill0].fillna(0)
    # Billing evidence: any payment attempt recorded on or before the cutoff. Customers on
    # a paid plan WITHOUT any billing record (e.g. the 787 Free->Basic upgrades, which have
    # no payments or invoices at all in this dataset) are flagged, not silently trusted.
    snap["has_billing_record"] = ((snap["hist_success_count"] + snap["hist_failed_count"]) > 0).astype(int)

    # ---- targets: strictly after the cutoff ----
    fut = succ[(succ["payment_date"] > cutoff) & (succ["payment_date"] <= horizon_end)]
    snap = snap.merge(fut.groupby("subscription_id")["amount"].sum().rename(TARGET_REVENUE),
                      on="subscription_id", how="left")
    snap[TARGET_REVENUE] = snap[TARGET_REVENUE].fillna(0.0)
    snap[TARGET_CHURN] = (snap["churn_date"].notna() & (snap["churn_date"] <= horizon_end)).astype(int)

    snap["cutoff"] = cutoff
    snap["horizon_days"] = horizon_days
    keep = ["subscription_id", "cutoff", "horizon_days"] + FEATURE_COLUMNS + [TARGET_REVENUE, TARGET_CHURN]
    snap = snap[keep].reset_index(drop=True)
    for col in CATEGORICAL:
        snap[col] = snap[col].astype("string")
    return snap
