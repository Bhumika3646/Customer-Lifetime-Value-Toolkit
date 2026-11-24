"""Rule-based customer value/risk segments and suggested retention actions."""
from __future__ import annotations

import numpy as np
import pandas as pd

UNBILLED_FLAG = "paid_plan_without_billing_records"
# "High" means at or above this quantile among PAYING customers with billing evidence.
# The median is used because plan drives both value and risk in this dataset (value rises
# with plan tier, risk falls), so a top-quartile cut on both never intersects.
HIGH_QUANTILE = 0.5


def build_value_risk_segments(customer_clv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add 12-month churn risk, value/risk segment, and recommendation columns.

    High value and high risk are at/above the paying-customer median (HIGH_QUANTILE) among PAYING customers with
    billing evidence; Free-plan customers form their own group. Unbilled paid-plan records are placed in a separate
    data-review group instead of being treated as proven low-value customers.
    Returns (customer_level, segment_summary, priority_list).
    """
    required = {"subscription_id", "clv_36m", "retention_prob_12m", "plan_price", "has_billing_record"}
    missing = required - set(customer_clv.columns)
    if missing:
        raise ValueError(f"Customer CLV data is missing required fields: {sorted(missing)}")

    out = customer_clv.copy()
    out["clv_36m"] = pd.to_numeric(out["clv_36m"], errors="coerce").fillna(0).clip(lower=0)
    out["retention_prob_12m"] = pd.to_numeric(out["retention_prob_12m"], errors="coerce").clip(0, 1)
    out["estimated_12m_churn_risk"] = (1.0 - out["retention_prob_12m"]).clip(0, 1)
    out["plan_price"] = pd.to_numeric(out["plan_price"], errors="coerce").fillna(0)
    if "data_quality_flag" not in out.columns:
        out["data_quality_flag"] = ""
    out["data_quality_flag"] = out["data_quality_flag"].fillna("").astype(str)
    unbilled = out["data_quality_flag"].str.contains(UNBILLED_FLAG, regex=False)
    free = (out["plan_price"] <= 0) & ~unbilled

    # Thresholds are computed ONLY among paying customers with billing evidence.
    # (Using every customer put Free users - ~50% churn, zero CLV - in the risk
    # quartile and left the high-value / high-risk segment empty.)
    paying = out.loc[~unbilled & ~free]
    positives = paying.loc[paying["clv_36m"] > 0, "clv_36m"]
    value_threshold = float(positives.quantile(HIGH_QUANTILE)) if not positives.empty else float("inf")
    risk_distribution = paying["estimated_12m_churn_risk"].dropna()
    risk_threshold = float(risk_distribution.quantile(HIGH_QUANTILE)) if not risk_distribution.empty else float("inf")

    out["high_value_threshold"] = value_threshold
    out["high_risk_threshold"] = risk_threshold
    is_high_value = (out["clv_36m"] >= value_threshold) & (out["clv_36m"] > 0)
    is_high_risk = out["estimated_12m_churn_risk"] >= risk_threshold

    conditions = [
        unbilled,
        free,
        is_high_value & is_high_risk,
        is_high_value & ~is_high_risk,
        ~is_high_value & is_high_risk,
    ]
    labels = [
        "Billing data review",
        "Free plan (conversion candidates)",
        "High value / high risk",
        "High value / lower risk",
        "Lower value / high risk",
    ]
    out["value_risk_segment"] = np.select(conditions, labels, default="Lower value / lower risk")

    action_by_segment = {
        "Billing data review": "Verify billing history and plan-change records before assigning a retention priority.",
        "Free plan (conversion candidates)": "No revenue at risk; focus on onboarding and upgrade nudges. CLV is zero by construction (no conversion model).",
        "High value / high risk": "Prioritize proactive retention outreach; review recent payment failures and engagement decline before choosing an incentive.",
        "High value / lower risk": "Protect the relationship with renewal reminders, service-quality monitoring, and relevant loyalty engagement.",
        "Lower value / high risk": "Use cost-effective self-service support and investigate billing or engagement issues before offering incentives.",
        "Lower value / lower risk": "Use standard engagement and reassess the segment when payment or usage behaviour changes.",
    }
    out["retention_recommendation"] = out["value_risk_segment"].map(action_by_segment)

    segment_order = [
        "Billing data review", "Free plan (conversion candidates)", "High value / high risk",
        "High value / lower risk", "Lower value / high risk", "Lower value / lower risk",
    ]
    summary = (out.groupby("value_risk_segment", dropna=False)
               .agg(customers=("subscription_id", "size"),
                    total_estimated_clv_36m=("clv_36m", "sum"),
                    mean_estimated_clv_36m=("clv_36m", "mean"),
                    mean_estimated_12m_churn_risk=("estimated_12m_churn_risk", "mean"),
                    flagged_billing_records=("data_quality_flag", lambda x: int(x.str.contains(UNBILLED_FLAG, regex=False).sum())))
               .reindex(segment_order, fill_value=0)
               .rename_axis("value_risk_segment").reset_index())
    summary["recommended_action"] = summary["value_risk_segment"].map(action_by_segment)

    priority = out[out["value_risk_segment"].isin([
        "Billing data review", "High value / high risk", "Lower value / high risk"
    ])].copy()
    priority["_order"] = priority["value_risk_segment"].map(
        {"High value / high risk": 0, "Lower value / high risk": 1, "Billing data review": 2})
    priority = priority.sort_values(["_order", "clv_36m", "estimated_12m_churn_risk"],
                                    ascending=[True, False, False]).drop(columns="_order")
    return out, summary, priority
