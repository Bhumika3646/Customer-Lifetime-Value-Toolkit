"""Survival-based customer lifetime value.

For a customer who is active on ``as_of`` with tenure ``t0`` days:

    CLV = sum_j  price * q * P(pays cycle j | active at t0) / (1 + r)^((tau_j - t0)/365.25)

* ``q``       share of billing attempts that succeed (estimated from past payments)
* ``tau_j``   tenure (days) of the j-th future billing date
* ``r``       annual discount rate (0 for undiscounted forecasts)

Billing rule (verified on this dataset, see audit): the recorded churn date is the
END of the last paid billing cycle (mean gap between last payment and churn date
= one billing period). Therefore a payment due at tenure ``tau`` is made if the
customer is still active at the MIDDLE of that cycle, ``tau + period / 2``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .survival import conditional_survival


def payment_success_rate(tables: dict, as_of) -> dict[str, float]:
    """Share of successful payment attempts by billing period, using rows <= as_of."""
    as_of = pd.Timestamp(as_of)
    pay = tables["payments"]
    pay = pay[pay["payment_date"] <= as_of]
    plans = tables["plans"].set_index("plan_id")["billing_period"]
    subs = tables["subscriptions"].set_index("subscription_id")["plan_id"]
    billing = pay["subscription_id"].map(subs).map(plans)
    rates = (pay["status"] == "successful").groupby(billing).mean().to_dict()
    overall = float((pay["status"] == "successful").mean())
    return {"monthly": rates.get("monthly", overall), "annual": rates.get("annual", overall)}


def expected_cashflows(model, snap: pd.DataFrame, horizon_months: float,
                       success_rate: dict[str, float], annual_discount: float = 0.0
                       ) -> pd.DataFrame:
    """Expected (discounted) revenue over the next ``horizon_months`` per customer."""
    n = len(snap)
    t0 = snap["tenure_days"].to_numpy(float)
    price = snap["plan_price"].to_numpy(float)
    if "has_billing_record" in snap:            # no billing evidence -> no expected revenue
        price = price * snap["has_billing_record"].to_numpy(float)
    period = snap["billing_period"].map(C.BILLING_PERIOD_DAYS).to_numpy(float)
    q = snap["billing_period"].map(success_rate).to_numpy(float)
    horizon_days = horizon_months * C.DAYS_PER_MONTH

    first_tau = (np.floor(t0 / period) + 1.0) * period
    n_slots = int(np.ceil(horizon_days / period.min())) + 1
    revenue = np.zeros(n)
    disc_revenue = np.zeros(n)
    exp_payments = np.zeros(n)
    for j in range(n_slots):
        tau = first_tau + j * period
        due = tau <= t0 + horizon_days
        if not due.any():
            break
        p_pay = conditional_survival(model, snap, t0, tau + period / 2.0)
        p_pay = np.where(due, p_pay, 0.0)
        cash = price * q * p_pay
        revenue += cash
        exp_payments += q * p_pay
        disc_revenue += cash / (1.0 + annual_discount) ** ((tau - t0) / C.DAYS_PER_YEAR)
    return pd.DataFrame({"expected_revenue": revenue, "clv": disc_revenue,
                         "expected_payments": exp_payments}, index=snap.index)


def retention_probability(model, snap: pd.DataFrame, horizon_months: float) -> np.ndarray:
    """P(customer still has not churned ``horizon_months`` from now | active now)."""
    t0 = snap["tenure_days"].to_numpy(float)
    return conditional_survival(model, snap, t0, t0 + horizon_months * C.DAYS_PER_MONTH)


def expected_active_months(model, snap: pd.DataFrame, horizon_months: int) -> np.ndarray:
    """Expected number of months (out of ``horizon_months``) the customer stays active."""
    total = np.zeros(len(snap))
    for m in range(1, horizon_months + 1):
        total += retention_probability(model, snap, m)
    return total


def naive_no_churn_revenue(snap: pd.DataFrame, horizon_months: float) -> np.ndarray:
    """Benchmark: price x number of billing dates in the window, assuming nobody churns."""
    t0 = snap["tenure_days"].to_numpy(float)
    period = snap["billing_period"].map(C.BILLING_PERIOD_DAYS).to_numpy(float)
    horizon_days = horizon_months * C.DAYS_PER_MONTH
    first_tau = (np.floor(t0 / period) + 1.0) * period
    n_pay = np.floor((t0 + horizon_days - first_tau) / period) + 1.0
    return snap["plan_price"].to_numpy(float) * np.clip(n_pay, 0, None)  # intentionally naive


def unbilled_billing_sensitivity(
    model,
    snap: pd.DataFrame,
    success_rate: dict[str, float],
    base_clv_total: float,
    horizon_months: float = C.CLV_HORIZON_MONTHS,
    annual_discount: float = C.ANNUAL_DISCOUNT_RATE,
) -> pd.DataFrame:
    """Quantify the impact of active paid-plan records with no billing history.

    The current dashboard treats this cohort conservatively as zero CLV. The
    alternative rows are transparent sensitivity scenarios, NOT estimates of
    what those customers actually paid and NOT confidence intervals. The
    observed-rate scenario assumes they were potentially billable and applies
    the observed billing-period success rate plus the fitted survival profile
    for comparable plan/billing-period customers. The stress case assumes
    100% payment success.
    """
    flagged = (pd.to_numeric(snap["plan_price"], errors="coerce").fillna(0) > 0) & (
        pd.to_numeric(snap["has_billing_record"], errors="coerce").fillna(0) == 0
    )
    cohort = snap.loc[flagged].copy()
    n_flagged = int(len(cohort))
    base_total = float(base_clv_total)

    def score_cohort(q: dict[str, float]) -> float:
        if cohort.empty:
            return 0.0
        counterfactual = cohort.copy()
        counterfactual["has_billing_record"] = 1
        return float(expected_cashflows(
            model, counterfactual, horizon_months, q, annual_discount
        )["clv"].sum())

    observed_rate_increment = score_cohort(success_rate)
    perfect_rate_increment = score_cohort({"monthly": 1.0, "annual": 1.0})
    rows = [
        {
            "scenario": "Conservative: no billing evidence -> zero CLV",
            "flagged_paid_plan_subscriptions": n_flagged,
            "payment_success_assumption": "No billing record; assigned zero",
            "estimated_clv_for_flagged_cohort": 0.0,
            "portfolio_total_clv_scenario": base_total,
            "change_vs_conservative_pct": 0.0,
        },
        {
            "scenario": "Sensitivity: potentially billable at observed billing-period success rates",
            "flagged_paid_plan_subscriptions": n_flagged,
            "payment_success_assumption": "Observed monthly/annual payment success rates",
            "estimated_clv_for_flagged_cohort": observed_rate_increment,
            "portfolio_total_clv_scenario": base_total + observed_rate_increment,
            "change_vs_conservative_pct": (100 * observed_rate_increment / base_total if base_total else np.nan),
        },
        {
            "scenario": "Stress case: potentially billable with 100% payment success",
            "flagged_paid_plan_subscriptions": n_flagged,
            "payment_success_assumption": "100% success; intentionally optimistic stress case",
            "estimated_clv_for_flagged_cohort": perfect_rate_increment,
            "portfolio_total_clv_scenario": base_total + perfect_rate_increment,
            "change_vs_conservative_pct": (100 * perfect_rate_increment / base_total if base_total else np.nan),
        },
    ]
    return pd.DataFrame(rows)
