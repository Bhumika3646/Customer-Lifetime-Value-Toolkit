"""Uncertainty quantification and scenario simulation for survival-based CLV.

Three complementary tools (all built on the same cash-flow matrices as ``clv.py``):

1. **Bootstrap confidence intervals** - *parameter* uncertainty. Spells are resampled with
   replacement, the survival model is refitted, and the portfolio is re-scored. The spread of
   the replicates is the sampling uncertainty of the estimate given finite data.
2. **Monte Carlo simulation** - *process* uncertainty. Given fitted survival curves, every
   customer's churn time and every payment attempt are simulated, giving a distribution of
   realised revenue (customers churn or pay at random even if the model were perfect).
3. **Scenario grid** - deterministic what-ifs: churn-hazard multiplier, discount rate,
   horizon and payment-success rate.

Combining 1 and 2 (one or more simulations per bootstrap refit) gives a *predictive* interval
for the revenue that will actually be collected. None of these capture model misspecification,
future shocks (price changes, market conditions) or errors in the data itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .survival import CoxModel, SegmentKM


# --------------------------------------------------------------------------
# Cash-flow components
# --------------------------------------------------------------------------
def survival_matrix(model, snap: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """S(t) for every customer (rows) at several times each (``times`` is n x J)."""
    times = np.asarray(times, float)
    n, J = times.shape
    if isinstance(model, CoxModel):
        ph = model.cph.predict_partial_hazard(model._design(snap)).values.ravel()
        strata = snap[model.strata].astype(str).values
        h0 = np.zeros((n, J))
        known = np.isin(strata, list(model.h0))
        for stratum in model.h0:
            m = strata == stratum
            if m.any():
                h0[m] = model._h0(stratum, times[m].ravel()).reshape(-1, J)
        if (~known).any():
            h0[~known] = model._h0(next(iter(model.h0)), times[~known].ravel()).reshape(-1, J)
        return np.exp(-h0 * ph[:, None])
    out = np.empty((n, J))                        # KM models: evaluate column by column
    for j in range(J):
        out[:, j] = model.survival(snap, times[:, j])
    return out


def cashflow_components(model, snap: pd.DataFrame, horizon_months: float,
                        success_rate: dict[str, float]) -> dict:
    """Matrices needed to evaluate (or simulate) CLV for any discount / hazard scenario.

    Mirrors ``clv.expected_cashflows``: payment j (billing date ``tau_j``) is collected if the
    customer is still active in the middle of that cycle, ``tau_j + period/2``.
    """
    t0 = snap["tenure_days"].to_numpy(float)
    price = snap["plan_price"].to_numpy(float)
    if "has_billing_record" in snap:
        price = price * snap["has_billing_record"].to_numpy(float)
    period = snap["billing_period"].map(C.BILLING_PERIOD_DAYS).to_numpy(float)
    q = snap["billing_period"].map(success_rate).to_numpy(float)
    horizon_days = horizon_months * C.DAYS_PER_MONTH
    first_tau = (np.floor(t0 / period) + 1.0) * period
    n_slots = int(np.ceil(horizon_days / period.min())) + 1
    tau = first_tau[:, None] + np.arange(n_slots)[None, :] * period[:, None]      # n x J
    elapsed = tau - t0[:, None]                                                    # days from now
    s0 = np.maximum(survival_matrix(model, snap, t0[:, None])[:, 0], 1e-12)
    s1 = survival_matrix(model, snap, tau + period[:, None] / 2.0)
    p_pay = np.clip(s1 / s0[:, None], 0.0, 1.0)
    return {"price": price, "q": q, "elapsed": elapsed, "p_pay": p_pay,
            "t0": t0, "snap_index": snap.index.to_numpy()}


def clv_from_components(comp: dict, horizon_months: float, annual_discount: float,
                        hazard_multiplier: float = 1.0, success_multiplier: float = 1.0) -> np.ndarray:
    """Per-customer expected discounted revenue under a scenario.

    ``hazard_multiplier`` m scales the churn hazard: survival becomes S**m (m < 1 = less churn).
    ``success_multiplier`` scales the payment-success probability (capped at 1).
    """
    due = comp["elapsed"] <= horizon_months * C.DAYS_PER_MONTH
    p = comp["p_pay"] ** hazard_multiplier if hazard_multiplier != 1.0 else comp["p_pay"]
    disc = (1.0 + annual_discount) ** (-comp["elapsed"] / C.DAYS_PER_YEAR)
    q = np.clip(comp["q"] * success_multiplier, 0.0, 1.0)
    return comp["price"] * q * (p * disc * due).sum(axis=1)


def simulate_totals(comp: dict, horizon_months: float, annual_discount: float,
                    n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo draws of the portfolio's collected (discounted) revenue.

    One uniform U per customer fixes the churn time (alive at slot j iff U < p_j; p_j is
    decreasing in j so this is a valid single churn time), and each payment attempt succeeds
    independently with probability q.
    """
    due = comp["elapsed"] <= horizon_months * C.DAYS_PER_MONTH
    disc = (1.0 + annual_discount) ** (-comp["elapsed"] / C.DAYS_PER_YEAR)
    weight = comp["price"][:, None] * disc * due
    keep = weight.sum(axis=1) > 0
    weight, p, q = weight[keep], comp["p_pay"][keep], comp["q"][keep][:, None]
    out = np.empty(n_sims)
    for s in range(n_sims):
        alive = rng.random(len(p))[:, None] < p
        paid = alive & (rng.random(p.shape) < q)
        out[s] = (weight * paid).sum()
    return out


# --------------------------------------------------------------------------
# Bootstrap (parameter uncertainty) + predictive intervals
# --------------------------------------------------------------------------
def iter_bootstrap_models(frame: pd.DataFrame, B: int, seed: int = C.RANDOM_STATE,
                          model_cls=CoxModel):
    """Yield survival models refitted on B resamples of the spells (with replacement).

    A generator on purpose: every fitted Cox model keeps its training data, so holding
    hundreds of them in memory is wasteful. Callers score each replicate and drop it.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    for _ in range(B):
        boot = frame.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        try:
            yield model_cls().fit(boot)
        except Exception:                          # a degenerate resample: skip it
            continue


def fit_bootstrap_models(frame: pd.DataFrame, B: int, seed: int = C.RANDOM_STATE,
                         model_cls=CoxModel) -> list:
    """List version of :func:`iter_bootstrap_models` (use only for small B)."""
    return list(iter_bootstrap_models(frame, B, seed, model_cls))


def percentile_ci(x, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(np.asarray(x, float), [a, 1 - a])
    return float(lo), float(hi)


def bootstrap_portfolio(models, snap: pd.DataFrame, success_rate: dict[str, float],
                        horizon_months: float, annual_discount: float,
                        retention_months: int = 12, sims_per_fit: int = 10,
                        seed: int = C.RANDOM_STATE) -> dict:
    """Replicate portfolio CLV, segment CLV, segment retention and predictive draws."""
    rng = np.random.default_rng(seed + 1)
    seg = snap["plan_name"].astype(str) + "|" + snap["billing_period"].astype(str)
    seg_codes, seg_names = pd.factorize(seg)
    totals, predictive, seg_tot_rows, seg_ret_rows = [], [], [], []
    t0 = snap["tenure_days"].to_numpy(float)
    for model in models:
        comp = cashflow_components(model, snap, horizon_months, success_rate)
        clv_i = clv_from_components(comp, horizon_months, annual_discount)
        totals.append(clv_i.sum())
        seg_tot_rows.append(np.bincount(seg_codes, weights=clv_i, minlength=len(seg_names)))
        s0 = np.maximum(survival_matrix(model, snap, t0[:, None])[:, 0], 1e-12)
        s1 = survival_matrix(model, snap, (t0 + retention_months * C.DAYS_PER_MONTH)[:, None])[:, 0]
        ret = np.clip(s1 / s0, 0, 1)
        seg_ret_rows.append(np.bincount(seg_codes, weights=ret, minlength=len(seg_names)) / np.maximum(
            np.bincount(seg_codes, minlength=len(seg_names)), 1))
        predictive.append(simulate_totals(comp, horizon_months, annual_discount, sims_per_fit, rng))
    return {"totals": np.array(totals), "predictive": np.concatenate(predictive),
            "segments": list(seg_names), "segment_totals": np.array(seg_tot_rows),
            "segment_retention": np.array(seg_ret_rows)}


def summarise_bootstrap(boot: dict, point_total: float, point_segment_totals: pd.Series,
                        point_segment_retention: pd.Series, level: float = 0.95
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo, hi = percentile_ci(boot["totals"], level)
    plo, phi = percentile_ci(boot["predictive"], level)
    total = pd.DataFrame([
        {"quantity": "Portfolio 36-month CLV - confidence interval (estimation uncertainty)",
         "point_estimate": point_total, "ci_low": lo, "ci_high": hi,
         "relative_half_width_pct": 100 * (hi - lo) / 2 / point_total, "replicates": len(boot["totals"])},
        {"quantity": "Portfolio 36-month revenue - predictive interval (estimation + random churn/payments)",
         "point_estimate": point_total, "ci_low": plo, "ci_high": phi,
         "relative_half_width_pct": 100 * (phi - plo) / 2 / point_total, "replicates": len(boot["predictive"])},
    ])
    rows = []
    for k, name in enumerate(boot["segments"]):
        tlo, thi = percentile_ci(boot["segment_totals"][:, k], level)
        rlo, rhi = percentile_ci(boot["segment_retention"][:, k], level)
        rows.append({"segment": name,
                     "total_clv_36m": float(point_segment_totals.get(name, np.nan)),
                     "clv_ci_low": tlo, "clv_ci_high": thi,
                     "retention_12m_mean": float(point_segment_retention.get(name, np.nan)),
                     "retention_ci_low": rlo, "retention_ci_high": rhi})
    seg = pd.DataFrame(rows).sort_values("total_clv_36m", ascending=False)
    return total, seg


# --------------------------------------------------------------------------
# Scenario grid
# --------------------------------------------------------------------------
HAZARD_CHANGES_PCT = [-50, -25, -10, 0, 10, 25, 50]        # change in churn hazard
DISCOUNT_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]
HORIZONS_GRID = [12, 24, 36, 60]
SUCCESS_MULTIPLIERS = [0.90, 0.95, 1.00]


def scenario_grid(model, snap: pd.DataFrame, success_rate: dict[str, float]) -> pd.DataFrame:
    """Segment-level CLV for every combination of the scenario levers (point estimates)."""
    comp = cashflow_components(model, snap, max(HORIZONS_GRID), success_rate)
    seg = (snap["plan_name"].astype(str) + "|" + snap["billing_period"].astype(str))
    codes, names = pd.factorize(seg)
    rows = []
    for dh in HAZARD_CHANGES_PCT:
        m = 1.0 + dh / 100.0
        for r in DISCOUNT_RATES:
            for h in HORIZONS_GRID:
                for sm in SUCCESS_MULTIPLIERS:
                    v = clv_from_components(comp, h, r, m, sm)
                    tot = np.bincount(codes, weights=v, minlength=len(names))
                    for k, name in enumerate(names):
                        rows.append({"churn_hazard_change_pct": dh, "annual_discount_rate": r,
                                     "horizon_months": h, "payment_success_multiplier": sm,
                                     "segment": name, "total_clv": float(tot[k])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Interval backtest helper
# --------------------------------------------------------------------------
def revenue_intervals(models, populations: dict, success_rate: dict[str, float],
                      sims_per_fit: int = 20, level: float = 0.90,
                      seed: int = C.RANDOM_STATE) -> dict:
    """Predictive intervals for undiscounted revenue over several horizons in one pass.

    ``populations`` maps horizon_months -> the customers to forecast. Each bootstrap model is
    scored for every horizon and then discarded.
    """
    rng = np.random.default_rng(seed + 7)
    point = {h: [] for h in populations}
    draws = {h: [] for h in populations}
    for model in models:
        for h, pop in populations.items():
            comp = cashflow_components(model, pop, h, success_rate)
            point[h].append(clv_from_components(comp, h, 0.0).sum())
            draws[h].append(simulate_totals(comp, h, 0.0, sims_per_fit, rng))
    out = {}
    for h in populations:
        lo, hi = percentile_ci(np.concatenate(draws[h]), level)
        out[h] = {"expected": float(np.mean(point[h])), "lo": lo, "hi": hi}
    return out
