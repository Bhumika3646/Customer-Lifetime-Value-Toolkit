"""Survival models for subscription lifetimes.

Three interchangeable models share one interface:

    model.fit(frame)                      # frame has duration, event + covariates
    model.survival(frame, times)          # S(times[i] | covariates of row i)

* ``GlobalKM``   - one Kaplan-Meier curve for everybody (naive benchmark)
* ``SegmentKM``  - one curve per plan x billing period
* ``CoxModel``   - Cox PH on plan + customer attributes, stratified by billing period

Beyond the longest observed duration every curve is extended with a constant
(exponential) hazard so that multi-year CLV can be computed. This tail is an
ASSUMPTION and is flagged in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from . import config as C
from .data import plan_at


# --------------------------------------------------------------------------
# Kaplan-Meier with an exponential tail
# --------------------------------------------------------------------------
class KMCurve:
    """Kaplan-Meier estimator supporting delayed entry (left truncation).

    A subject is at risk at event time tau if entry < tau <= duration.
    """

    def __init__(self, duration, event, entry=None):
        d = np.asarray(duration, float)
        e = np.asarray(event, int)
        en = np.zeros_like(d) if entry is None else np.asarray(entry, float)
        self.n, self.n_events = len(d), int(e.sum())
        times = np.unique(d[e == 1])
        if len(times):
            deaths = np.bincount(np.searchsorted(times, d[e == 1]), minlength=len(times))
            d_sorted, en_sorted = np.sort(d), np.sort(en)
            at_risk = (len(d) - np.searchsorted(d_sorted, times, side="left")) \
                      - (len(en) - np.searchsorted(en_sorted, times, side="left"))
            surv = np.cumprod(1.0 - deaths / np.maximum(at_risk, 1))
        else:
            surv = np.array([])
        self.times, self.surv = times, surv
        self.t_max = float(d.max()) if self.n else 0.0
        self.tail_hazard = self._tail_hazard(d, e, en)

    @staticmethod
    def _hazard(d, e, en, lo, hi):
        exposure = np.clip(np.minimum(d, hi) - np.maximum(en, lo), 0, None).sum()
        events = int(((e == 1) & (d > lo) & (d <= hi)).sum())
        return events, exposure

    def _tail_hazard(self, d, e, en) -> float:
        lo = max(0.0, self.t_max - C.TAIL_WINDOW_DAYS)
        events, exposure = self._hazard(d, e, en, lo, self.t_max)
        if events < C.TAIL_MIN_EVENTS or exposure <= 0:
            events, exposure = self._hazard(d, e, en, 0.0, self.t_max)
        return events / exposure if exposure > 0 else 0.0

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        idx = np.searchsorted(self.times, t, side="right") - 1
        if len(self.surv):
            s = np.where(idx >= 0, self.surv[np.clip(idx, 0, None)], 1.0)
        else:
            s = np.ones_like(t)
        beyond = t > self.t_max
        if beyond.any():
            s_at_max = self(np.array([self.t_max]))[0]
            s = np.where(beyond, s_at_max * np.exp(-self.tail_hazard * (t - self.t_max)), s)
        return s


class _Base:
    name = "base"

    def survival(self, frame: pd.DataFrame, times) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GlobalKM(_Base):
    name = "Global KM"

    def fit(self, frame):
        self.curve = KMCurve(frame["duration"].values, frame["event"].values, frame["entry"].values)
        return self

    def survival(self, frame, times):
        return self.curve(times)


def segment_key(frame: pd.DataFrame) -> pd.Series:
    return frame["plan_name"].astype(str) + "|" + frame["billing_period"].astype(str)


class SegmentKM(_Base):
    name = "Segment KM (plan x billing)"

    def fit(self, frame):
        self.fallback = KMCurve(frame["duration"].values, frame["event"].values, frame["entry"].values)
        self.curves = {}
        keys = segment_key(frame)
        for key, g in frame.groupby(keys):
            if len(g) >= C.MIN_SEGMENT_SIZE and g["event"].sum() > 0:
                self.curves[key] = KMCurve(g["duration"].values, g["event"].values, g["entry"].values)
        return self

    def survival(self, frame, times):
        times = np.asarray(times, float)
        out = self.fallback(times)
        keys = segment_key(frame).values
        for key, curve in self.curves.items():
            m = keys == key
            if m.any():
                out[m] = curve(times[m])
        return out


# --------------------------------------------------------------------------
# Cox proportional hazards (stratified by billing period)
# --------------------------------------------------------------------------
# Billing period is a STRATUM, not a covariate. Annual subscribers can only churn
# at renewal dates (day ~365 / ~730) while monthly subscribers churn continuously,
# so their hazards are not proportional (a single hazard ratio of ~10 for
# monthly-vs-annual badly violates the PH assumption). Stratifying gives each
# billing period its own baseline hazard and keeps proportional hazards only for
# the plan and customer-attribute effects within a stratum.
COX_COVARIATES = ["plan_name", "company_size", "industry", "acquisition_channel", "country"]
COX_STRATA = "billing_period"


class CoxModel(_Base):
    name = "Cox PH (stratified by billing period)"

    def __init__(self, covariates=COX_COVARIATES, strata=COX_STRATA):
        self.covariates = list(covariates)
        self.strata = strata

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(frame[self.covariates].astype(str), drop_first=False, dtype=float)
        if hasattr(self, "columns"):
            X = X.reindex(columns=self.columns, fill_value=0.0)
        return X

    def fit(self, frame):
        X = self._design(frame)
        X = X.loc[:, X.std() > 0]                       # drop constant columns
        # drop one dummy per covariate (reference level) to avoid perfect collinearity
        first_per_cov = {}
        for c in X.columns:
            for v in self.covariates:
                if c.startswith(f"{v}_") and v not in first_per_cov:
                    first_per_cov[v] = c
        X = X.drop(columns=list(first_per_cov.values()))
        self.columns = list(X.columns)
        data = X.copy()
        data[self.strata] = frame[self.strata].astype(str).values
        data["duration"] = frame["duration"].values
        data["event"] = frame["event"].values
        data["entry"] = frame["entry"].values
        self.cph = CoxPHFitter(penalizer=C.COX_PENALIZER)
        self.cph.fit(data, duration_col="duration", event_col="event", entry_col="entry",
                     strata=[self.strata])
        self.h0 = {}
        base = self.cph.baseline_cumulative_hazard_
        for stratum in base.columns:
            sub = data[data[self.strata] == str(stratum)]
            t_max = float(sub["duration"].max())
            h = base[stratum]
            times, vals = h.index.values.astype(float), h.values.astype(float)
            lo = max(0.0, t_max - C.TAIL_WINDOW_DAYS)
            idx_hi = np.searchsorted(times, t_max, side="right") - 1
            idx_lo = np.searchsorted(times, lo, side="right") - 1
            h_hi = vals[idx_hi] if idx_hi >= 0 else 0.0
            h_lo = vals[idx_lo] if idx_lo >= 0 else 0.0
            slope = (h_hi - h_lo) / max(t_max - lo, 1.0)
            if slope <= 0:
                slope = h_hi / max(t_max, 1.0)
            self.h0[str(stratum)] = {"times": times, "vals": vals, "t_max": t_max, "slope": slope}
        return self

    def _h0(self, stratum: str, t: np.ndarray) -> np.ndarray:
        m = self.h0.get(stratum) or next(iter(self.h0.values()))
        idx = np.searchsorted(m["times"], t, side="right") - 1
        base = np.where(idx >= 0, m["vals"][np.clip(idx, 0, None)], 0.0)
        beyond = t > m["t_max"]
        if beyond.any():
            base = np.where(beyond, m["vals"][-1] + m["slope"] * (t - m["times"][-1]), base)
        return base

    def summary(self) -> pd.DataFrame:
        s = self.cph.summary[["coef", "exp(coef)", "se(coef)", "p"]].copy()
        s.columns = ["coef", "hazard_ratio", "se", "p_value"]
        return s.reset_index().rename(columns={"covariate": "term"})

    def proportional_hazards_check(self, frame: pd.DataFrame, p_threshold: float = 0.01) -> pd.DataFrame:
        """Schoenfeld-residual test of the PH assumption (rank-transformed time).

        lifelines cannot compute residuals with delayed entry, so the check refits the
        same stratified model without the entry column. In the default data every spell
        starts at tenure 0 (post-upgrade unbilled spells are dropped), so this is exact.
        """
        from lifelines.statistics import proportional_hazard_test
        X = self._design(frame)[self.columns]
        data = X.copy()
        data[self.strata] = frame[self.strata].astype(str).values
        data["duration"], data["event"] = frame["duration"].values, frame["event"].values
        cph = CoxPHFitter(penalizer=C.COX_PENALIZER).fit(
            data, duration_col="duration", event_col="event", strata=[self.strata])
        out = proportional_hazard_test(cph, data, time_transform="rank").summary.reset_index()
        out = out.rename(columns={out.columns[0]: "term", "p": "p_value"})
        out["violates_ph"] = out["p_value"] < p_threshold
        return out[["term", "test_statistic", "p_value", "violates_ph"]]

    def survival(self, frame, times):
        times = np.asarray(times, float)
        ph = self.cph.predict_partial_hazard(self._design(frame)).values.ravel()
        strata = frame[self.strata].astype(str).values
        h0 = np.zeros(len(frame))
        for stratum in self.h0:
            m = strata == stratum
            if m.any():
                h0[m] = self._h0(stratum, times[m])
        unknown = ~np.isin(strata, list(self.h0))
        if unknown.any():
            h0[unknown] = self._h0(next(iter(self.h0)), times[unknown])
        return np.exp(-h0 * ph)


# --------------------------------------------------------------------------
# Helpers shared by CLV and validation
# --------------------------------------------------------------------------
def survival_frame(subs: pd.DataFrame, tables: dict, cutoff,
                   include_unbilled_upgrade_spells: bool = False) -> pd.DataFrame:
    """Spell-level duration table using ONLY information available at ``cutoff``.

    * A churn after the cutoff is treated as censored at the cutoff.
    * Plan changes are handled as time-varying, with delayed entry: a customer who
      upgraded (Free -> Basic) contributes a censored spell to the old plan up to the
      upgrade, and a spell to the new plan that ENTERS the risk set at the upgrade
      tenure. Without this, upgraders would be counted as Basic from day 0, which
      overstates Basic survival (immortal-time bias).
    * Only plan changes dated <= cutoff are used.
    * DATA-QUALITY RULE: in this dataset every upgraded subscription has no payment or
      invoice rows and never churns, so its post-upgrade spell carries no billing
      information and would bias the paid-plan hazard downward. It is therefore dropped
      unless ``include_unbilled_upgrade_spells`` is True.
    """
    cutoff = pd.Timestamp(cutoff)
    plans = tables["plans"].set_index("plan_id")[["plan_name", "billing_period"]]
    f = subs[subs["start_date"] <= cutoff].copy()
    churned = f["churn_date"].notna() & (f["churn_date"] <= cutoff)
    stop = f["churn_date"].where(churned, cutoff)
    f["total_duration"] = (stop - f["start_date"]).dt.days.astype(float)
    f["event"] = churned.astype(int)

    ch = tables["changes"]
    ch = ch[ch["change_date"] <= cutoff].sort_values("change_date").groupby("subscription_id").first()
    f = f.merge(ch[["old_plan_id", "new_plan_id", "change_date"]], left_on="subscription_id",
                right_index=True, how="left")
    changed = f["change_date"].notna()
    f["change_tenure"] = np.where(changed, (f["change_date"] - f["start_date"]).dt.days, np.nan)

    keep = ["subscription_id", "company_size", "industry", "acquisition_channel", "country"]
    # unchanged subscriptions: one spell on the plan held at the cutoff
    u = f[~changed].copy()
    # IMPORTANT: plan_id_final may reveal an upgrade that happened after the
    # historical cutoff. Reconstruct the plan held on the cutoff for all rows
    # without an observed pre-cutoff change. This keeps the survival backtest
    # leakage-safe for future Free -> Basic changes.
    cutoff_plan = plan_at(subs, tables["changes"], cutoff)
    u["plan_id"] = u["subscription_id"].map(cutoff_plan)
    u["entry"], u["duration"] = 0.0, u["total_duration"]
    # changed subscriptions: spell 1 (old plan, censored at change) + spell 2 (new plan, delayed entry)
    c = f[changed]
    s1 = c.copy(); s1["plan_id"] = s1["old_plan_id"]
    s1["entry"], s1["duration"], s1["event"] = 0.0, s1["change_tenure"], 0
    s2 = c.copy(); s2["plan_id"] = s2["new_plan_id"]
    s2["entry"], s2["duration"] = s2["change_tenure"], s2["total_duration"]
    parts = [u, s1] + ([s2] if include_unbilled_upgrade_spells else [])
    spells = pd.concat(parts, ignore_index=True)
    spells = spells.join(plans, on="plan_id")
    spells = spells[spells["duration"] > spells["entry"]]
    return spells[keep + ["plan_name", "billing_period", "entry", "duration", "event"]].reset_index(drop=True)


def conditional_survival(model: _Base, frame: pd.DataFrame, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """P(T > t1 | T > t0), clipped to [0, 1]."""
    s0 = np.maximum(model.survival(frame, t0), 1e-12)
    s1 = model.survival(frame, t1)
    return np.clip(s1 / s0, 0.0, 1.0)


def attribute_information_test(frame: pd.DataFrame) -> dict:
    """Likelihood-ratio test: do customer attributes add information beyond the plan?

    Compares Cox(plan, stratified by billing period) with Cox(plan + company size, industry,
    acquisition channel, country). Unpenalised fits.
    """
    from scipy.stats import chi2

    def fit(covs):
        X = pd.get_dummies(frame[covs].astype(str), drop_first=True, dtype=float)
        X = X.loc[:, X.std() > 0]
        data = X.copy()
        data[COX_STRATA] = frame[COX_STRATA].astype(str).values
        for c in ("duration", "event", "entry"):
            data[c] = frame[c].values
        cph = CoxPHFitter(penalizer=0.0).fit(data, duration_col="duration", event_col="event",
                                             entry_col="entry", strata=[COX_STRATA])
        return cph.log_likelihood_, X.shape[1]

    ll_plan, k_plan = fit(["plan_name"])
    ll_full, k_full = fit(COX_COVARIATES)
    stat, df = 2 * (ll_full - ll_plan), k_full - k_plan
    return {"LR_statistic": float(stat), "df": int(df), "p_value": float(chi2.sf(stat, df))}
