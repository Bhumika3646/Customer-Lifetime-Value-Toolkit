"""Rolling-origin backtests.

Every forecast origin ("cutoff") is treated as if it were today:
  * survival models are fitted on spells censored at the cutoff;
  * ML models are trained only on snapshots whose LABEL WINDOW has already ended
    by the cutoff (train_cutoff + horizon <= cutoff);
  * predictions are scored against what actually happened after the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import clv, config as C, features as F, ml_models, survival as sv, uncertainty as U
from .data import build_subscribers
from . import explainability

SURVIVAL_MODELS = {"Global KM": sv.GlobalKM, "Segment KM": sv.SegmentKM, "Cox PH": sv.CoxModel}


def _hdays(h_months: int) -> int:
    return int(round(h_months * C.DAYS_PER_MONTH))


class Backtester:
    def __init__(self, tables: dict, cutoffs=None, horizons=None, verbose=True,
                 first_test_cutoff=None):
        self.t = tables
        self.subs = build_subscribers(tables)
        self.cutoffs = [pd.Timestamp(c) for c in (cutoffs or C.BACKTEST_CUTOFFS)]
        first = pd.Timestamp(first_test_cutoff or C.FIRST_TEST_CUTOFF)
        self.test_cutoffs = [c for c in self.cutoffs if c >= first]
        self.horizons = horizons or C.HORIZONS_MONTHS
        self.verbose = verbose
        self._snaps, self._models, self._q = {}, {}, {}
        self.retention_predictions: list[pd.DataFrame] = []

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---- cached builders -------------------------------------------------
    def snapshot(self, cutoff, h_months) -> pd.DataFrame:
        key = (cutoff, h_months)
        if key not in self._snaps:
            self._snaps[key] = F.build_snapshot(self.t, cutoff, _hdays(h_months))
        return self._snaps[key]

    def models(self, cutoff) -> dict:
        if cutoff not in self._models:
            frame = sv.survival_frame(self.subs, self.t, cutoff)
            self._models[cutoff] = {n: cls().fit(frame) for n, cls in SURVIVAL_MODELS.items()}
        return self._models[cutoff]

    def success_rate(self, cutoff):
        if cutoff not in self._q:
            self._q[cutoff] = clv.payment_success_rate(self.t, cutoff)
        return self._q[cutoff]

    def _train_snapshots(self, cutoff, h_months, population) -> pd.DataFrame | None:
        limit = cutoff - pd.Timedelta(days=_hdays(h_months))
        parts = [population(self.snapshot(c, h_months)) for c in self.cutoffs if c <= limit]
        parts = [p for p in parts if len(p)]
        if not parts:
            return None
        train = pd.concat(parts, ignore_index=True)
        return train if len(train) >= 500 else None

    # ---- populations -----------------------------------------------------
    @staticmethod
    def paid_monthly(s: pd.DataFrame) -> pd.DataFrame:
        return s[(s["billing_period"] == "monthly") & (s["plan_price"] > 0)
                 & (s["has_billing_record"] == 1)].reset_index(drop=True)

    @staticmethod
    def churn_population(s: pd.DataFrame) -> pd.DataFrame:
        # Free customers, plus paid customers that have billing evidence.
        return s[(s["plan_price"] == 0) | (s["has_billing_record"] == 1)].reset_index(drop=True)

    # ---- revenue backtest ------------------------------------------------
    def run_revenue(self) -> pd.DataFrame:
        rows = []
        for h in self.horizons:
            for c in self.test_cutoffs:
                if c + pd.Timedelta(days=_hdays(h)) > C.PAYMENT_DATA_END:
                    continue
                test = self.paid_monthly(self.snapshot(c, h))
                if test.empty:
                    continue
                self._log(f"  revenue backtest  H={h:>2}m  cutoff={c.date()}  n={len(test)}")
                actual = test[F.TARGET_REVENUE].to_numpy()
                preds = {"No-churn benchmark": clv.naive_no_churn_revenue(test, h)}
                for name, model in self.models(c).items():
                    preds[name] = clv.expected_cashflows(model, test, h, self.success_rate(c)).expected_revenue.to_numpy()
                train = self._train_snapshots(c, h, self.paid_monthly)
                if train is not None:
                    preds["Gradient boosting"] = ml_models.fit_predict_revenue(train, test, F.TARGET_REVENUE)
                groups = (test["plan_name"] + "|" + test["acquisition_channel"]).to_numpy()
                for name, p in preds.items():
                    err = p - actual
                    gdf = pd.DataFrame({"g": groups, "p": p, "a": actual}).groupby("g").sum()
                    rows.append({
                        "horizon_months": h, "cutoff": c, "model": name, "n": len(test),
                        "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                        "mean_actual": actual.mean(), "mean_pred": p.mean(),
                        "portfolio_bias_pct": 100 * (p.sum() / actual.sum() - 1),
                        "segment_abs_error_pct": 100 * (gdf["p"] - gdf["a"]).abs().sum() / gdf["a"].sum(),
                    })
        return pd.DataFrame(rows)

    # ---- predictive-interval coverage -----------------------------------
    def run_interval_coverage(self, B: int = 25, sims_per_fit: int = 20, level: float = 0.90) -> pd.DataFrame:
        """Do 90% predictive intervals for portfolio revenue contain what actually happened?

        For every origin the Cox model is refitted on B bootstrap resamples (parameter
        uncertainty) and revenue is simulated from each fit (random churn and payments).
        The interval is then compared with realised revenue of paid monthly customers.
        """
        rows = []
        for c in self.test_cutoffs:
            pops = {}
            for h in self.horizons:
                if c + pd.Timedelta(days=_hdays(h)) > C.PAYMENT_DATA_END:
                    continue
                test = self.paid_monthly(self.snapshot(c, h))
                if not test.empty:
                    pops[h] = test
            if not pops:
                continue
            frame = sv.survival_frame(self.subs, self.t, c)
            seed = C.RANDOM_STATE + int(c.strftime("%Y%m%d")) % 1000
            ivs = U.revenue_intervals(U.iter_bootstrap_models(frame, B, seed), pops,
                                      self.success_rate(c), sims_per_fit, level)
            for h, test in pops.items():
                iv, actual = ivs[h], float(test[F.TARGET_REVENUE].sum())
                rows.append({"horizon_months": h, "cutoff": c, "customers": len(test),
                             "actual_revenue": actual, "expected_revenue": iv["expected"],
                             "interval_low": iv["lo"], "interval_high": iv["hi"],
                             "interval_level": level,
                             "actual_within_interval": bool(iv["lo"] <= actual <= iv["hi"]),
                             "actual_vs_expected_pct": 100 * (actual / iv["expected"] - 1),
                             "half_width_pct": 100 * (iv["hi"] - iv["lo"]) / 2 / iv["expected"]})
            self._log(f"  interval coverage cutoff={c.date()} horizons={list(pops)}")
        return pd.DataFrame(rows).sort_values(["horizon_months", "cutoff"]).reset_index(drop=True)

    # ---- retention / churn backtest -------------------------------------
    def run_retention(self) -> pd.DataFrame:
        rows = []
        for h in self.horizons:
            for c in self.test_cutoffs:
                if c + pd.Timedelta(days=_hdays(h)) > C.OBSERVATION_END:
                    continue
                test = self.churn_population(self.snapshot(c, h))
                y = test[F.TARGET_CHURN].to_numpy()
                if y.sum() < 20 or y.sum() == len(y):
                    continue
                self._log(f"  retention backtest H={h:>2}m  cutoff={c.date()}  n={len(test)}")
                preds = {}
                for name, model in self.models(c).items():
                    preds[name] = 1.0 - clv.retention_probability(model, test, h)
                train = self._train_snapshots(c, h, self.churn_population)
                if train is not None and train[F.TARGET_CHURN].nunique() == 2:
                    preds["Gradient boosting"] = ml_models.fit_predict_churn(train, test, F.TARGET_CHURN)
                for name, p in preds.items():
                    rows.append({
                        "horizon_months": h, "cutoff": c, "model": name, "n": len(test),
                        "actual_churn_rate": y.mean(), "mean_pred_churn": p.mean(),
                        "calibration_gap": p.mean() - y.mean(),
                        "ROC_AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
                    })
                    if name in ("Segment KM", "Cox PH"):
                        self.retention_predictions.append(pd.DataFrame({
                            "horizon_months": h, "cutoff": c, "model": name,
                            "plan": (test["plan_name"] + "|" + test["billing_period"]).values,
                            "pred_churn": p, "churned": y}))
        return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and std of each metric across forecast origins, per horizon and model."""
    # Compare models on the SAME forecast origins only (a model that could not be trained
    # at early origins would otherwise be averaged over different, easier/harder dates).
    n_here = results.groupby(["horizon_months", "cutoff"])["model"].transform("nunique")
    n_all = results.groupby("horizon_months")["model"].transform("nunique")
    results = results[n_here == n_all]
    g = results.groupby(["horizon_months", "model"])
    out = g[metrics].mean()
    out.columns = [f"{m}" for m in metrics]
    std = g[metrics].std().add_suffix("_std")
    out = out.join(std)
    out["origins"] = g.size()
    return out.reset_index()


def group_shuffle_holdout(
    frame: pd.DataFrame,
    group_col: str = "customer_id",
    test_size: float = C.CUSTOMER_GROUP_HOLDOUT_FRACTION,
    random_state: int = C.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split rows by group ID; never split a group across sets."""
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")
    if group_col not in frame.columns:
        raise ValueError(f"Required grouping column {group_col!r} is missing.")
    if frame[group_col].isna().any():
        raise ValueError(f"Grouping column {group_col!r} contains missing IDs.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, groups=frame[group_col]))
    train = frame.iloc[train_idx].copy().reset_index(drop=True)
    test = frame.iloc[test_idx].copy().reset_index(drop=True)
    overlap = set(train[group_col]) & set(test[group_col])
    if overlap:
        raise AssertionError(f"Group leakage: {len(overlap)} groups occur in both splits.")
    return train, test


def run_customer_grouped_holdout(
    tables: dict,
    cutoff=C.CUSTOMER_GROUP_HOLDOUT_CUTOFF,
    horizon_months: int = 3,
    test_size: float = C.CUSTOMER_GROUP_HOLDOUT_FRACTION,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate supervised predictors on customers excluded from all training snapshots.

    Training examples are historical snapshots whose full target horizon had
    ended by the fixed forecast cutoff. A deterministic grouped split chooses
    test customers from the current-cutoff cohort; those IDs are then removed
    from every historical snapshot before training. This complements, but does
    not replace, the rolling-origin temporal backtest.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_days = _hdays(horizon_months)
    subscribers = build_subscribers(tables)
    customer_map = subscribers.set_index("subscription_id")["customer_id"]

    def attach_customer(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["customer_id"] = frame["subscription_id"].map(customer_map)
        if frame["customer_id"].isna().any():
            raise ValueError("Could not map every subscription to a customer ID.")
        return frame

    current = attach_customer(F.build_snapshot(tables, cutoff, horizon_days))
    history_parts = []
    for train_cutoff in C.BACKTEST_CUTOFFS:
        origin = pd.Timestamp(train_cutoff)
        if origin >= cutoff or origin + pd.Timedelta(days=horizon_days) > cutoff:
            continue
        historical = attach_customer(F.build_snapshot(tables, origin, horizon_days))
        history_parts.append(historical)
    if not history_parts:
        raise ValueError("No historical snapshots have fully observed labels before holdout cutoff.")
    history = pd.concat(history_parts, ignore_index=True)

    rows = []
    common = {
        "forecast_cutoff": cutoff.date().isoformat(),
        "horizon_months": int(horizon_months),
        "test_size_requested": float(test_size),
        "historical_origins_used": int(len(history_parts)),
    }

    # 1) Revenue task: restrict the current cohort and every training origin to
    # paid monthly subscriptions with billing evidence, matching the main revenue
    # backtest population.
    current_rev = Backtester.paid_monthly(current)
    _, test_rev = group_shuffle_holdout(current_rev, "customer_id", test_size, C.RANDOM_STATE)
    held_rev_ids = set(test_rev["customer_id"].tolist())
    train_rev = Backtester.paid_monthly(history[~history["customer_id"].isin(held_rev_ids)].copy())
    overlap_rev = set(train_rev["customer_id"]) & held_rev_ids
    if overlap_rev:
        raise AssertionError("Revenue grouped holdout leaked test customers into historical training.")
    if len(train_rev) < 500 or test_rev.empty:
        raise ValueError("Insufficient rows for customer-grouped revenue evaluation.")

    actual_rev = test_rev[F.TARGET_REVENUE].to_numpy(float)
    revenue_models = ml_models.fit_comparison_models(train_rev, test_rev, F.TARGET_REVENUE, "regression")
    zero_rev = np.zeros(len(test_rev), dtype=float)
    train_groups_rev = int(train_rev["customer_id"].nunique())
    test_groups_rev = int(test_rev["customer_id"].nunique())
    revenue_explanation_model = revenue_models.get("XGBoost")

    def append_revenue(model_name, prediction):
        metrics = ml_models.score_regression_predictions(actual_rev, prediction)
        rows.append({
            **common, "task": "3-month revenue", "model": model_name,
            "train_customer_groups": train_groups_rev, "test_customer_groups": test_groups_rev,
            "train_rows": int(len(train_rev)), "test_rows": int(len(test_rev)),
            "customer_overlap": int(len(set(train_rev["customer_id"]) & set(test_rev["customer_id"]))),
            **metrics, "ROC_AUC": np.nan, "PR_AUC": np.nan, "Brier": np.nan, "calibration_gap": np.nan,
        })

    for model_name in ml_models.REGRESSION_MODEL_NAMES:
        append_revenue(model_name, revenue_models[model_name]["prediction"])
    append_revenue("Zero-revenue baseline", zero_rev)
    # Strong naive benchmark: every customer pays their plan price for each month of the horizon.
    price_baseline = test_rev["plan_price"].to_numpy(float) * float(horizon_months)
    append_revenue("Plan-price baseline (price x months)", price_baseline)

    # Per-plan accuracy: pooled R2 is inflated by between-plan differences in price.
    plan_rows = []
    all_preds = {n: revenue_models[n]["prediction"] for n in ml_models.REGRESSION_MODEL_NAMES}
    all_preds["Plan-price baseline (price x months)"] = price_baseline
    for plan, idx in test_rev.groupby("plan_name").indices.items():
        for model_name, pred in all_preds.items():
            m = ml_models.score_regression_predictions(actual_rev[idx], pred[idx])
            plan_rows.append({"plan": plan, "model": model_name, "test_rows": int(len(idx)),
                              "mean_actual": float(actual_rev[idx].mean()), **m})
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plan_rows).to_csv(C.RESULTS_DIR / "06_grouped_revenue_by_plan.csv", index=False)

    # 2) Churn task: Free customers plus paid customers with billing evidence.
    current_churn = Backtester.churn_population(current)
    _, test_churn = group_shuffle_holdout(current_churn, "customer_id", test_size, C.RANDOM_STATE + 1)
    held_churn_ids = set(test_churn["customer_id"].tolist())
    train_churn = Backtester.churn_population(history[~history["customer_id"].isin(held_churn_ids)].copy())
    overlap_churn = set(train_churn["customer_id"]) & held_churn_ids
    if overlap_churn:
        raise AssertionError("Churn grouped holdout leaked test customers into historical training.")
    y_train = train_churn[F.TARGET_CHURN].to_numpy(int)
    y_test = test_churn[F.TARGET_CHURN].to_numpy(int)
    if len(train_churn) < 500 or len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("Insufficient class variation for customer-grouped churn evaluation.")
    churn_models = ml_models.fit_comparison_models(train_churn, test_churn, F.TARGET_CHURN, "classification")
    prevalence = float(y_train.mean())
    prevalence_p = np.full(len(y_test), prevalence, dtype=float)
    train_groups_churn = int(train_churn["customer_id"].nunique())
    test_groups_churn = int(test_churn["customer_id"].nunique())
    churn_explanation_model = churn_models.get("XGBoost")

    def append_churn(model_name, probability, include_auc=True):
        metrics = ml_models.score_classification_predictions(y_test, probability)
        if not include_auc:
            metrics["ROC_AUC"] = np.nan
            metrics["PR_AUC"] = np.nan
        rows.append({
            **common, "task": "3-month churn", "model": model_name,
            "train_customer_groups": train_groups_churn, "test_customer_groups": test_groups_churn,
            "train_rows": int(len(train_churn)), "test_rows": int(len(test_churn)),
            "customer_overlap": int(len(set(train_churn["customer_id"]) & set(test_churn["customer_id"]))),
            "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "portfolio_bias_pct": np.nan,
            **metrics,
        })

    for model_name in ml_models.CLASSIFICATION_MODEL_NAMES:
        append_churn(model_name, churn_models[model_name]["prediction"])
    append_churn("Historical prevalence baseline", prevalence_p)

    # Explain XGBoost revenue-proxy and churn predictions on held-out customers.
    # Explainability failures are recorded to a status file and do not erase valid backtests.
    _, _, shap_status = explainability.write_shap_outputs(
        {
            "revenue": {"pipeline": revenue_explanation_model["pipeline"], "test_frame": test_rev}
                if revenue_explanation_model else None,
            "churn": {"pipeline": churn_explanation_model["pipeline"], "test_frame": test_churn}
                if churn_explanation_model else None,
        },
        C.RESULTS_DIR,
    )
    if verbose:
        print(f"  {shap_status}", flush=True)

    result = pd.DataFrame(rows)
    if result["customer_overlap"].fillna(0).astype(int).ne(0).any():
        raise AssertionError("Customer-grouped holdout contains train/test customer overlap.")
    if verbose:
        print(
            f"  customer-grouped holdout cutoff={cutoff.date()} H={horizon_months}m "
            f"revenue test n={len(test_rev):,}; churn test n={len(test_churn):,}; overlap=0",
            flush=True,
        )
    return result
