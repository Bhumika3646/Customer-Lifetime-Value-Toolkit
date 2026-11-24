"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clvkit import clv, config as C, data, explainability, features as F, ml_models, recommendations, survival as sv, validation as V  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return data.load_tables()


# ---------------------------------------------------------------- survival
def test_km_matches_lifelines_with_delayed_entry():
    rng = np.random.default_rng(0)
    n = 3000
    entry = rng.uniform(0, 100, n) * (rng.random(n) < 0.4)
    dur = entry + rng.exponential(200, n) + 1
    event = (rng.random(n) < 0.7).astype(int)
    ours = sv.KMCurve(dur, event, entry)
    ref = KaplanMeierFitter().fit(dur, event, entry=entry)
    t = np.array([20, 100, 250, 400, 600.0])
    np.testing.assert_allclose(ours(t), ref.survival_function_at_times(t).values, atol=1e-9)


def test_tail_extrapolation_is_monotone_and_positive():
    rng = np.random.default_rng(1)
    dur = rng.exponential(300, 5000)
    ev = (dur < 600).astype(int)
    dur = np.minimum(dur, 600)
    curve = sv.KMCurve(dur, ev)
    s = curve(np.array([100, 600, 900, 2000, 5000.0]))
    assert np.all(np.diff(s) <= 1e-12) and s.min() > 0


def test_survival_frame_never_uses_post_cutoff_information(tables):
    cutoff = pd.Timestamp("2024-06-30")
    subs = data.build_subscribers(tables)
    f = sv.survival_frame(subs, tables, cutoff)
    start = subs.set_index("subscription_id")["start_date"]
    max_allowed = (cutoff - f["subscription_id"].map(start)).dt.days
    assert (f["duration"] <= max_allowed + 1e-9).all()
    assert f["entry"].ge(0).all() and (f["duration"] > f["entry"]).all()


# ---------------------------------------------------------------- leakage
def test_forbidden_columns_guard():
    with pytest.raises(ValueError):
        F.assert_no_leakage(["tenure_days", "status"])
    F.assert_no_leakage(F.FEATURE_COLUMNS)  # real feature list must pass


def test_survival_frame_does_not_use_plan_changes_after_cutoff(tables):
    """A future upgrade must not label a historical spell with the final plan."""
    cutoff = pd.Timestamp("2024-06-30")
    subs = data.build_subscribers(tables)
    changes = tables["changes"]
    future = set(changes.loc[changes["change_date"] > cutoff, "subscription_id"])
    past = set(changes.loc[changes["change_date"] <= cutoff, "subscription_id"])
    candidates = future - past
    active_ids = set(subs.loc[
        (subs["start_date"] <= cutoff) &
        (subs["end_date"].isna() | (subs["end_date"] > cutoff)),
        "subscription_id"
    ])
    candidates &= active_ids
    assert candidates, "Fixture should contain subscriptions upgraded after the cutoff"

    frame = sv.survival_frame(subs, tables, cutoff)
    observed = frame[frame["subscription_id"].isin(candidates)]
    expected_plan_id = data.plan_at(subs, changes, cutoff)
    plan_names = tables["plans"].set_index("plan_id")["plan_name"]
    expected = {sid: plan_names.loc[expected_plan_id.loc[sid]] for sid in candidates}
    actual = observed.drop_duplicates("subscription_id").set_index("subscription_id")["plan_name"]
    assert set(candidates).issubset(actual.index)
    for sid in candidates:
        assert actual.loc[sid] == expected[sid]


def _truncate_world(tables, cutoff):
    """Rebuild the tables as they would have looked on `cutoff` (no future rows at all)."""
    t = {k: v.copy() for k, v in tables.items()}
    ch = t["changes"]
    future = ch[ch["change_date"] > cutoff].set_index("subscription_id")["old_plan_id"]
    subs = t["subscriptions"]
    m = subs["subscription_id"].isin(future.index)
    subs.loc[m, "plan_id"] = subs.loc[m, "subscription_id"].map(future)
    gone = subs["end_date"] > cutoff
    subs.loc[gone, "end_date"] = pd.NaT
    subs.loc[gone, "status"] = "active"
    t["subscriptions"] = subs[subs["start_date"] <= cutoff]
    t["payments"] = t["payments"][t["payments"]["payment_date"] <= cutoff]
    t["invoices"] = t["invoices"][t["invoices"]["invoice_date"] <= cutoff]
    t["churn"] = t["churn"][t["churn"]["churn_date"] <= cutoff]
    t["changes"] = ch[ch["change_date"] <= cutoff]
    return t


@pytest.mark.parametrize("cutoff", ["2024-03-31", "2025-03-31"])
def test_features_identical_when_the_future_is_deleted(tables, cutoff):
    """Strongest leakage test: features must not change if all post-cutoff rows vanish."""
    cutoff = pd.Timestamp(cutoff)
    full = F.build_snapshot(tables, cutoff, 91).set_index("subscription_id")
    trunc = F.build_snapshot(_truncate_world(tables, cutoff), cutoff, 91).set_index("subscription_id")
    assert full.index.equals(trunc.index)
    pd.testing.assert_frame_equal(full[F.FEATURE_COLUMNS], trunc[F.FEATURE_COLUMNS], check_dtype=False)


def test_unbilled_upgrades_are_flagged(tables):
    snap = F.build_snapshot(tables, "2025-06-30", 91)
    upgraded = snap[snap["plan_changes_before_cutoff"] > 0]
    assert len(upgraded) > 0
    assert (upgraded["has_billing_record"] == 0).all()
    assert (upgraded[F.TARGET_REVENUE] == 0).all()


# ---------------------------------------------------------------- CLV maths
def _toy(price=10.0, t0=100.0, billing="monthly", has_billing=1):
    return pd.DataFrame({"tenure_days": [t0], "plan_price": [price], "billing_period": [billing],
                         "plan_name": ["pro"], "has_billing_record": [has_billing]})


def test_clv_matches_closed_form_under_constant_hazard():
    rng = np.random.default_rng(3)
    lam = 0.04 / C.DAYS_PER_MONTH                       # 4% monthly hazard
    dur = rng.exponential(1 / lam, 60000)
    ev = (dur <= 900).astype(int)
    frame = pd.DataFrame({"duration": np.minimum(dur, 900.0), "event": ev, "entry": 0.0,
                          "plan_name": "pro", "billing_period": "monthly"})
    model = sv.GlobalKM().fit(frame)
    snap = _toy(price=10.0, t0=100.0)
    got = clv.expected_cashflows(model, snap, 12, {"monthly": 1.0, "annual": 1.0}).expected_revenue.iloc[0]
    P = C.DAYS_PER_MONTH
    tau1 = (np.floor(100.0 / P) + 1) * P
    taus = tau1 + P * np.arange(12)
    taus = taus[taus <= 100.0 + 12 * P]
    expected = 10.0 * np.exp(-lam * (taus + P / 2 - 100.0)).sum()
    assert got == pytest.approx(expected, rel=0.02)


def test_clv_basic_properties():
    rng = np.random.default_rng(4)
    dur = rng.exponential(250, 20000)
    frame = pd.DataFrame({"duration": np.minimum(dur, 800.0), "event": (dur <= 800).astype(int),
                          "entry": 0.0, "plan_name": "pro", "billing_period": "monthly"})
    model = sv.GlobalKM().fit(frame)
    q = {"monthly": 0.93, "annual": 0.93}
    s = _toy()
    c12 = clv.expected_cashflows(model, s, 12, q, 0.10)["clv"].iloc[0]
    c36 = clv.expected_cashflows(model, s, 36, q, 0.10)["clv"].iloc[0]
    undiscounted = clv.expected_cashflows(model, s, 36, q, 0.0)["clv"].iloc[0]
    assert 0 < c12 < c36 < undiscounted                           # horizon up, discounting down
    assert clv.expected_cashflows(model, _toy(price=0.0), 36, q)["clv"].iloc[0] == 0
    assert clv.expected_cashflows(model, _toy(has_billing=0), 36, q)["clv"].iloc[0] == 0
    assert clv.expected_cashflows(model, s, 12, {"monthly": 0.5, "annual": 0.5})["clv"].iloc[0] == \
        pytest.approx(clv.expected_cashflows(model, s, 12, {"monthly": 1.0, "annual": 1.0})["clv"].iloc[0] * 0.5)


def test_retention_probability_is_a_probability_and_decreases(tables):
    subs = data.build_subscribers(tables)
    frame = sv.survival_frame(subs, tables, "2025-06-30")
    model = sv.SegmentKM().fit(frame)
    snap = F.build_snapshot(tables, "2025-06-30", 91).head(500)
    p3, p12 = (clv.retention_probability(model, snap, m) for m in (3, 12))
    assert ((0 <= p12) & (p12 <= p3) & (p3 <= 1)).all()


def test_unbilled_billing_sensitivity_is_explicit_and_monotone():
    rng = np.random.default_rng(19)
    durations = rng.exponential(250, 30000)
    frame = pd.DataFrame({
        "duration": np.minimum(durations, 800.0),
        "event": (durations <= 800).astype(int),
        "entry": 0.0,
        "plan_name": "basic",
        "billing_period": "monthly",
    })
    model = sv.GlobalKM().fit(frame)
    snap = _toy(price=20.0, t0=100.0, billing="monthly", has_billing=0)
    table = clv.unbilled_billing_sensitivity(
        model, snap, {"monthly": 0.9, "annual": 0.9}, base_clv_total=100.0,
        horizon_months=12, annual_discount=0.1,
    )
    assert len(table) == 3
    assert table["flagged_paid_plan_subscriptions"].eq(1).all()
    assert table.loc[0, "estimated_clv_for_flagged_cohort"] == 0
    assert table.loc[1, "estimated_clv_for_flagged_cohort"] > 0
    assert table.loc[2, "estimated_clv_for_flagged_cohort"] >= table.loc[1, "estimated_clv_for_flagged_cohort"]
    assert table["portfolio_total_clv_scenario"].is_monotonic_increasing


def test_group_shuffle_holdout_never_splits_customer_groups():
    frame = pd.DataFrame({
        "customer_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        "value": range(10),
    })
    train, test = V.group_shuffle_holdout(frame, test_size=0.4, random_state=42)
    assert set(train["customer_id"]).isdisjoint(set(test["customer_id"]))
    assert len(test["customer_id"].unique()) == 2
    assert len(train) + len(test) == len(frame)


# ---------------------------------------------------------------- temporal behavior features

def test_temporal_delta_features_are_present_and_consistent(tables):
    snap = F.build_snapshot(tables, "2025-06-30", 91)
    required = {
        "revenue_prev30d", "revenue_days61_90", "revenue_delta_30d_vs_prev30d",
        "revenue_trend_ratio_30d_vs_prev30d", "payment_count_delta_30d_vs_prev30d",
        "failed_count_delta_30d_vs_prev30d",
    }
    assert required.issubset(snap.columns)
    np.testing.assert_allclose(
        snap["revenue_delta_30d_vs_prev30d"].to_numpy(),
        (snap["revenue_30d"] - snap["revenue_prev30d"]).to_numpy(),
    )
    expected_ratio = ((snap["revenue_30d"] - snap["revenue_prev30d"])
                      / (snap["revenue_30d"] + snap["revenue_prev30d"] + 1.0))
    np.testing.assert_allclose(snap["revenue_trend_ratio_30d_vs_prev30d"], expected_ratio)
    assert snap["revenue_trend_ratio_30d_vs_prev30d"].between(-1, 1).all()
    assert (snap[["revenue_prev30d", "revenue_days61_90"]].to_numpy() >= 0).all()


def _tiny_model_frame(n=48):
    rng = np.random.default_rng(121)
    frame = pd.DataFrame(index=range(n))
    for c in F.CATEGORICAL:
        frame[c] = rng.choice(["basic", "pro", "monthly", "annual"], size=n)
    for c in F.NUMERIC:
        frame[c] = rng.uniform(0, 50, size=n)
    frame["days_since_last_success"] = rng.choice([np.nan, 1.0, 15.0, 50.0], size=n)
    frame[F.TARGET_REVENUE] = np.maximum(0, frame["revenue_30d"] * 0.5 + rng.normal(0, 3, n))
    frame[F.TARGET_CHURN] = np.array([0, 1] * (n // 2), dtype=int)
    return frame


def test_four_regression_models_fit_and_return_predictions():
    frame = _tiny_model_frame(48)
    train, test = frame.iloc[:36].copy(), frame.iloc[36:].copy()
    fitted = ml_models.fit_comparison_models(train, test, F.TARGET_REVENUE, "regression")
    assert set(fitted) == {"Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"}
    for item in fitted.values():
        assert len(item["prediction"]) == len(test)
        assert np.isfinite(item["prediction"]).all()
        assert (item["prediction"] >= 0).all()


def test_four_churn_classifiers_fit_and_return_probabilities():
    frame = _tiny_model_frame(60)
    train, test = frame.iloc[:48].copy(), frame.iloc[48:].copy()
    fitted = ml_models.fit_comparison_models(train, test, F.TARGET_CHURN, "classification")
    assert set(fitted) == {"Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"}
    for item in fitted.values():
        p = item["prediction"]
        assert len(p) == len(test)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()


def test_value_risk_recommendations_keep_unbilled_as_data_review():
    df = pd.DataFrame({
        "subscription_id": [f"s{i}" for i in range(8)],
        "plan_name": ["pro"] * 8,
        "billing_period": ["monthly"] * 8,
        "plan_price": [30.0] * 8,
        "has_billing_record": [1] * 7 + [0],
        "clv_36m": [1200, 1000, 900, 700, 400, 200, 100, 0],
        "retention_prob_12m": [0.8, 0.1, 0.2, 0.7, 0.05, 0.75, 0.6, 0.2],
        "data_quality_flag": [""] * 7 + ["paid_plan_without_billing_records"],
    })
    customer, summary, priority = recommendations.build_value_risk_segments(df)
    assert "Billing data review" in set(customer["value_risk_segment"])
    assert customer.loc[7, "value_risk_segment"] == "Billing data review"
    assert customer.loc[7, "retention_recommendation"].startswith("Verify billing history")
    assert summary["customers"].sum() == len(df)
    assert set(priority["subscription_id"]).issubset(set(df["subscription_id"]))
    assert customer["estimated_12m_churn_risk"].between(0, 1).all()


def test_shap_feature_mapping_restores_original_feature_names():
    assert explainability._original_feature("cat__plan_name_premium") == "plan_name"
    assert explainability._original_feature("num__revenue_30d") == "revenue_30d"
    assert explainability._original_feature("num__missingindicator_days_since_last_success") == "days_since_last_success"


# ---------------------------------------------------------------- original-objective alignment

def test_temporal_delta_features_are_present_and_consistent(tables):
    snap = F.build_snapshot(tables, "2025-06-30", 91)
    required = {
        "revenue_prev30d", "revenue_days61_90", "revenue_delta_30d_vs_prev30d",
        "revenue_trend_ratio_30d_vs_prev30d", "payment_count_delta_30d_vs_prev30d",
        "failed_count_delta_30d_vs_prev30d",
    }
    assert required.issubset(snap.columns)
    np.testing.assert_allclose(
        snap["revenue_delta_30d_vs_prev30d"].to_numpy(),
        (snap["revenue_30d"] - snap["revenue_prev30d"]).to_numpy(),
    )
    expected_ratio = ((snap["revenue_30d"] - snap["revenue_prev30d"])
                      / (snap["revenue_30d"] + snap["revenue_prev30d"] + 1.0))
    np.testing.assert_allclose(snap["revenue_trend_ratio_30d_vs_prev30d"], expected_ratio)
    assert snap["revenue_trend_ratio_30d_vs_prev30d"].between(-1, 1).all()
    assert (snap[["revenue_prev30d", "revenue_days61_90"]].to_numpy() >= 0).all()


def _tiny_model_frame(n=48):
    rng = np.random.default_rng(121)
    frame = pd.DataFrame(index=range(n))
    for c in F.CATEGORICAL:
        frame[c] = rng.choice(["basic", "pro", "monthly", "annual"], size=n)
    for c in F.NUMERIC:
        frame[c] = rng.uniform(0, 50, size=n)
    frame["days_since_last_success"] = rng.choice([np.nan, 1.0, 15.0, 50.0], size=n)
    frame[F.TARGET_REVENUE] = np.maximum(0, frame["revenue_30d"] * 0.5 + rng.normal(0, 3, n))
    frame[F.TARGET_CHURN] = np.array([0, 1] * (n // 2), dtype=int)
    return frame


def test_four_regression_models_fit_and_return_predictions():
    frame = _tiny_model_frame(48)
    train, test = frame.iloc[:36].copy(), frame.iloc[36:].copy()
    fitted = ml_models.fit_comparison_models(train, test, F.TARGET_REVENUE, "regression")
    assert set(fitted) == {"Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"}
    for item in fitted.values():
        assert len(item["prediction"]) == len(test)
        assert np.isfinite(item["prediction"]).all()
        assert (item["prediction"] >= 0).all()


def test_four_churn_classifiers_fit_and_return_probabilities():
    frame = _tiny_model_frame(60)
    train, test = frame.iloc[:48].copy(), frame.iloc[48:].copy()
    fitted = ml_models.fit_comparison_models(train, test, F.TARGET_CHURN, "classification")
    assert set(fitted) == {"Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"}
    for item in fitted.values():
        p = item["prediction"]
        assert len(p) == len(test)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()


def test_value_risk_recommendations_keep_unbilled_as_data_review():
    df = pd.DataFrame({
        "subscription_id": [f"s{i}" for i in range(8)],
        "plan_name": ["pro"] * 8,
        "billing_period": ["monthly"] * 8,
        "plan_price": [30.0] * 8,
        "has_billing_record": [1] * 7 + [0],
        "clv_36m": [1200, 1000, 900, 700, 400, 200, 100, 0],
        "retention_prob_12m": [0.8, 0.1, 0.2, 0.7, 0.05, 0.75, 0.6, 0.2],
        "data_quality_flag": [""] * 7 + ["paid_plan_without_billing_records"],
    })
    customer, summary, priority = recommendations.build_value_risk_segments(df)
    assert "Billing data review" in set(customer["value_risk_segment"])
    assert customer.loc[7, "value_risk_segment"] == "Billing data review"
    assert customer.loc[7, "retention_recommendation"].startswith("Verify billing history")
    assert summary["customers"].sum() == len(df)
    assert set(priority["subscription_id"]).issubset(set(df["subscription_id"]))
    assert customer["estimated_12m_churn_risk"].between(0, 1).all()


def test_shap_feature_mapping_restores_original_feature_names():
    assert explainability._original_feature("cat__plan_name_premium") == "plan_name"
    assert explainability._original_feature("num__revenue_30d") == "revenue_30d"
    assert explainability._original_feature("num__missingindicator_days_since_last_success") == "days_since_last_success"


# ---------------------------------------------------------------- final-review fixes
def test_cox_is_stratified_by_billing_period_and_annual_churn_is_stepwise():
    """Annual subscribers churn only at renewal; a stratified Cox must not give them early churn."""
    rng = np.random.default_rng(5)
    n = 4000
    monthly = pd.DataFrame({"billing_period": "monthly", "plan_name": rng.choice(["basic", "pro"], n),
                            "duration": np.minimum(rng.exponential(300, n) + 1, 700.0)})
    monthly["event"] = (monthly["duration"] < 700).astype(int)
    annual = pd.DataFrame({"billing_period": "annual", "plan_name": rng.choice(["basic", "pro"], n),
                           "duration": np.where(rng.random(n) < 0.08, 365.0, 700.0)})
    annual["event"] = (annual["duration"] < 700).astype(int)
    frame = pd.concat([monthly, annual], ignore_index=True)
    for c in ("company_size", "industry", "acquisition_channel", "country"):
        frame[c] = rng.choice(["a", "b"], len(frame))
    frame["entry"] = 0.0
    model = sv.CoxModel().fit(frame)
    assert set(model.h0) == {"annual", "monthly"}
    assert not any(c.startswith("billing_period") for c in model.columns)
    probe = frame.iloc[[0, n]].copy()           # one monthly, one annual row
    s_100 = model.survival(probe, np.array([100.0, 100.0]))
    assert s_100[1] > 0.999 and s_100[0] < s_100[1]  # annual has no churn before renewal
    grid = np.array([50.0, 200.0, 400.0, 900.0, 2000.0])
    surv = np.array([model.survival(probe.iloc[[0]], np.array([t]))[0] for t in grid])
    assert np.all(np.diff(surv) <= 1e-12)


def test_value_risk_segments_use_paying_customers_and_free_is_separate():
    n = 40
    df = pd.DataFrame({
        "subscription_id": [f"s{i}" for i in range(n)],
        "plan_name": ["pro"] * 30 + ["free"] * 10,
        "billing_period": ["monthly"] * n,
        "plan_price": [49.0] * 30 + [0.0] * 10,
        "has_billing_record": [1] * n,
        "clv_36m": list(np.linspace(100, 2000, 30)) + [0.0] * 10,
        "retention_prob_12m": list(np.tile([0.9, 0.7, 0.4], 10)) + [0.3] * 10,
        "data_quality_flag": [""] * n,
    })
    customer, summary, _ = recommendations.build_value_risk_segments(df)
    free_segments = set(customer.loc[customer["plan_name"] == "free", "value_risk_segment"])
    assert free_segments == {"Free plan (conversion candidates)"}
    paying_segments = set(customer.loc[customer["plan_name"] == "pro", "value_risk_segment"])
    assert "High value / high risk" in paying_segments   # not empty any more
    assert summary["customers"].sum() == n


# ---------------------------------------------------------------- uncertainty / scenarios
def _toy_survival_setup(seed=11, n=1500):
    from clvkit import uncertainty  # noqa: F401  (import check)
    rng = np.random.default_rng(seed)
    bp = rng.choice(["monthly", "annual"], n, p=[.7, .3])
    frame = pd.DataFrame({
        "billing_period": bp,
        "plan_name": rng.choice(["basic", "pro"], n),
        "duration": np.where(bp == "monthly", np.minimum(rng.exponential(350, n) + 1, 700.0),
                             np.where(rng.random(n) < .1, 365.0, 700.0)),
    })
    frame["event"] = (frame["duration"] < 700).astype(int)
    for c in ("company_size", "industry", "acquisition_channel", "country"):
        frame[c] = rng.choice(["a", "b"], n)
    frame["entry"] = 0.0
    snap = frame.iloc[:300].copy()
    snap["tenure_days"] = rng.uniform(20, 500, len(snap))
    snap["plan_price"] = np.where(snap["plan_name"] == "pro", 50.0, 20.0)
    snap["has_billing_record"] = 1
    return frame, snap, {"monthly": 0.93, "annual": 0.92}


def test_cashflow_components_match_clv_module_exactly():
    from clvkit import uncertainty as U
    frame, snap, q = _toy_survival_setup()
    for model in (sv.CoxModel().fit(frame), sv.SegmentKM().fit(frame)):
        comp = U.cashflow_components(model, snap, 36, q)
        for h, d in ((12, 0.10), (36, 0.0), (36, 0.20)):
            ref = clv.expected_cashflows(model, snap, h, q, d)["clv"].to_numpy()
            assert np.allclose(U.clv_from_components(comp, h, d), ref, atol=1e-8)


def test_monte_carlo_mean_matches_analytic_clv():
    from clvkit import uncertainty as U
    frame, snap, q = _toy_survival_setup()
    comp = U.cashflow_components(sv.CoxModel().fit(frame), snap, 24, q)
    analytic = U.clv_from_components(comp, 24, 0.10).sum()
    sims = U.simulate_totals(comp, 24, 0.10, 600, np.random.default_rng(3))
    assert abs(sims.mean() / analytic - 1) < 0.02
    assert sims.std() > 0                       # there is genuine randomness


def test_scenario_levers_move_clv_in_the_right_direction():
    from clvkit import uncertainty as U
    frame, snap, q = _toy_survival_setup()
    comp = U.cashflow_components(sv.CoxModel().fit(frame), snap, 36, q)
    base = U.clv_from_components(comp, 36, 0.10).sum()
    assert np.isclose(U.clv_from_components(comp, 36, 0.10, 1.0, 1.0).sum(), base)
    assert U.clv_from_components(comp, 36, 0.10, hazard_multiplier=0.5).sum() > base   # less churn
    assert U.clv_from_components(comp, 36, 0.10, hazard_multiplier=1.5).sum() < base   # more churn
    assert U.clv_from_components(comp, 36, 0.20).sum() < base                          # higher discount
    assert U.clv_from_components(comp, 12, 0.10).sum() < base                          # shorter horizon
    assert np.isclose(U.clv_from_components(comp, 36, 0.10, success_multiplier=0.9).sum() / base, 0.9)


def test_bootstrap_returns_ordered_intervals():
    from clvkit import uncertainty as U
    frame, snap, q = _toy_survival_setup()
    boot = U.bootstrap_portfolio(U.iter_bootstrap_models(frame, 6), snap, q, 36, 0.10, sims_per_fit=5)
    assert len(boot["totals"]) == 6 and len(boot["predictive"]) == 30
    lo, hi = U.percentile_ci(boot["totals"], 0.95)
    assert lo <= boot["totals"].mean() <= hi and hi > lo
    assert boot["segment_totals"].shape == (6, len(boot["segments"]))
    assert np.all((boot["segment_retention"] >= 0) & (boot["segment_retention"] <= 1))


def test_scenario_grid_baseline_equals_point_estimate():
    from clvkit import uncertainty as U
    frame, snap, q = _toy_survival_setup()
    model = sv.CoxModel().fit(frame)
    grid = U.scenario_grid(model, snap, q)
    base = grid[(grid.churn_hazard_change_pct == 0) & np.isclose(grid.annual_discount_rate, .10)
                & (grid.horizon_months == 36) & np.isclose(grid.payment_success_multiplier, 1.0)]
    ref = clv.expected_cashflows(model, snap, 36, q, 0.10)["clv"].sum()
    assert np.isclose(base["total_clv"].sum(), ref)
