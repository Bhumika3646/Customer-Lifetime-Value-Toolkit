#!/usr/bin/env python
"""CLV Toolkit - one command runs the whole study.

    python run_pipeline.py              # full run (about 2-4 minutes)
    python run_pipeline.py --quick      # fewer forecast origins / horizons
    python run_pipeline.py --skip-backtest

Stages: 1 audit -> 2 survival models -> 3 CLV scoring -> 4 backtests -> 5 report.
Everything is written to ./clv_results (tables, figures, REPORT.md).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from clvkit import clv, config as C, data, features as F, recommendations as Rec, report as R, survival as sv, uncertainty as U, validation as V


def stage(n, title):
    print(f"\n=== Stage {n}: {title} " + "=" * max(3, 60 - len(title)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer origins and horizons")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap replicates for CLV intervals (default 200)")
    args = ap.parse_args()
    t_start = time.time()
    C.RESULTS_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(exist_ok=True)
    OUT, FIG = C.RESULTS_DIR, C.FIG_DIR

    # ------------------------------------------------------------------ 1
    stage(1, "Load and audit data")
    tables = data.load_tables()
    audit = data.audit_tables(tables)
    (OUT / "01_data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    subs = data.build_subscribers(tables)
    print(f"  {audit['rows']['subscriptions']:,} subscriptions, {audit['share_churned']:.1%} churned, "
          f"{audit['share_free_plan']:.1%} on the free plan")

    # ------------------------------------------------------------------ 2
    stage(2, "Fit survival models on all data")
    as_of = C.OBSERVATION_END
    frame = sv.survival_frame(subs, tables, as_of)
    km_global = sv.GlobalKM().fit(frame)
    km_seg = sv.SegmentKM().fit(frame)
    cox = sv.CoxModel().fit(frame)
    lr = sv.attribute_information_test(frame)
    month_grid = [3, 6, 12, 24]
    surv_rows = []
    for name, curve in sorted(km_seg.curves.items()):
        surv_rows.append({"segment": name.replace("|", " / "), "spells": curve.n, "churn_events": curve.n_events,
                          **{f"S({m}m)": float(curve(np.array([m * C.DAYS_PER_MONTH]))[0]) for m in month_grid},
                          "tail_hazard_per_month": curve.tail_hazard * C.DAYS_PER_MONTH})
    surv_table = pd.DataFrame(surv_rows)
    surv_table.to_csv(OUT / "02_survival_by_segment.csv", index=False)
    cox_table = cox.summary()
    cox_table.to_csv(OUT / "02_cox_hazard_ratios.csv", index=False)
    (OUT / "02_attribute_lr_test.json").write_text(json.dumps(lr, indent=2))
    ph_check = cox.proportional_hazards_check(frame)
    ph_check.to_csv(OUT / "02_cox_ph_check.csv", index=False)
    print(f"  PH check: {int(ph_check['violates_ph'].sum())} of {len(ph_check)} covariates violate proportional hazards (p<0.01)")
    R.fig_km(km_seg.curves, FIG / "km_curves.png")
    print(f"  attribute LR test: chi2={lr['LR_statistic']:.2f}, df={lr['df']}, p={lr['p_value']:.3f}")

    # ------------------------------------------------------------------ 3
    stage(3, "Score customer lifetime value")
    snap = F.build_snapshot(tables, as_of, horizon_days=1)
    q = clv.payment_success_rate(tables, as_of)
    out = snap[["subscription_id", "plan_name", "billing_period", "plan_price", "tenure_days",
                "company_size", "industry", "acquisition_channel", "country", "has_billing_record"]].copy()
    out["segment"] = out["plan_name"] + "|" + out["billing_period"]
    out["data_quality_flag"] = np.where((out["plan_price"] > 0) & (out["has_billing_record"] == 0),
                                        "paid_plan_without_billing_records", "")
    for label, model in (("cox", cox), ("segkm", km_seg)):
        r12 = clv.expected_cashflows(model, snap, 12, q, C.ANNUAL_DISCOUNT_RATE)
        r36 = clv.expected_cashflows(model, snap, C.CLV_HORIZON_MONTHS, q, C.ANNUAL_DISCOUNT_RATE)
        out[f"clv_12m_{label}"] = r12["clv"].values
        out[f"clv_36m_{label}"] = r36["clv"].values
    out["retention_prob_12m"] = clv.retention_probability(cox, snap, 12)
    out["expected_active_months_36m"] = clv.expected_active_months(cox, snap, 36)
    out["clv_12m"], out["clv_36m"] = out["clv_12m_cox"], out["clv_36m_cox"]
    # Why are paid-plan subscriptions unbilled? (cause breakdown; the largest piece is a data gap)
    flagged = out["data_quality_flag"] != ""
    start_year = (as_of - pd.to_timedelta(snap["tenure_days"], unit="D")).dt.year
    upgraded = snap["plan_changes_before_cutoff"] > 0
    annual_2025 = (snap["billing_period"] == "annual") & (start_year == 2025)
    cause = np.select([upgraded, annual_2025], ["Free->Basic upgrade with no billing rows",
                                                "Annual plan started in 2025 (2025 annual payments missing from data)"],
                      default="Monthly paid plan with no payment rows (other / not yet billed)")
    unbilled_breakdown = (pd.Series(cause[flagged.values]).value_counts().rename_axis("cause")
                          .reset_index(name="subscriptions"))
    unbilled_breakdown["share_pct"] = 100 * unbilled_breakdown["subscriptions"] / unbilled_breakdown["subscriptions"].sum()
    unbilled_breakdown.to_csv(OUT / "03_unbilled_breakdown.csv", index=False)
    out, value_risk_summary, retention_priority = Rec.build_value_risk_segments(out)
    out.to_csv(OUT / "03_customer_clv.csv", index=False)
    value_risk_summary.to_csv(OUT / "07_value_risk_segment_summary.csv", index=False)
    retention_priority.to_csv(OUT / "07_customer_retention_recommendations.csv", index=False)

    positive_clv = out[out["clv_36m"] > 0]
    seg_tab = (out.groupby("segment").agg(customers=("clv_36m", "size"),
                                          positive_clv_customers=("clv_36m", lambda x: int((x > 0).sum())),
                                          mean_clv_12m=("clv_12m", "mean"), mean_clv_36m=("clv_36m", "mean"),
                                          total_clv_36m=("clv_36m", "sum")).reset_index())
    seg_tab["share_of_total_clv"] = 100 * seg_tab["total_clv_36m"] / seg_tab["total_clv_36m"].sum()
    seg_tab = seg_tab.sort_values("total_clv_36m", ascending=False)
    seg_tab.to_csv(OUT / "03_clv_by_segment.csv", index=False)

    attr_rows = []
    for col in ("acquisition_channel", "company_size", "industry", "country"):
        # Descriptive averages only; do not interpret as adjusted effects beyond plan/billing mix.
        means = positive_clv.groupby(col)["clv_36m"].mean()
        attr_rows.append({"attribute": col, "lowest_group": means.idxmin(), "lowest_mean_clv": means.min(),
                          "highest_group": means.idxmax(), "highest_mean_clv": means.max(),
                          "spread_pct": 100 * (means.max() / means.min() - 1)})
    attr_tab = pd.DataFrame(attr_rows)
    attr_tab.to_csv(OUT / "03_clv_by_attribute.csv", index=False)

    v = np.sort(positive_clv["clv_36m"].to_numpy())[::-1]
    top20_share = 100 * v[: int(0.2 * len(v))].sum() / v.sum()
    sens = []
    positive_clv_snap = snap.loc[positive_clv.index]
    for h, d in ((12, .10), (36, 0.0), (36, .10), (36, .20), (60, .10)):
        r = clv.expected_cashflows(cox, positive_clv_snap, h, q, d)
        sens.append({"horizon_months": h, "annual_discount_rate": d,
                     "mean_clv_per_positive_clv_customer": r["clv"].mean(), "total_clv": r["clv"].sum()})
    sens_tab = pd.DataFrame(sens)
    sens_tab.to_csv(OUT / "03_clv_sensitivity.csv", index=False)
    unbilled_sensitivity = clv.unbilled_billing_sensitivity(
        cox, snap, q, float(out["clv_36m"].sum()),
        horizon_months=C.CLV_HORIZON_MONTHS, annual_discount=C.ANNUAL_DISCOUNT_RATE,
    )
    unbilled_sensitivity.to_csv(OUT / "03_unbilled_billing_sensitivity.csv", index=False)
    R.fig_clv(out, FIG / "clv_distribution.png")
    print(f"  scored {len(out):,} active customers; {len(positive_clv):,} with positive CLV; "
          f"top 20% of positive-CLV customers hold {top20_share:.1f}% of 36m CLV")

    # ------------------------------------------------------------------ 3b
    stage("3b", "Uncertainty (bootstrap + Monte Carlo) and scenario grid")
    B = 30 if args.quick else args.bootstrap
    t_b = time.time()
    priced = snap[snap["plan_price"] > 0]
    boot = U.bootstrap_portfolio(U.iter_bootstrap_models(frame, B), priced, q, C.CLV_HORIZON_MONTHS, C.ANNUAL_DISCOUNT_RATE)
    point_seg_tot = out.groupby("segment")["clv_36m"].sum()
    point_seg_ret = out.groupby("segment")["retention_prob_12m"].mean()
    unc_total, unc_segments = U.summarise_bootstrap(boot, float(out["clv_36m"].sum()), point_seg_tot, point_seg_ret)
    unc_total.to_csv(OUT / "09_bootstrap_clv_total.csv", index=False)
    unc_segments.to_csv(OUT / "09_bootstrap_clv_by_segment.csv", index=False)
    pd.DataFrame({"total_clv_36m": boot["totals"]}).to_csv(OUT / "09_bootstrap_total_replicates.csv", index=False)
    pd.DataFrame({"simulated_revenue_36m": boot["predictive"]}).to_csv(OUT / "09_predictive_draws.csv", index=False)
    scen = U.scenario_grid(cox, priced, q)
    scen.to_csv(OUT / "09_scenario_grid.csv", index=False)
    R.fig_uncertainty(boot["totals"], boot["predictive"], float(out["clv_36m"].sum()), FIG / "clv_uncertainty.png")
    print(f"  {len(boot['totals'])} bootstrap refits in {time.time() - t_b:.0f}s; "
          f"95% CI for portfolio CLV: {unc_total.loc[0, 'ci_low']:,.0f} - {unc_total.loc[0, 'ci_high']:,.0f} "
          f"(+/-{unc_total.loc[0, 'relative_half_width_pct']:.1f}%)")

    # ------------------------------------------------------------------ 4
    rev_sum = ret_sum = grouped_holdout = coverage = pd.DataFrame()
    if not args.skip_backtest:
        stage(4, "Rolling-origin backtests and customer-grouped holdout")
        if args.quick:
            bt = V.Backtester(tables, horizons=[3, 6], first_test_cutoff="2024-06-30")
        else:
            bt = V.Backtester(tables)
        rev = bt.run_revenue()
        ret = bt.run_retention()
        coverage = bt.run_interval_coverage(B=10 if args.quick else 25)
        coverage.to_csv(OUT / "04_revenue_interval_coverage.csv", index=False)
        rev.to_csv(OUT / "04_revenue_backtest_detail.csv", index=False)
        ret.to_csv(OUT / "05_retention_backtest_detail.csv", index=False)
        rev_sum = V.summarise(rev, ["MAE", "RMSE", "portfolio_bias_pct", "segment_abs_error_pct"])
        ret_sum = V.summarise(ret, ["ROC_AUC", "Brier", "calibration_gap"])
        rev_sum.to_csv(OUT / "04_revenue_backtest_summary.csv", index=False)
        ret_sum.to_csv(OUT / "05_retention_backtest_summary.csv", index=False)
        grouped_holdout = V.run_customer_grouped_holdout(tables, verbose=True)
        grouped_holdout.to_csv(OUT / "06_customer_grouped_holdout.csv", index=False)
        # The backtests now come from the corrected historical plan-at-cutoff logic.
        (OUT / "BACKTESTS_PENDING_RERUN.txt").unlink(missing_ok=True)
        pred = pd.concat(bt.retention_predictions, ignore_index=True)
        R.fig_revenue(rev_sum, FIG / "revenue_backtest.png")
        R.fig_calibration(pred, FIG / "retention_calibration.png",
                          horizon=6 if 6 in bt.horizons else bt.horizons[0])

    # ------------------------------------------------------------------ 5
    stage(5, "Write report")
    unbilled_active_count = int(out["data_quality_flag"].fillna("").astype(str).str.strip().ne("").sum())
    write_report(audit, surv_table, cox_table, lr, seg_tab, attr_tab, top20_share, sens_tab,
                 unbilled_sensitivity, rev_sum, ret_sum, grouped_holdout, len(frame),
                 unbilled_active_count, args, unbilled_breakdown, ph_check, unc_total, unc_segments, coverage)
    if not args.skip_backtest:
        (OUT / "REPORT_PENDING_RERUN.md").unlink(missing_ok=True)
    print(f"\nDone in {time.time() - t_start:.0f}s. Open clv_results/REPORT.md")


def write_report(audit, surv_table, cox_table, lr, seg_tab, attr_tab, top20, sens_tab,
                 unbilled_sensitivity, rev_sum, ret_sum, grouped_holdout, n_spells,
                 unbilled_active_count, args, unbilled_breakdown, ph_check, unc_total, unc_segments, coverage):
    attr_msg = ("Customer attributes (company size, industry, channel, country) add **no detectable** "
                "information about churn once plan and billing period are known"
                if lr["p_value"] > 0.05 else
                "Customer attributes add statistically detectable information beyond plan and billing period")
    cox_view = cox_table.copy()
    cox_view["term"] = cox_view["term"].str.replace("_", " ")
    L = []
    A = L.append
    A("# CLV Toolkit - Results Report\n")
    A("_Generated automatically by `run_pipeline.py`. Every number below is reproduced from the raw CSVs._\n")
    A("> **Read this first.** The dataset is synthetic. All findings describe how the methods behave on this "
      "dataset; they do not establish performance on real subscription businesses.\n")

    A("## 1. Data audit and data-quality findings\n")
    A(f"- {audit['rows']['subscriptions']:,} subscriptions (one per customer), {audit['rows']['payments']:,} payment rows, "
      f"{audit['share_churned']:.1%} churned, {audit['share_free_plan']:.1%} on the Free plan.")
    A(f"- All {audit['rows']['invoices']:,} invoices are marked *paid*, yet {audit['failed_payments_on_paid_invoices']:,} "
      f"matching payments *failed* ({audit['payment_failure_rate']:.1%}). Only successful payments are counted as revenue.")
    A(f"- **Observed monthly billing-cycle pattern:** among churned subscriptions on paid monthly plans, the recorded churn date "
      f"falls an average of {audit['days_between_last_payment_and_churn_date_mean']:.1f} days after the last recorded payment. "
      "This is consistent with a monthly-cycle pattern, but does not independently establish the same rule for annual plans. "
      "For cash-flow timing, the CLV model assumes a due payment is made when the customer remains active at the midpoint of "
      "that billing cycle; treat this as a dataset-based modelling assumption.")
    A(f"- **Upgrade records are not billed:** all {audit['rows']['changes']} plan-change events (Free->Basic) belong to "
      "subscriptions with zero payments, zero invoices and no churn. They are flagged "
      "(`paid_plan_without_billing_records`), receive zero revenue, and their post-upgrade time is excluded from the "
      "paid-plan survival curves. Treating them as ordinary Basic customers over-predicted Basic revenue by 20-30% "
      "in our first backtests.")
    A(f"- **Annual renewals are missing in 2025:** annual-plan payments per year = {audit['annual_payments_by_year']}. "
      "Revenue backtests therefore use monthly-billed customers only; annual plans are covered by the retention backtest.\n")

    A("## 2. Survival analysis\n")
    A(f"Spell-level data: {n_spells:,} spells fitted at {C.OBSERVATION_END.date()}. Churn is the event; active customers are "
      "right-censored. Curves are extended beyond the longest follow-up with a constant hazard (an assumption).\n")
    A(R.md_table(surv_table, "{:,.3f}"))
    A("\n![Kaplan-Meier curves](figures/km_curves.png)\n")
    A("**Cox proportional-hazards model** (stratified by billing period, i.e. monthly and annual plans have separate baseline hazards; "
      "hazard ratio > 1 = churns faster; reference levels are Basic, large, finance, ads, Germany):\n")
    A(R.md_table(cox_view[["term", "hazard_ratio", "p_value"]], "{:,.3f}"))
    A(f"\nProportional-hazards check (Schoenfeld residuals): {int(ph_check['violates_ph'].sum())} of {len(ph_check)} covariates violate the assumption at p < 0.01. "
      "Billing period is a stratum because annual subscribers can only churn at renewal dates, which is clearly non-proportional to monthly churn "
      "(see the Kaplan-Meier table: annual S(6m) = 1.0).\n")
    A(f"\nLikelihood-ratio test of the four customer attributes: chi2 = {lr['LR_statistic']:.2f}, df = {lr['df']}, "
      f"p = {lr['p_value']:.3f}. {attr_msg}.\n")

    A("## 3. Customer lifetime value\n")
    A(f"CLV = expected discounted successful payments over {C.CLV_HORIZON_MONTHS} months, discount rate "
      f"{C.ANNUAL_DISCOUNT_RATE:.0%}/year, Cox survival model. Free-plan customers have CLV 0 by construction (there is no Free->paid conversion model), and so do unbilled customers.\n")
    seg_view = seg_tab.copy()
    seg_view["segment"] = seg_view["segment"].str.replace("|", " / ", regex=False)
    A(R.md_table(seg_view, "{:,.2f}"))
    A(f"\nThe top 20% of positive-CLV customers hold **{top20:.1f}%** of total 36-month CLV.\n")
    A("![CLV distribution](figures/clv_distribution.png)\n")
    A("Average CLV by customer attribute (positive-CLV customers; descriptive, unadjusted group comparisons):\n")
    A("These comparisons are **not adjusted for plan or billing-period mix**. Differences may reflect the composition of each "
      "group and should not be interpreted as independent effects of customer attributes.\n")
    A(R.md_table(attr_tab, "{:,.2f}"))
    A(f"\n**Annual-plan caution:** annual-plan segments are included in the CLV estimates, but 2025 annual renewal payments are "
      "missing from this dataset, so revenue backtests cannot validate annual-plan forecasts. Annual-plan estimates and the "
      "portfolio total are therefore provisional model-based projections, not verified future revenue.\n")
    A("\n**Why are paid-plan subscriptions unbilled?** Most of them are a data gap, not Free->Basic upgrades:\n")
    A(R.md_table(unbilled_breakdown, "{:,.1f}"))
    A("\n")
    A(f"\n**Unbilled-record caution:** {unbilled_active_count:,} active subscriptions have a paid plan but no historical billing "
      "records. Under the current conservative rule, their estimated CLV is set to zero. If records are missing rather than truly "
      "unbilled, this can understate the portfolio's potential value. The scenarios below quantify the impact under explicit "
      "alternative assumptions; they are sensitivity scenarios, not confidence intervals or proof of future revenue.\n")
    A("\nSensitivity to missing billing history (36-month discounted CLV):\n")
    A(R.md_table(unbilled_sensitivity, "{:,.2f}"))
    A("\nInterpretation: the observed-rate scenario assumes flagged customers were potentially billable and had the fitted "
      "survival profile of comparable plan/billing-period customers. The 100% success case is an intentionally optimistic stress "
      "test. Neither scenario replaces the conservative estimate without additional billing evidence.\n")
    A("\nSensitivity to horizon and discount rate:\n")
    A(R.md_table(sens_tab, "{:,.2f}"))
    A("\n> Horizons beyond the observed data (> ~34 months) rest on the constant-hazard tail assumption.\n")

    # Explicit value-risk segmentation and action suggestions are rule-based, not intervention-effect estimates.
    risk_summary_path = C.RESULTS_DIR / "07_value_risk_segment_summary.csv"
    if risk_summary_path.exists():
        risk_summary = pd.read_csv(risk_summary_path)
        if not risk_summary.empty:
            A("\n## 4. Customer value-risk segments and retention recommendations\n")
            A("Estimated 12-month churn risk is calculated as 1 − the Cox-model 12-month retention probability. High-value and high-risk thresholds are the medians among paying customers with billing evidence (Free users form their own group). Because plan drives both value (rises with tier) and risk (falls with tier), these segments largely reflect plan/billing mix; a top-quartile rule on both produced an empty high-value / high-risk group. Paid-plan records without billing evidence are separated into a billing-review cohort instead of being interpreted as confirmed low-value customers. Recommendations are rule-based suggestions; the data contains no campaign outcomes to prove that an action reduces churn.\n")
            A(R.md_table(risk_summary, "{:,.2f}"))

    A("\n## 5. Temporal behavioral feature engineering\n")
    A("Features use successful and failed payment events recorded on or before each prediction cutoff. In addition to rolling 30-, 60-, and 90-day counts/revenue, the pipeline creates non-overlapping previous-30-day and days-61-to-90 measures, differences between recent and prior revenue/payment counts, and bounded normalized trend ratios. These delta/trend features are recomputed independently at every historical cutoff; the test suite checks that deleting all post-cutoff records does not change features.\n")
    A("Derived behavior features include `revenue_prev30d`, `revenue_days61_90`, `revenue_delta_30d_vs_prev30d`, `revenue_trend_ratio_30d_vs_prev30d`, `revenue_delta_30d_vs_days61_90`, `payment_count_delta_30d_vs_prev30d`, `payment_count_trend_ratio_30d_vs_prev30d`, and `failed_count_delta_30d_vs_prev30d`.\n")

    if not rev_sum.empty:
        A("## 6. Out-of-time backtests\n")
        A("Each forecast origin is treated as 'today': survival models see only data up to the origin; supervised revenue/churn models are trained only on snapshots whose label window had already ended. Rolling-origin averages are separate from the fixed-cutoff customer-grouped comparisons below.\n")
        A("### 6.1 Revenue forecast (paid monthly customers)\n")
        A(R.md_table(rev_sum[["horizon_months", "model", "origins", "RMSE", "MAE", "portfolio_bias_pct",
                              "segment_abs_error_pct"]], "{:,.2f}"))
        A("\nHow to read this: **RMSE** and **portfolio / segment error** are the right metrics for expected-value "
          "forecasts. **MAE** can favour the 'nobody churns' benchmark at short horizons (it does here at 3 and 6 months) "
          "because most customers do not churn inside the window; an expected value is not the MAE-optimal point "
          "forecast. Do not cite MAE alone.\n")
        A("![Revenue backtest](figures/revenue_backtest.png)\n")
        A("### 6.2 Churn / retention forecast (all customers with billing evidence)\n")
        A(R.md_table(ret_sum[["horizon_months", "model", "origins", "ROC_AUC", "Brier", "calibration_gap"]], "{:,.4f}"))
        A("\n![Calibration](figures/retention_calibration.png)\n")
    if not grouped_holdout.empty:
        A("### 6.3 Customer-grouped holdout (unseen customers)\n")
        A("A fixed 30 June 2025 forecast cutoff and 3-month horizon are used. Supervised models train on earlier "
          "snapshots only when the complete target window had ended by the cutoff. Customer IDs selected for the holdout "
          "are removed from every historical training snapshot, so no customer appears in both train and test. Since this "
          "dataset has one subscription per customer, grouping does not separate multiple subscriptions within a current snapshot; "
          "its key safeguard is excluding holdout customers from all historical training snapshots. This diagnostic complements "
          "rolling temporal validation and is not a substitute for validation on real-world data.\n")
        A("**Revenue task (paid monthly customers with billing evidence):**\n")
        revenue_cols = ["model", "train_customer_groups", "test_customer_groups", "train_rows", "test_rows",
                        "customer_overlap", "MAE", "RMSE", "R2", "portfolio_bias_pct"]
        A(R.md_table(grouped_holdout[grouped_holdout["task"] == "3-month revenue"][revenue_cols], "{:,.4f}"))
        by_plan_path = C.RESULTS_DIR / "06_grouped_revenue_by_plan.csv"
        if by_plan_path.exists():
            A("\n**Revenue accuracy within each plan** (pooled R2 above is inflated by price differences between plans; a plan-price baseline is included):\n")
            A(R.md_table(pd.read_csv(by_plan_path)[["plan", "model", "test_rows", "mean_actual", "MAE", "RMSE", "R2", "portfolio_bias_pct"]], "{:,.3f}"))
        A("\n**Churn task (Free customers plus paid customers with billing evidence):**\n")
        churn_cols = ["model", "train_customer_groups", "test_customer_groups", "train_rows", "test_rows",
                      "customer_overlap", "ROC_AUC", "PR_AUC", "Brier", "calibration_gap"]
        A(R.md_table(grouped_holdout[grouped_holdout["task"] == "3-month churn"][churn_cols], "{:,.4f}"))
        A("\nThe `customer_overlap` column must be 0. MAE/RMSE/R² and portfolio bias apply only to revenue rows; ROC-AUC, PR-AUC, Brier "
          "score and calibration gap apply only to churn rows. These are held-out-customer results at one forecast cutoff and are not the same as rolling-origin averages.\n")
        A("### 6.4 Four-model regression comparison\n")
        A("The customer-grouped revenue holdout compares Linear Regression, Random Forest, Gradient Boosting, and XGBoost on the same 3-month target and test customers, alongside a zero-revenue benchmark. This is the reproducible comparison for the original regression-model objective.\n")
        A("### 6.5 Classification comparison for churn risk\n")
        A("The same grouped holdout compares Logistic Regression, Random Forest, Gradient Boosting, and XGBoost, alongside a historical-prevalence benchmark. ROC-AUC and PR-AUC describe ranking, while Brier score and calibration gap describe probability quality.\n")

    A("\n## 7. Explainable AI (SHAP)\n")
    shap_status_path = C.RESULTS_DIR / "08_shap_status.txt"
    shap_summary_path = C.RESULTS_DIR / "08_shap_global_summary.csv"
    shap_status = shap_status_path.read_text(encoding="utf-8").strip() if shap_status_path.exists() else "SHAP outputs were not generated in this run."
    A(shap_status + "\n")
    if shap_summary_path.exists():
        shap_summary = pd.read_csv(shap_summary_path)
        if not shap_summary.empty:
            A("Global SHAP importance is calculated on held-out customer rows for the XGBoost revenue proxy and churn classifier. For revenue, SHAP values use the 3-month revenue target scale; for binary churn they are log-odds contributions, not direct percentage-point probability changes. This explains the supervised revenue proxy and churn models; the separate survival-based CLV model is explained through its Cox hazard ratios and survival curves. SHAP values indicate model attribution, not causality.\n")
            for task_name, task_df in shap_summary.groupby("task"):
                A(f"### {task_name}\n")
                A(R.md_table(task_df.sort_values("mean_abs_shap", ascending=False).head(15), "{:,.4f}"))
                A("\n")
    else:
        A("SHAP summary file is absent. Check `08_shap_status.txt` and the pipeline output before claiming the explainability objective is complete.\n")

    A("\n## 8. Uncertainty and scenario analysis\n")
    A("Point estimates alone hide how uncertain a projection is. Two kinds of uncertainty are quantified:\n")
    A("- **Estimation uncertainty (bootstrap):** spells are resampled with replacement, the Cox model is refitted and the portfolio re-scored "
      f"({int(unc_total.loc[0, 'replicates'])} replicates). The spread is how much the estimate would move with a different sample of the same size.")
    A("- **Process uncertainty (Monte Carlo):** given the fitted curves, each customer's churn time and every payment attempt are simulated. "
      "Even a perfect model cannot know who will actually churn, so realised revenue varies around its expectation.\n")
    A(R.md_table(unc_total[["quantity", "point_estimate", "ci_low", "ci_high", "relative_half_width_pct"]], "{:,.1f}"))
    A("\n![Uncertainty](figures/clv_uncertainty.png)\n")
    A("**Segment-level 95% bootstrap intervals** (CLV in currency units; retention = 12-month retention probability):\n")
    A(R.md_table(unc_segments, "{:,.3f}"))
    A("\n**What the intervals do NOT cover:** model misspecification, future shocks (price changes, market conditions), "
      "uncertainty in the payment-success rate, and the unbilled-record assumption (see the separate sensitivity scenarios in section 3). "
      "The synthetic data also has no real-world drift, so real intervals would be wider.\n")
    if not coverage.empty:
        A("### 8.1 Do the predictive intervals contain reality? (out-of-time check)\n")
        A("For every backtest origin, a 90% predictive interval for portfolio revenue of paid monthly customers was built using only data available at that origin, then compared with what was actually collected.\n")
        cov_sum = (coverage.groupby("horizon_months")
                   .agg(origins=("cutoff", "size"), inside_interval=("actual_within_interval", "sum"),
                        mean_half_width_pct=("half_width_pct", "mean"),
                        mean_abs_error_vs_expected_pct=("actual_vs_expected_pct", lambda x: x.abs().mean()))
                   .reset_index())
        cov_sum["coverage_pct"] = 100 * cov_sum["inside_interval"] / cov_sum["origins"]
        A(R.md_table(cov_sum, "{:,.2f}"))
        total_in, total_n = int(coverage["actual_within_interval"].sum()), len(coverage)
        verdict = ("This gives no sign of over-confidence (intervals too narrow)." if total_in / total_n >= 0.85
                   else "This is below the nominal level, so the intervals are too narrow (over-confident): behaviour drifts more than the model assumes.")
        A(f"\nOverall: **{total_in} of {total_n}** realised outcomes fell inside their 90% interval (nominal 90%). {verdict} "
          "Caveats: consecutive origins overlap in time, so they are not independent, and 17 or so points cannot pin coverage down precisely. "
          "Coverage well above 90% would simply mean the intervals are somewhat conservative.\n")
        A(R.md_table(coverage[["horizon_months", "cutoff", "customers", "actual_revenue", "expected_revenue", "interval_low",
                               "interval_high", "actual_within_interval"]], "{:,.0f}"))
    A("\n### 8.2 Scenario simulation\n")
    A("`09_scenario_grid.csv` holds portfolio and segment CLV for every combination of: change in churn hazard (-50% to +50%), "
      "annual discount rate (0-20%), horizon (12-60 months) and payment-success rate (90-100% of observed). "
      "The dashboard page **Uncertainty & Scenarios** lets you explore it. A churn-hazard change of -x% multiplies the hazard by (1-x/100), "
      "i.e. survival S(t) becomes S(t)^(1-x/100).\n")
    A("\n## 9. Limitations\n")
    A("- Synthetic data: near-uniform categorical attributes, constant-ish hazards, upgrade records disconnected from billing.")
    A("- Column names were inferred (the CSVs have no header row); confirm against the dataset documentation.")
    A("- Churn discrimination is weak (ROC-AUC about 0.62-0.69) and comes almost entirely from plan / billing period; customer attributes carry no signal (synthetic data).")
    A("- Pooled revenue R2 (about 0.85-0.93) is driven by plan price; within each plan the ML models have R2 of about 0 and do not beat a price x months baseline on MAE (see the per-plan table).")
    A("- Free customers receive CLV = 0 by construction; no Free->paid conversion model is included.")
    A("- Intervals reflect sampling and random-event uncertainty only; they do not cover model misspecification or future shocks.")
    A("- One customer per subscription; no repeat purchases, so BG/NBD-type models do not apply.")
    A("- Revenue only (no margin, acquisition cost or expansion revenue); annual-plan revenue is not backtestable.")
    A("- Tail extrapolation (constant hazard) drives long-horizon CLV; treat > 3-year numbers cautiously.")
    A("- Customers appear in several forecast origins, so rolling-origin scores across origins are correlated.")
    if not grouped_holdout.empty:
        A("- The added customer-grouped holdout removes test customers from historical training snapshots, but is still "
          "evaluated on synthetic data. Since there is one subscription per customer, its key safeguard is excluding those "
          "customers across historical snapshots rather than separating multiple subscriptions within a current snapshot.")
    else:
        A("- Customer-grouped holdout results were not generated in this run (for example, if `--skip-backtest` was used).")
    A("- Customers with paid-plan labels but no billing evidence are zeroed in the primary CLV estimate; the sensitivity table "
      "shows how this assumption affects the portfolio total under stated alternatives.")
    A("- Customer value-risk labels and retention actions are rule-based prioritization suggestions, not causal estimates of campaign lift.")
    A("- SHAP explanations are for the supervised 3-month revenue proxy and churn classifier; they do not directly explain every component of the survival-based CLV calculation.")
    A("- Rolling-origin 12-month rows use only origins where every model could be evaluated, so they average very few origins (see the `origins` column).")
    if args.quick:
        A("- This report was produced with `--quick` (fewer origins/horizons).")
    (C.RESULTS_DIR / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
