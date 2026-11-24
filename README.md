# SubscriptIQ — Customer Lifetime Value Toolkit (Final-Year Project)

Survival-based customer lifetime value (CLV), churn risk, explainability and retention
recommendations for a subscription business, plus a Streamlit dashboard.
**The dataset is synthetic**; every finding describes how the methods behave on this data only.

## Quick start (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py          # ~5-6 minutes (200 bootstrap refits); regenerates clv_results/ incl. REPORT.md
python run_pipeline.py --quick  # ~2 minutes, fewer origins and bootstrap replicates (for a fast check)
python -m pytest -q             # 25 tests
python -m streamlit run app.py
```

## Folder map

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (8 pages, incl. Survival-Based CLV, Retention Recommendations and Uncertainty & Scenarios) |
| `clvkit/` | Library: data audit, leakage-safe features, Kaplan–Meier, stratified Cox, CLV maths, ML comparators, validation, SHAP, recommendations, **uncertainty and scenarios** (`uncertainty.py`) |
| `run_pipeline.py` | One command that produces all results and `clv_results/REPORT.md` |
| `clv_results/` | Generated results (current outputs of `run_pipeline.py`) |
| `outputs/` | Saved outputs of the earlier 90-day revenue/churn experiments used by the dashboard |
| `raw_data/` | The seven raw source CSVs (headerless; column names are inferred, see `clvkit/config.py`) |
| `tests/` | 25 automated tests (leakage, survival, CLV maths, grouped split, recommendations, SHAP mapping, uncertainty and scenarios) |
| `legacy_scripts/` | Archived exploratory scripts, kept for traceability only |
| `docs/archive/` | Historical change logs |

## Method summary

1. **Survival models:** Global KM, segment KM (plan × billing), and a **Cox PH model stratified by billing period**
   (annual subscribers can only churn at renewal, so monthly and annual hazards are not proportional; within strata a
   Schoenfeld test finds no violations). Delayed entry handles plan upgrades; curves use a constant-hazard tail beyond follow-up.
2. **CLV** = expected discounted successful payments over 36 months (10 %/yr), revenue only (not profit).
3. **Validation:** rolling-origin out-of-time backtests (models only see data up to each origin), plus a fixed-cutoff
   customer-grouped holdout comparing Linear Regression, Random Forest, Gradient Boosting and XGBoost against zero-revenue and
   plan-price baselines.
4. **Explainability:** SHAP for the supervised XGBoost revenue and churn models (not for the Cox CLV itself).
5. **Uncertainty and scenarios:** 200-replicate bootstrap (re-fit Cox on resampled spells) gives confidence intervals for portfolio and
   segment CLV and retention; a Monte Carlo simulation of every customer's churn and payments gives a predictive interval for
   collected revenue; an out-of-time backtest checks whether those 90% intervals contain what actually happened; and a scenario
   grid (churn hazard ±50 %, discount rate, horizon, payment success) powers the dashboard's what-if simulator.
6. **Retention:** rule-based segments from CLV and 12-month churn risk. "High" = at/above the median among paying customers
   with billing evidence. Free users and unbilled paid-plan records are separate groups.

## Key results (all reproducible with `run_pipeline.py`)

- Portfolio 36-month CLV ≈ 4.35 M with a **95 % bootstrap interval of about ±3 %** (4.20 M – 4.46 M); segment intervals are in
  `clv_results/09_bootstrap_clv_by_segment.csv`. The 90 % predictive intervals contained the realised revenue in all 17 out-of-time
  backtest checks (`04_revenue_interval_coverage.csv`; origins overlap, so this is a rough check).

- Survival CLV forecasts have ~0–2 % portfolio bias at 3/6/12 months vs +14–29 % for a "nobody churns" benchmark, with RMSE
  equal to or better than gradient boosting.
- Customer attributes add no churn information beyond plan (likelihood-ratio p ≈ 0.18); Cox ≈ segment KM.
- Churn discrimination is modest (ROC-AUC ≈ 0.65–0.69) and comes from plan/billing period only.
- Pooled revenue R² (≈ 0.85) is driven by plan price; within each plan R² ≈ 0 and a price × months baseline has lower MAE.
  See `clv_results/06_grouped_revenue_by_plan.csv`.

## Important data caveats

- **2,786 active paid-plan subscriptions have no billing rows and receive CLV = 0** (conservative rule). Cause breakdown
  (`clv_results/03_unbilled_breakdown.csv`): 62 % annual plans started in 2025 (2025 annual payments are missing from the
  data), 28 % Free→Basic upgrades, 10 % monthly plans with no payment rows. Including them would raise portfolio CLV by
  about 43 % (`03_unbilled_billing_sensitivity.csv`; an assumption-based scenario, not a confidence interval).
- Annual-plan CLV and the portfolio total are provisional because annual renewals cannot be backtested.
- Free customers have CLV = 0 by construction (no Free→paid conversion model).
- The 90-day revenue forecast (30 Jun 2025 snapshot, monthly plans) and the 36-month CLV (31 Dec 2025 snapshot) have
  different scopes and are not comparable. Do not call the 90-day figure "lifetime value".
- Intervals reflect sampling and random-event uncertainty only. They do not cover model misspecification, future shocks, the
  payment-success rate, or the missing-billing assumption.
- Leakage safeguards: features use only rows ≤ cutoff (tested by deleting all future rows); plan-at-cutoff is reconstructed
  so future upgrades cannot leak; final status / churn reason are never features.
- Retention actions are rule-based suggestions; the data has no campaign outcomes to estimate their effect.
