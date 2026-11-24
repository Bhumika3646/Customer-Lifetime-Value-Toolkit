# CLV Toolkit — Final report interpretation updates

This update tightens the report and dashboard wording without changing the underlying survival or revenue models.

1. **Billing-cycle evidence is scoped correctly.** The mean delay between churn date and last recorded payment is described as an observed pattern among churned paid monthly subscriptions. It is not presented as a rule independently proven for annual plans.
2. **Annual-plan CLV is marked provisional.** Because annual-plan renewal payments are missing from 2025, the revenue backtest cannot validate annual-plan forecasts; the annual segments and total portfolio CLV are model-based projections.
3. **CLV segment count is renamed.** `positive_clv_customers` counts records with positive estimated 36-month CLV. It does not assert that every such record has independently verified successful-payment history.
4. **Customer attribute comparisons are described as unadjusted.** Their spread is descriptive and may reflect subscription-plan and billing-period composition; it is not described as a plan-standardized effect.
5. **Unbilled records are caveated.** The current conservative rule assigns zero CLV to active paid-plan subscriptions with no historical billing records, with an explicit note that this can understate potential value if records are incomplete.
6. CLV figures and the report now use “positive-CLV customers” rather than “paying customers” where the available field only establishes positive model-estimated CLV.

## Validation status

- `app.py`, `run_pipeline.py`, `clvkit/`, and `tests/` passed Python syntax compilation in the editing environment.
- The full test suite and pipeline were **not rerun in this environment**, because `lifelines` is not installed here and package installation is unavailable. The project includes the dependency in `requirements.txt`; run the pipeline and tests locally before using newly generated metrics in a report.

## Regenerate locally

```powershell
python run_pipeline.py
python -m pytest -q
python -m streamlit run app.py
```

7. Added `03_unbilled_billing_sensitivity.csv` with a conservative scenario, an observed billing-period payment-success scenario, and a 100%-success stress case for active paid-plan subscriptions lacking billing records. The alternative scenarios are explicitly labelled as assumptions, not validated future revenue or confidence intervals.
8. Added `06_customer_grouped_holdout.csv`: supervised revenue/churn evaluation at 30 June 2025 uses only earlier snapshots with fully observed target windows. Holdout customer IDs are excluded from every historical training snapshot and overlap is asserted to be zero.
9. The report and dashboard now display the missing-billing sensitivity and separate customer-grouped results from rolling temporal validation. The dataset has one subscription per customer, so the grouped split's key additional protection is removing held-out customer IDs from all historical snapshots.

## Validation of the latest robustness update

- Python syntax compilation succeeded for `app.py`, `run_pipeline.py`, all `clvkit` modules, and tests.
- Targeted tests for the new sensitivity helper, customer-group split helper, and existing leakage guard passed (3 passed).
- A smoke run on the bundled raw data completed the new customer-grouped supervised evaluation: 658 held-out monthly-paid revenue customers, 2,181 held-out churn-cohort customers, and zero train/test customer overlap.
- The full pipeline and complete test suite were not run in this editing environment because `lifelines` could not be installed here. Run the full pipeline and `pytest` on the project computer before using regenerated report values.

## Original-objective alignment update

10. Added cutoff-safe temporal behavior features: non-overlapping prior-30-day and days-61-to-90 payment/revenue counts, deltas, and normalized trend ratios.
11. Expanded the customer-grouped holdout to compare Linear Regression, Random Forest, Gradient Boosting, and XGBoost for revenue, plus four churn classifiers, on the same held-out customer cohorts. Regression reports MAE/RMSE/R²/portfolio bias; churn reports ROC-AUC/PR-AUC/Brier/calibration gap.
12. Added SHAP global and local outputs for the grouped-holdout XGBoost revenue-proxy and churn models. The report warns that these explanations are not a direct decomposition of the Cox survival CLV calculation.
13. Added `07_value_risk_segment_summary.csv` and `07_customer_retention_recommendations.csv`, plus the Retention Recommendations dashboard page. Risk/CLV thresholds use eligible-cohort 75th percentiles; missing-billing customers are routed to billing review rather than low-value classification.
14. Added tests for delta-feature integrity, the four regression/comparison pipelines, churn classifier probabilities, recommendation segments, and SHAP source-feature mapping.

The current editor environment could not run the complete survival pipeline because `lifelines` is unavailable. The grouped model comparison and SHAP were smoke-tested on the bundled raw dataset: all model families returned predictions, SHAP summaries were generated for revenue and churn, and customer overlap was zero. On the user's Windows environment, install `requirements.txt`, then run the full pipeline and test suite before using new report numbers.

