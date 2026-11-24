# Final local verification checklist

Run these commands from the folder containing `app.py`, `run_pipeline.py`, and `requirements.txt`:

```powershell
python run_pipeline.py
python -m pytest -q
python -m streamlit run app.py
```

After `run_pipeline.py` finishes, confirm these new output files exist:

- `clv_results/03_unbilled_billing_sensitivity.csv`
- `clv_results/06_customer_grouped_holdout.csv`
- `clv_results/07_value_risk_segment_summary.csv`
- `clv_results/07_customer_retention_recommendations.csv`
- `clv_results/08_shap_global_summary.csv`
- `clv_results/08_shap_local_examples.csv`
- `clv_results/08_shap_status.txt`
- `clv_results/REPORT.md`

The customer-grouped table should report `customer_overlap = 0` for every row. It uses a fixed 30 June 2025 cutoff and 3-month horizon, and excludes the held-out customer IDs from all historical training snapshots whose labels were available by the cutoff. This is a diagnostic on the synthetic dataset, not evidence of real-world performance.

The billing sensitivity table should include three scenarios: conservative zero CLV for subscriptions without billing history; a scenario using observed billing-period payment-success rates; and a 100%-success stress case. The latter two are assumptions, not confidence intervals or verified future revenue.

The tests now include objective-alignment checks for the 30/60/90-day behavioral deltas, four regression models, four churn classifiers, value-risk recommendations, and SHAP feature-name mapping. Expected full suite count: 18 passing cases, including parametrized tests. If the pipeline or tests fail, save the entire error output before editing or deleting any generated files.
