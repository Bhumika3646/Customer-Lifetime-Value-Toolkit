
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"

test = pd.read_csv(OUT / "temporal_test_snapshot_2025-06-30.csv")

print("Test snapshot rows:", len(test))
print("\nPlan distribution and future revenue:")

summary = (
    test.groupby(
        ["billing_period_at_cutoff", "plan_name_at_cutoff"],
        dropna=False,
    )
    .agg(
        customers=("customer_id", "nunique"),
        historical_revenue=("historical_successful_revenue", "sum"),
        revenue_30d=("revenue_30d", "sum"),
        revenue_60d=("revenue_60d", "sum"),
        revenue_90d=("revenue_90d", "sum"),
        future_revenue=("target_future_90d_revenue", "sum"),
        customers_with_future_revenue=(
            "target_future_90d_revenue",
            lambda s: int((s > 0).sum()),
        ),
    )
    .reset_index()
)

print(summary.to_string(index=False))

print("\nAnnual-plan rows with positive historical revenue:")
annual = test[test["billing_period_at_cutoff"] == "annual"]
print("Annual rows:", len(annual))
print(
    "Positive historical revenue:",
    int((annual["historical_successful_revenue"] > 0).sum()),
)
print(
    "Positive recent 90-day revenue:",
    int((annual["revenue_90d"] > 0).sum()),
)
print(
    "Positive future 90-day revenue:",
    int((annual["target_future_90d_revenue"] > 0).sum()),
)

print("\nAudit complete. No files were modified.")
