
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

FILES = [
    OUT / "temporal_train_snapshot_2024-06-30.csv",
    OUT / "temporal_test_snapshot_2025-06-30.csv",
]

CHECK_COLUMNS = [
    "plan_id",
    "plan_id_at_cutoff",
    "plan_id_lookup",
    "plan_name_at_cutoff",
    "billing_period_at_cutoff",
    "plan_price_at_cutoff",
    "status",
]

for file in FILES:
    df = pd.read_csv(file, low_memory=False)

    period = (
        df["billing_period_at_cutoff"]
        .astype("string").str.strip().str.lower()
    )
    df = df.loc[period.eq("monthly")].copy()

    print("\n" + "=" * 75)
    print(file.name, "| Monthly customers:", len(df))
    print("=" * 75)

    available = [c for c in CHECK_COLUMNS if c in df.columns]

    print("\nPlan-related value counts:")
    for col in available:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(12).to_string())

    if {"plan_id", "plan_id_at_cutoff"}.issubset(df.columns):
        a = df["plan_id"].astype("string").str.strip()
        b = df["plan_id_at_cutoff"].astype("string").str.strip()
        valid = a.notna() & b.notna()

        print("\nplan_id vs plan_id_at_cutoff:")
        print("Rows with both values:", int(valid.sum()))
        if valid.any():
            print(
                "Exact agreement (%):",
                round((a[valid] == b[valid]).mean() * 100, 2)
            )
            print("Different-value examples:")
            print(
                df.loc[valid & (a != b),
                       ["plan_id", "plan_id_at_cutoff"]]
                .drop_duplicates()
                .head(15)
                .to_string(index=False)
            )

    print("\nPlan price by plan ID:")
    cols = [
        c for c in [
            "plan_id_at_cutoff",
            "plan_price_at_cutoff",
            "target_future_90d_revenue",
        ]
        if c in df.columns
    ]
    if len(cols) == 3:
        print(
            df.groupby(cols[:2], dropna=False)[cols[2]]
            .agg(["count", "mean", "min", "max"])
            .round(2)
            .to_string()
        )

print("\nCheck complete. No files were modified.")
