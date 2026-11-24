
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

TARGET = "target_future_90d_revenue"


def inspect_snapshot(path, label):
    df = pd.read_csv(path, low_memory=False)

    period = (
        df["billing_period_at_cutoff"]
        .astype("string").str.strip().str.lower()
    )
    df = df.loc[period.eq("monthly")].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["plan_price_at_cutoff"] = pd.to_numeric(
        df["plan_price_at_cutoff"], errors="coerce"
    )
    df = df.dropna(subset=[TARGET, "plan_price_at_cutoff"])

    print("\n" + "=" * 65)
    print(label)
    print("=" * 65)

    # Does the target match 3 times the monthly plan price?
    df["three_month_price"] = 3 * df["plan_price_at_cutoff"]
    df["matches_three_month_price"] = np.isclose(
        df[TARGET], df["three_month_price"], atol=1e-8
    )

    print("\nResults grouped by plan price:")
    summary = df.groupby("plan_price_at_cutoff").agg(
        customers=(TARGET, "size"),
        zero_targets=(TARGET, lambda s: int((s == 0).sum())),
        positive_targets=(TARGET, lambda s: int((s > 0).sum())),
        mean_target=(TARGET, "mean"),
        max_target=(TARGET, "max"),
        exact_three_month_matches=("matches_three_month_price", "sum"),
    )
    summary["zero_target_pct"] = (
        summary["zero_targets"] / summary["customers"] * 100
    )
    summary["three_month_match_pct"] = (
        summary["exact_three_month_matches"] / summary["customers"] * 100
    )
    print(summary.round(2).to_string())

    print("\nTarget value counts:")
    print(
        df[TARGET].value_counts()
        .sort_index()
        .tail(20)
        .to_string()
    )

    print("\nOverall:")
    print(f"Monthly customers: {len(df):,}")
    print(f"Targets equal to 3 x plan price: "
          f"{df['matches_three_month_price'].sum():,}")
    print(f"Percentage matching: "
          f"{df['matches_three_month_price'].mean() * 100:.2f}%")

    positive = df[df[TARGET] > 0].copy()
    if not positive.empty:
        print("\nPositive targets only:")
        print(f"Positive customers: {len(positive):,}")
        print(
            "Positive targets equal to 3 x plan price: "
            f"{positive['matches_three_month_price'].sum():,} "
            f"({positive['matches_three_month_price'].mean() * 100:.2f}%)"
        )


def main():
    inspect_snapshot(
        OUT / "temporal_train_snapshot_2024-06-30.csv",
        "TRAINING SNAPSHOT — MONTHLY PLANS",
    )
    inspect_snapshot(
        OUT / "temporal_test_snapshot_2025-06-30.csv",
        "TEST SNAPSHOT — MONTHLY PLANS",
    )

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()
