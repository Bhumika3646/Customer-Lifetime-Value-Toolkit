
from pathlib import Path
import pandas as pd

# Project output folder
OUTPUT_DIR = Path("outputs")

FILES = {
    "TRAIN": OUTPUT_DIR / "temporal_train_snapshot_2024-06-30.csv",
    "TEST": OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv",
}

TARGET_CANDIDATES = [
    "target_future_90d_revenue",
    "future_revenue_90d",
    "future_90d_revenue",
    "target_revenue_90d",
    "revenue_next_90d",
]

def find_target(df):
    """Find the future 90-day revenue target column."""
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    # Show likely target names if the exact name differs.
    likely = [
        col for col in df.columns
        if "future" in col.lower()
        or "target" in col.lower()
        or ("revenue" in col.lower() and "90" in col.lower())
    ]

    if likely:
        return likely[0]

    return None


def audit_snapshot(label, file_path):
    print("\n" + "=" * 70)
    print(f"{label} SNAPSHOT")
    print("=" * 70)

    if not file_path.exists():
        print(f"FILE NOT FOUND: {file_path}")
        return None

    df = pd.read_csv(file_path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nAvailable columns:")
    print(", ".join(df.columns))

    target = find_target(df)

    if target is None:
        print("\nCould not automatically identify the future revenue target.")
        print("Look at the column list above for the target column name.")
        return df

    df[target] = pd.to_numeric(df[target], errors="coerce")

    print(f"\nRevenue target column: {target}")
    print(f"Missing targets: {df[target].isna().sum():,}")
    print(f"Zero future revenue: {(df[target] == 0).sum():,}")
    print(f"Positive future revenue: {(df[target] > 0).sum():,}")
    print(f"Negative future revenue: {(df[target] < 0).sum():,}")
    print(f"Mean future revenue: {df[target].mean():.2f}")
    print(f"Median future revenue: {df[target].median():.2f}")
    print(f"Maximum future revenue: {df[target].max():.2f}")

    # Find plan and historical-revenue columns.
    plan_col = next(
        (c for c in df.columns if c.lower() == "billing_period_at_cutoff"),
        None
    )
    history_col = next(
        (c for c in df.columns
         if c.lower() == "historical_successful_revenue"),
        None
    )

    if plan_col:
        print("\nFuture revenue by billing period:")
        summary = df.groupby(plan_col, dropna=False)[target].agg(
            customers="size",
            zero_revenue=lambda s: (s == 0).sum(),
            mean_revenue="mean",
            median_revenue="median",
            total_revenue="sum"
        )
        print(summary.round(2).to_string())

    if history_col:
        df[history_col] = pd.to_numeric(
            df[history_col], errors="coerce"
        ).fillna(0)

        historical = df[history_col] > 0
        future_zero = df[target] == 0

        print("\nHistorical vs future revenue:")
        print(f"Customers with historical revenue > 0: {historical.sum():,}")
        print(
            "Those customers with zero future revenue: "
            f"{(historical & future_zero).sum():,}"
        )
        print(
            "Percentage of historically paying customers with zero future "
            f"revenue: {(future_zero[historical].mean() * 100):.2f}%"
            if historical.any()
            else "No customers with historical revenue > 0."
        )

    return df


def main():
    snapshots = {}

    for label, path in FILES.items():
        snapshots[label] = audit_snapshot(label, path)

    train = snapshots.get("TRAIN")
    test = snapshots.get("TEST")

    if train is not None and test is not None:
        print("\n" + "=" * 70)
        print("TRAIN VS TEST COMPARISON")
        print("=" * 70)

        for label, df in [("TRAIN", train), ("TEST", test)]:
            target = find_target(df)
            if target:
                values = pd.to_numeric(df[target], errors="coerce")
                print(
                    f"{label}: rows={len(df):,}, "
                    f"zero share={(values == 0).mean() * 100:.2f}%, "
                    f"mean={values.mean():.2f}, "
                    f"median={values.median():.2f}"
                )

        train_plan = "billing_period_at_cutoff"
        test_plan = "billing_period_at_cutoff"

        if train_plan in train.columns and test_plan in test.columns:
            print("\nPlan distribution (%):")
            print("TRAIN:")
            print(
                (train[train_plan].value_counts(normalize=True) * 100)
                .round(2).to_string()
            )
            print("\nTEST:")
            print(
                (test[test_plan].value_counts(normalize=True) * 100)
                .round(2).to_string()
            )

    print("\nAudit complete. No project data has been modified.")


if __name__ == "__main__":
    main()
