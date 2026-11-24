
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. FILES
# ============================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

TRAIN_FILE = OUT / "temporal_train_snapshot_2024-06-30.csv"
TEST_FILE = OUT / "temporal_test_snapshot_2025-06-30.csv"

TARGET = "target_future_90d_revenue"


# ============================================================
# 2. HELPERS
# ============================================================

def metrics(y, prediction):
    prediction = np.asarray(prediction, dtype=float)
    return {
        "MAE": mean_absolute_error(y, prediction),
        "RMSE": np.sqrt(mean_squared_error(y, prediction)),
        "R2": r2_score(y, prediction),
    }


def check_prediction(name, y, prediction):
    result = metrics(y, prediction)
    print(
        f"{name}: MAE={result['MAE']:.4f}, "
        f"RMSE={result['RMSE']:.4f}, "
        f"R2={result['R2']:.4f}"
    )


def monthly_only(path):
    df = pd.read_csv(path, low_memory=False)

    required = [TARGET, "billing_period_at_cutoff"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    period = (
        df["billing_period_at_cutoff"]
        .astype("string").str.strip().str.lower()
    )

    df = df.loc[period.eq("monthly")].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    return df


def compare_feature(df, feature, target=TARGET):
    if feature not in df.columns:
        print(f"{feature}: not present")
        return

    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = x.notna() & y.notna()

    if valid.sum() == 0:
        print(f"{feature}: no usable numeric rows")
        return

    x = x[valid]
    y = y[valid]

    print(f"\nFeature: {feature}")
    print(f"  Valid rows: {len(x):,}")
    print(f"  Correlation with target: {x.corr(y):.4f}")

    if x.nunique() > 1:
        check_prediction(
            f"  Simple one-feature linear prediction ({feature})",
            y,
            np.polyval(np.polyfit(x, y, 1), x),
        )

    exact = np.isclose(x.to_numpy(), y.to_numpy(), atol=1e-8)
    print(f"  Exact target matches: {exact.sum():,} / {len(y):,}")


# ============================================================
# 3. AUDIT
# ============================================================

def main():
    print("=" * 70)
    print("MONTHLY REVENUE TARGET LEAKAGE / RECONSTRUCTION AUDIT")
    print("=" * 70)

    train = monthly_only(TRAIN_FILE)
    test = monthly_only(TEST_FILE)

    for name, df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{'=' * 25} {name} {'=' * 25}")
        y = df[TARGET]

        print(f"Rows: {len(df):,}")
        print(f"Target mean: {y.mean():.4f}")
        print(f"Target median: {y.median():.4f}")
        print(f"Zero targets: {(y == 0).sum():,} ({(y == 0).mean():.2%})")
        print(f"Positive targets: {(y > 0).sum():,}")

        # A mean baseline is a useful reference point.
        mean_prediction = np.full(len(y), y.mean())
        check_prediction("Mean baseline on this same dataset", y, mean_prediction)

        print("\nPotentially related features:")
        candidates = [
            "revenue_30d",
            "revenue_60d",
            "revenue_90d",
            "historical_successful_revenue",
            "historical_successful_payment_count",
            "payment_count_30d",
            "payment_count_60d",
            "payment_count_90d",
            "plan_price_at_cutoff",
            "tenure_days",
            "subscription_age_days",
        ]

        for feature in candidates:
            compare_feature(df, feature)

        # Test simple, explicitly labeled heuristic formulas.
        # These are diagnostics, not assumed valid prediction methods.
        print("\nSimple formula checks:")

        if "revenue_90d" in df.columns:
            x = pd.to_numeric(df["revenue_90d"], errors="coerce")
            valid = x.notna()
            if valid.any():
                print("Using historical revenue_90d directly:")
                check_prediction(
                    "  Historical revenue_90d as prediction",
                    y[valid],
                    x[valid],
                )

        if "plan_price_at_cutoff" in df.columns:
            price = pd.to_numeric(
                df["plan_price_at_cutoff"], errors="coerce"
            )
            valid = price.notna()
            if valid.any():
                print("Using plan price directly:")
                check_prediction(
                    "  Plan price as prediction",
                    y[valid],
                    price[valid],
                )

        # Group-level target patterns can reveal a synthetic
        # data-generation rule or a strong plan-specific effect.
        if "plan_price_at_cutoff" in df.columns:
            print("\nTarget statistics by plan price:")
            print(
                df.groupby("plan_price_at_cutoff", dropna=False)[TARGET]
                .agg(["count", "mean", "median", "min", "max"])
                .round(3)
                .to_string()
            )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE — NO DATA OR MODELS WERE MODIFIED")
    print("=" * 70)
    print(
        "Note: correlations and simple formula checks are diagnostic. "
        "They do not prove leakage by themselves."
    )


if __name__ == "__main__":
    main()
