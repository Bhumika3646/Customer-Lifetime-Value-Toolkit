
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "plan_wise_predictions.csv"
TEST_PATH = OUTPUT_DIR / "temporal_test_snapshot_2025-06-30.csv"
OUTPUT_PATH = OUTPUT_DIR / "revenue_range_error_analysis.csv"


def main():
    if not PREDICTIONS_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Required files are missing. Run scripts 24 and 05 first."
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    test = pd.read_csv(TEST_PATH)

    required_predictions = [
        "customer_id",
        "plan_name",
        "actual_future_90d_revenue",
        "full_xgboost_prediction",
        "three_month_plan_price_prediction",
    ]
    required_test = ["customer_id", "plan_price_at_cutoff"]

    for col in required_predictions:
        if col not in predictions.columns:
            raise ValueError(f"Missing prediction column: {col}")

    for col in required_test:
        if col not in test.columns:
            raise ValueError(f"Missing test column: {col}")

    # Bring the plan price into the saved predictions.
    prices = test[["customer_id", "plan_price_at_cutoff"]].drop_duplicates(
        subset=["customer_id"]
    )

    df = predictions.merge(
        prices,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    for col in [
        "actual_future_90d_revenue",
        "full_xgboost_prediction",
        "three_month_plan_price_prediction",
        "plan_price_at_cutoff",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "actual_future_90d_revenue",
            "full_xgboost_prediction",
            "three_month_plan_price_prediction",
            "plan_price_at_cutoff",
        ]
    ).copy()

    # Analyze only Pro and Premium, as these were the weaker groups.
    df = df[df["plan_name"].isin(["Pro", "Premium"])].copy()

    # Revenue bands are defined relative to each customer's plan price.
    # This makes comparisons more meaningful across plans.
    def revenue_band(row):
        actual = row["actual_future_90d_revenue"]
        price = row["plan_price_at_cutoff"]

        if actual == 0:
            return "Zero revenue"
        if actual <= price:
            return "Positive: up to 1x plan price"
        if actual <= 2 * price:
            return "Positive: over 1x to 2x"
        if actual <= 3 * price:
            return "Positive: over 2x to 3x"
        return "Positive: over 3x plan price"

    df["actual_revenue_band"] = df.apply(revenue_band, axis=1)

    model_columns = [
        ("Full XGBoost", "full_xgboost_prediction"),
        (
            "Three-month plan-price baseline",
            "three_month_plan_price_prediction",
        ),
    ]

    rows = []

    for (plan, band), group in df.groupby(
        ["plan_name", "actual_revenue_band"],
        sort=True,
        dropna=False,
    ):
        actual = group["actual_future_90d_revenue"].to_numpy(dtype=float)

        for model_name, prediction_col in model_columns:
            predicted = group[prediction_col].to_numpy(dtype=float)

            rows.append({
                "plan_name": plan,
                "actual_revenue_band": band,
                "model": model_name,
                "customers": len(group),
                "actual_mean": actual.mean(),
                "predicted_mean": predicted.mean(),
                "prediction_bias": (predicted - actual).mean(),
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("PRO AND PREMIUM: ERROR BY ACTUAL REVENUE RANGE")
    print("=" * 100)

    if result.empty:
        print("No Pro or Premium rows were found.")
    else:
        print(
            result.to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

    print("\nBias interpretation:")
    print("  Positive = predictions are higher than actual revenue on average.")
    print("  Negative = predictions are lower than actual revenue on average.")
    print("  MAE and RMSE are error measures; lower values are better.")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
