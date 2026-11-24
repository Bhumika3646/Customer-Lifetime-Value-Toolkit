
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "outputs" / "plan_wise_predictions.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "prediction_bias_analysis.csv"


def calculate_metrics(group, prediction_column):
    actual = group["actual_future_90d_revenue"].astype(float)
    predicted = group[prediction_column].astype(float)

    # Bias > 0 means overprediction; bias < 0 means underprediction.
    bias = (predicted - actual).mean()

    return {
        "customers": len(group),
        "actual_mean": actual.mean(),
        "predicted_mean": predicted.mean(),
        "prediction_bias": bias,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2 and actual.nunique() >= 2
            else np.nan
        ),
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run "
            "24_plan_wise_error_analysis.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "plan_name",
        "actual_future_90d_revenue",
        "full_xgboost_prediction",
        "three_month_plan_price_prediction",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for plan_name, group in df.groupby("plan_name", dropna=False):
        for model_name, prediction_column in [
            ("Full XGBoost", "full_xgboost_prediction"),
            ("Three-month plan-price baseline",
             "three_month_plan_price_prediction"),
        ]:
            metrics = calculate_metrics(group, prediction_column)
            rows.append({
                "plan_name": plan_name,
                "model": model_name,
                **metrics,
            })

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 95)
    print("PREDICTION BIAS ANALYSIS BY PLAN")
    print("=" * 95)

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nHow to read prediction_bias:")
    print("  Positive value = average overprediction")
    print("  Negative value = average underprediction")
    print("  Near zero = little average bias (individual errors may still be large)")

    print("\nSaved results:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
