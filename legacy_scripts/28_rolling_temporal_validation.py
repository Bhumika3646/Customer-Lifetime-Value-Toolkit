
from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_DAYS = 90
RANDOM_STATE = 42

# Each date is the date on which the prediction is made.
# Its future revenue target covers the following 90 days.
CUTOFFS = [
    "2023-06-30",
    "2023-09-30",
    "2023-12-31",
    "2024-03-31",
    "2024-06-30",
    "2024-09-30",
    "2024-12-31",
    "2025-03-31",
    "2025-06-30",
]

# Reuse the existing project data-processing functions.
ORIGINAL_SCRIPT = BASE_DIR / "05_temporal_validation.py"

if not ORIGINAL_SCRIPT.exists():
    raise FileNotFoundError(
        f"Cannot find {ORIGINAL_SCRIPT.name} in {BASE_DIR}"
    )

spec = importlib.util.spec_from_file_location(
    "original_temporal_validation",
    ORIGINAL_SCRIPT,
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

TARGET = original.TARGET_REGRESSION


# ============================================================
# 2. MODEL AND FEATURE HELPERS
# ============================================================

def prepare_features(df, feature_columns):
    X = df[feature_columns].copy()

    # Convert date-like features to numeric values only if the
    # original feature rules have not already excluded them.
    datetime_cols = [
        c for c in X.columns
        if pd.api.types.is_datetime64_any_dtype(X[c])
    ]
    X = X.drop(columns=datetime_cols, errors="ignore")

    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype(object)

    return X


def make_xgboost_pipeline(X):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. In your project terminal run: "
            "python -m pip install xgboost"
        ) from exc

    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        c for c in X.columns if c not in numeric
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def calculate_metrics(actual, predicted):
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 0
    )
    actual = np.asarray(actual, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) > 1 and np.var(actual) > 0
            else np.nan
        ),
    }


def is_monthly(df):
    if "billing_period_at_cutoff" not in df.columns:
        raise ValueError(
            "billing_period_at_cutoff is missing from the snapshot."
        )

    return (
        df["billing_period_at_cutoff"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.contains("month", na=False)
    )


def make_plan_baseline(df):
    price = pd.to_numeric(
        df["plan_price_at_cutoff"], errors="coerce"
    ).fillna(0)

    # Baseline: three months of the monthly plan price.
    return np.maximum(price.to_numpy(dtype=float) * 3, 0)


def make_hybrid_predictions(df, xgb_predictions):
    plans = (
        df["plan_name_at_cutoff"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
        .to_numpy()
    )

    baseline = make_plan_baseline(df)
    hybrid = np.asarray(xgb_predictions, dtype=float).copy()

    for i, plan in enumerate(plans):
        if plan == "free":
            hybrid[i] = 0.0
        elif plan in {"pro", "premium"}:
            hybrid[i] = baseline[i]
        # Basic and other plans retain the XGBoost prediction.

    return np.maximum(hybrid, 0)


# ============================================================
# 3. BUILD HISTORICAL SNAPSHOTS
# ============================================================

def main():
    print("=" * 72)
    print("ROLLING TEMPORAL VALIDATION")
    print("=" * 72)

    data = original.load_data()

    # Do not create targets extending beyond the available payment data.
    valid_payment_dates = data["payments"]["payment_date"].dropna()

    if valid_payment_dates.empty:
        raise ValueError("No valid payment dates found.")

    latest_payment_date = valid_payment_dates.max().normalize()

    print("Latest payment date in raw data:", latest_payment_date.date())

    snapshots = {}

    for date_text in CUTOFFS:
        cutoff = pd.Timestamp(date_text)
        horizon_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

        if horizon_end > latest_payment_date:
            print(
                f"Skipping {cutoff.date()}: its 90-day target "
                f"ends after the latest payment date."
            )
            continue

        snapshot = original.build_snapshot(data, cutoff)

        # Keep monthly-plan observations, matching the earlier experiment.
        snapshot = snapshot.loc[is_monthly(snapshot)].copy()

        snapshot[TARGET] = pd.to_numeric(
            snapshot[TARGET], errors="coerce"
        ).fillna(0)

        if not snapshot.empty:
            snapshots[cutoff] = snapshot
            print(
                f"Stored monthly snapshot {cutoff.date()}: "
                f"{len(snapshot):,} rows"
            )

    if len(snapshots) < 3:
        raise ValueError(
            "Fewer than three eligible snapshots were created. "
            "Check the date coverage and billing_period values."
        )

    # Use only features available under the original feature rules.
    first_snapshot = next(iter(snapshots.values()))
    feature_columns = original.get_feature_columns(first_snapshot)

    # Retain columns present in every snapshot.
    common_columns = set(feature_columns)
    for snapshot in snapshots.values():
        common_columns &= set(snapshot.columns)

    feature_columns = [
        c for c in feature_columns if c in common_columns
    ]

    if not feature_columns:
        raise ValueError("No common eligible features were found.")

    print("\nNumber of features:", len(feature_columns))

    # ========================================================
    # 4. EXPANDING-WINDOW ROLLING EVALUATION
    # ========================================================

    result_rows = []
    prediction_rows = []

    ordered_dates = sorted(snapshots.keys())

    for test_cutoff in ordered_dates:
        # Every training target must have fully elapsed by the
        # test prediction date. This avoids future-target leakage.
        eligible_train_dates = [
            d for d in ordered_dates
            if d < test_cutoff
            and d + pd.Timedelta(days=HORIZON_DAYS) <= test_cutoff
        ]

        if not eligible_train_dates:
            print(
                f"\nSkipping test cutoff {test_cutoff.date()}: "
                "no earlier snapshot has a fully elapsed target window."
            )
            continue

        train_df = pd.concat(
            [snapshots[d] for d in eligible_train_dates],
            ignore_index=True,
        )

        test_df = snapshots[test_cutoff].copy()

        X_train = prepare_features(train_df, feature_columns)
        X_test = prepare_features(test_df, feature_columns)
        X_test = X_test.reindex(columns=X_train.columns)

        y_train = pd.to_numeric(
            train_df[TARGET], errors="coerce"
        ).fillna(0).to_numpy(dtype=float)

        y_test = pd.to_numeric(
            test_df[TARGET], errors="coerce"
        ).fillna(0).to_numpy(dtype=float)

        if len(X_train) == 0 or len(X_test) == 0:
            print("Skipping empty train/test fold.")
            continue

        if np.unique(y_train).size < 2:
            print(
                f"Skipping {test_cutoff.date()}: "
                "training target has insufficient variation."
            )
            continue

        print("\n" + "-" * 72)
        print("Test cutoff:", test_cutoff.date())
        print("Training snapshots:", [
            d.date().isoformat() for d in eligible_train_dates
        ])
        print("Training rows:", f"{len(X_train):,}")
        print("Test rows:", f"{len(X_test):,}")

        pipeline = make_xgboost_pipeline(X_train)
        pipeline.fit(X_train, y_train)

        xgb_predictions = np.maximum(
            pipeline.predict(X_test), 0
        )
        baseline_predictions = make_plan_baseline(test_df)
        hybrid_predictions = make_hybrid_predictions(
            test_df, xgb_predictions
        )

        predictions_by_model = {
            "XGBoost": xgb_predictions,
            "Three-month plan-price baseline": baseline_predictions,
            "Hybrid": hybrid_predictions,
        }

        for model_name, predictions in predictions_by_model.items():
            metrics = calculate_metrics(y_test, predictions)

            row = {
                "test_cutoff": test_cutoff.date().isoformat(),
                "target_horizon_end": (
                    test_cutoff + pd.Timedelta(days=HORIZON_DAYS)
                ).date().isoformat(),
                "training_cutoffs": ";".join(
                    d.date().isoformat() for d in eligible_train_dates
                ),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "model": model_name,
                **metrics,
            }
            result_rows.append(row)

            print(
                f"{model_name}: "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}"
            )

            for i, actual in enumerate(y_test):
                prediction_rows.append({
                    "test_cutoff": test_cutoff.date().isoformat(),
                    "customer_id": test_df.iloc[i]["customer_id"],
                    "plan_name": test_df.iloc[i][
                        "plan_name_at_cutoff"
                    ],
                    "actual_future_90d_revenue": actual,
                    "model": model_name,
                    "predicted_future_90d_revenue": predictions[i],
                })

    # ========================================================
    # 5. SAVE RESULTS
    # ========================================================

    if not result_rows:
        raise ValueError(
            "No folds were evaluated. Check cutoff dates and data coverage."
        )

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)

    results_path = OUTPUT_DIR / "rolling_temporal_validation_results.csv"
    predictions_path = OUTPUT_DIR / "rolling_temporal_predictions.csv"

    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\n" + "=" * 72)
    print("ROLLING VALIDATION SUMMARY")
    print("=" * 72)

    print(
        results.groupby("model")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print("\nSaved results:")
    print(results_path)
    print(predictions_path)

    print(
        "\nImportant: this is rolling temporal evaluation on a synthetic "
        "dataset. Repeated customers can appear across different folds. "
        "The reported metrics do not establish real-world performance."
    )


if __name__ == "__main__":
    main()
