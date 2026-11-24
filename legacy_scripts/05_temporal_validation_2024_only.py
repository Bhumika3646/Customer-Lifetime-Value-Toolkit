
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF = pd.Timestamp("2023-12-31")
TEST_CUTOFF = pd.Timestamp("2024-06-30")
HORIZON_DAYS = 90
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

TRAIN_HORIZON_END = TRAIN_CUTOFF + pd.Timedelta(days=HORIZON_DAYS)
TEST_HORIZON_END = TEST_CUTOFF + pd.Timedelta(days=HORIZON_DAYS)

FILE_COLUMNS = {
    "customers": [
        "customer_id", "signup_date", "company_size",
        "industry", "acquisition_channel", "country",
    ],
    "subscriptions": [
        "subscription_id", "customer_id", "start_date",
        "end_date", "status", "plan_id",
    ],
    "invoices": [
        "invoice_id", "subscription_id", "invoice_date",
        "due_date", "amount", "status",
    ],
    "payments": [
        "payment_id", "subscription_id", "payment_date",
        "amount", "status", "extra_blank",
    ],
    "plans": [
        "plan_id", "plan_name", "billing_period", "price",
    ],
    "churn_events": [
        "event_id", "customer_id", "churn_date", "reason",
    ],
    "plan_changes": [
        "change_id", "subscription_id", "old_plan_id",
        "new_plan_id", "change_date",
    ],
}

FILE_NAMES = {
    "customers": "customers.csv",
    "subscriptions": "subscriptions.csv",
    "invoices": "invoices.csv",
    "payments": "payments.csv",
    "plans": "subscription_plans.csv",
    "churn_events": "Churn_events.csv",
    "plan_changes": "plan_chnages.csv",
}

ID_COLUMNS = {
    "customers": ["customer_id"],
    "subscriptions": ["subscription_id", "customer_id", "plan_id"],
    "invoices": ["invoice_id", "subscription_id"],
    "payments": ["payment_id", "subscription_id"],
    "plans": ["plan_id"],
    "churn_events": ["event_id", "customer_id"],
    "plan_changes": [
        "change_id", "subscription_id", "old_plan_id", "new_plan_id",
    ],
}

TARGET_REGRESSION = "target_future_90d_revenue"
TARGET_CLASSIFICATION = "target_churn_within_90d"


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

def load_data():
    data = {}

    print("\nLoading raw CSV files...")

    for table_name, filename in FILE_NAMES.items():
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Check that all seven CSV files are inside raw_data."
            )

        df = pd.read_csv(
            path,
            header=None,
            names=FILE_COLUMNS[table_name],
            low_memory=False,
        )

        # Standardize identifiers as strings.
        for column in ID_COLUMNS[table_name]:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

        data[table_name] = df
        print(f"  {filename}: {len(df):,} rows")

    date_columns = {
        "customers": ["signup_date"],
        "subscriptions": ["start_date", "end_date"],
        "invoices": ["invoice_date", "due_date"],
        "payments": ["payment_date"],
        "churn_events": ["churn_date"],
        "plan_changes": ["change_date"],
    }

    for table_name, columns in date_columns.items():
        for column in columns:
            data[table_name][column] = pd.to_datetime(
                data[table_name][column],
                errors="coerce",
            )

    for table_name in ["payments", "invoices"]:
        data[table_name]["amount"] = pd.to_numeric(
            data[table_name]["amount"],
            errors="coerce",
        )

    data["plans"]["price"] = pd.to_numeric(
        data["plans"]["price"],
        errors="coerce",
    )

    for table_name in ["payments", "invoices"]:
        data[table_name]["status"] = (
            data[table_name]["status"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    data["payments"] = data["payments"].drop(
        columns=["extra_blank"],
        errors="ignore",
    )

    # Duplicate IDs could create duplicate customer rows or targets.
    for table_name, id_column in [
        ("customers", "customer_id"),
        ("subscriptions", "subscription_id"),
        ("plans", "plan_id"),
        ("payments", "payment_id"),
        ("churn_events", "event_id"),
        ("plan_changes", "change_id"),
    ]:
        if data[table_name][id_column].duplicated().any():
            raise ValueError(
                f"Duplicate {id_column} values found in "
                f"{FILE_NAMES[table_name]}. Audit these rows first."
            )

    print("CSV loading and type standardization complete.")
    return data


def is_successful_payment(status_series):
    normalized = (
        status_series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized.isin([
        "success", "successful", "succeeded",
        "paid", "completed",
    ])


# ============================================================
# 3. RECONSTRUCT PLAN AS OF THE CUTOFF
# ============================================================

def reconstruct_plans(subscriptions, plan_changes, cutoff):
    subscriptions = subscriptions.copy()
    changes = plan_changes.copy()

    changes["change_date"] = pd.to_datetime(
        changes["change_date"],
        errors="coerce",
    )

    # The subscription's recorded plan is the fallback.
    subscriptions["plan_id_at_cutoff"] = subscriptions["plan_id"]

    changes = changes[
        changes["subscription_id"].notna()
        & changes["change_date"].notna()
    ].copy()

    # Process the change history for each subscription.
    for subscription_id, history in changes.groupby(
        "subscription_id",
        sort=False,
    ):
        history = history.sort_values(
            ["change_date", "change_id"],
            kind="stable",
        )

        past = history[history["change_date"] <= cutoff]

        if not past.empty:
            # The most recent change on/before the cutoff.
            plan_at_cutoff = past.iloc[-1]["new_plan_id"]
        else:
            # If the first change is in the future, the old
            # plan is effective until that future change.
            future = history[history["change_date"] > cutoff]

            if future.empty:
                continue

            plan_at_cutoff = future.iloc[0]["old_plan_id"]

        mask = subscriptions["subscription_id"].eq(subscription_id)

        subscriptions.loc[
            mask, "plan_id_at_cutoff"
        ] = plan_at_cutoff

    return subscriptions


# ============================================================
# 4. BUILD CUSTOMER SNAPSHOT
# ============================================================

def build_snapshot(data, cutoff):
    cutoff = pd.Timestamp(cutoff)
    horizon_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

    print(
        f"\nBuilding snapshot: cutoff={cutoff.date()}, "
        f"target end={horizon_end.date()}"
    )

    customers = data["customers"].copy()
    subscriptions = data["subscriptions"].copy()
    plans = data["plans"].copy()
    payments = data["payments"].copy()
    churn_events = data["churn_events"].copy()

    # Customers must exist by the snapshot date.
    customers = customers[
        customers["signup_date"].notna()
        & (customers["signup_date"] <= cutoff)
    ].copy()

    # Do not include customers already recorded as churned.
    churned_before_cutoff = set(
        churn_events.loc[
            churn_events["churn_date"].notna()
            & (churn_events["churn_date"] <= cutoff),
            "customer_id",
        ].dropna()
    )

    before_filter = len(customers)

    customers = customers[
        ~customers["customer_id"].isin(churned_before_cutoff)
    ].copy()

    print(
        "Customers excluded due to previous recorded churn:",
        f"{before_filter - len(customers):,}",
    )

    # Only subscriptions that had started by the cutoff.
    subscriptions = subscriptions[
        subscriptions["start_date"].notna()
        & (subscriptions["start_date"] <= cutoff)
    ].copy()

    # Reconstruct historical plan before joining plan details.
    subscriptions = reconstruct_plans(
        subscriptions,
        data["plan_changes"],
        cutoff,
    )

    # Keep eligible customers only.
    base = subscriptions.merge(
        customers,
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )

    # This project assumes one subscription per customer.
    # Stop if that assumption is violated rather than silently
    # overweighting customers in the evaluation.
    if base["customer_id"].duplicated().any():
        raise ValueError(
            "More than one eligible subscription exists for a customer. "
            "This script assumes one subscription per customer; "
            "aggregate to customer level before continuing."
        )

    plan_lookup = plans.rename(
        columns={
            "plan_id": "plan_id_lookup",
            "plan_name": "plan_name_at_cutoff",
            "billing_period": "billing_period_at_cutoff",
            "price": "plan_price_at_cutoff",
        }
    )

    base = base.merge(
        plan_lookup,
        left_on="plan_id_at_cutoff",
        right_on="plan_id_lookup",
        how="left",
        validate="many_to_one",
    )

    base["tenure_days"] = (
        cutoff - base["signup_date"]
    ).dt.days.clip(lower=0)

    base["subscription_age_days"] = (
        cutoff - base["start_date"]
    ).dt.days.clip(lower=0)

    # Successful payments only.
    successful = payments[
        is_successful_payment(payments["status"])
        & payments["payment_date"].notna()
    ].copy()

    subscription_customer = base[
        ["subscription_id", "customer_id"]
    ].copy()

    successful = successful.merge(
        subscription_customer,
        on="subscription_id",
        how="inner",
        validate="many_to_one",
    )

    # Historical payments are features.
    historical = successful[
        successful["payment_date"] <= cutoff
    ].copy()

    # Future payments are targets only.
    future_payments = successful[
        (successful["payment_date"] > cutoff)
        & (successful["payment_date"] <= horizon_end)
    ].copy()

    # Historical totals and last payment date.
    if historical.empty:
        historical_totals = pd.DataFrame(
            columns=[
                "customer_id",
                "historical_successful_payment_count",
                "historical_successful_revenue",
            ]
        )
        last_payment = pd.DataFrame(
            columns=[
                "customer_id",
                "last_successful_payment_date",
            ]
        )
    else:
        historical_totals = (
            historical.groupby("customer_id")
            .agg(
                historical_successful_payment_count=(
                    "payment_id", "nunique"
                ),
                historical_successful_revenue=("amount", "sum"),
            )
            .reset_index()
        )

        last_payment = (
            historical.groupby("customer_id")["payment_date"]
            .max()
            .reset_index()
            .rename(
                columns={
                    "payment_date": "last_successful_payment_date"
                }
            )
        )

    base = base.merge(
        historical_totals,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        last_payment,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # Payment count and revenue in recent time windows.
    for days in [30, 60, 90]:
        start_date = cutoff - pd.Timedelta(days=days)

        recent = historical[
            (historical["payment_date"] > start_date)
            & (historical["payment_date"] <= cutoff)
        ]

        if recent.empty:
            recent_agg = pd.DataFrame(
                columns=[
                    "customer_id",
                    f"payment_count_{days}d",
                    f"revenue_{days}d",
                ]
            )
        else:
            recent_agg = (
                recent.groupby("customer_id")
                .agg(
                    **{
                        f"payment_count_{days}d": (
                            "payment_id", "nunique"
                        ),
                        f"revenue_{days}d": ("amount", "sum"),
                    }
                )
                .reset_index()
            )

        base = base.merge(
            recent_agg,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

    base["days_since_last_successful_payment"] = (
        cutoff - base["last_successful_payment_date"]
    ).dt.days

    zero_columns = [
        "historical_successful_payment_count",
        "historical_successful_revenue",
        "payment_count_30d",
        "revenue_30d",
        "payment_count_60d",
        "revenue_60d",
        "payment_count_90d",
        "revenue_90d",
    ]

    for column in zero_columns:
        if column not in base.columns:
            base[column] = 0

        base[column] = base[column].fillna(0)

    # Target 1: revenue in the following 90 days.
    if future_payments.empty:
        future_revenue = pd.DataFrame(
            columns=["customer_id", TARGET_REGRESSION]
        )
    else:
        future_revenue = (
            future_payments.groupby("customer_id")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": TARGET_REGRESSION})
        )

    base = base.merge(
        future_revenue,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    base[TARGET_REGRESSION] = (
        base[TARGET_REGRESSION].fillna(0)
    )

    # Target 2: recorded churn in the following 90 days.
    future_churn = churn_events[
        churn_events["churn_date"].notna()
        & (churn_events["churn_date"] > cutoff)
        & (churn_events["churn_date"] <= horizon_end)
    ]

    churned_ids = set(future_churn["customer_id"].dropna())

    base[TARGET_CLASSIFICATION] = (
        base["customer_id"].isin(churned_ids).astype(int)
    )

    base["snapshot_cutoff_date"] = cutoff
    base["target_horizon_end_date"] = horizon_end

    print(f"Snapshot rows: {len(base):,}")
    print(
        "Churn target positives:",
        int(base[TARGET_CLASSIFICATION].sum()),
    )
    print(
        "Mean future 90-day revenue:",
        round(float(base[TARGET_REGRESSION].mean()), 2),
    )

    return base


# ============================================================
# 5. FEATURES AND PREPROCESSING
# ============================================================

EXCLUDED_FEATURES = {
    # Identifiers
    "customer_id",
    "subscription_id",
    "invoice_id",
    "payment_id",
    "event_id",
    "change_id",

    # Raw dates
    "signup_date",
    "start_date",
    "end_date",
    "invoice_date",
    "due_date",
    "payment_date",
    "churn_date",
    "change_date",
    "last_successful_payment_date",
    "snapshot_cutoff_date",
    "target_horizon_end_date",

    # Status fields may encode information unavailable at prediction time.
    "status",
    "reason",

    # Targets
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,

    # Raw plan identifiers
    "plan_id",
    "plan_id_at_cutoff",
    "plan_id_lookup",
}


def get_feature_columns(df):
    forbidden_terms = [
        "target_",
        "churn_date",
        "churn_reason",
        "end_date",
    ]

    return [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURES
        and not any(term in column.lower() for term in forbidden_terms)
    ]


def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# 6. MODEL CANDIDATES
# ============================================================

def get_regression_models():
    models = {
        "Mean Baseline": None,
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not installed; skipping XGBoost regression.")

    return models


def get_classification_models(y_fit):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
        ),
    }

    try:
        from xgboost import XGBClassifier

        positives = int((y_fit == 1).sum())
        negatives = int((y_fit == 0).sum())
        ratio = negatives / max(positives, 1)

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost not installed; skipping XGBoost classification.")

    return models


def make_pipeline(model, X_fit):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_fit)),
            ("model", clone(model)),
        ]
    )


# ============================================================
# 7. REGRESSION VALIDATION AND FINAL TEST
# ============================================================

def evaluate_regression(
    X_fit,
    y_fit,
    X_val,
    y_val,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print("\n" + "=" * 68)
    print("REGRESSION: FUTURE 90-DAY REVENUE")
    print("=" * 68)

    rows = []
    candidates = get_regression_models()

    # Validation baseline: predict mean revenue in the fit subset.
    baseline_val = np.full(len(y_val), y_fit.mean())
    baseline_val_mae = mean_absolute_error(y_val, baseline_val)

    print(f"Validation mean-baseline MAE: {baseline_val_mae:.4f}")

    # Select the model using validation MAE only.
    for name, model in candidates.items():
        try:
            if model is None:
                continue

            pipeline = make_pipeline(model, X_fit)
            pipeline.fit(X_fit, y_fit)

            val_predictions = np.maximum(
                pipeline.predict(X_val), 0
            )

            mae = mean_absolute_error(y_val, val_predictions)
            rmse = np.sqrt(
                mean_squared_error(y_val, val_predictions)
            )
            r2 = r2_score(y_val, val_predictions)

            rows.append({
                "model": name,
                "validation_MAE": mae,
                "validation_RMSE": rmse,
                "validation_R2": r2,
            })

            print(
                f"{name}: validation MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, R2={r2:.4f}"
            )

        except Exception as exc:
            print(f"{name} validation failed: {exc}")

    validation_results = pd.DataFrame(rows)

    validation_path = (
        OUTPUT_DIR / "2024_only_temporal_regression_validation_results.csv"
    )
    validation_results.to_csv(validation_path, index=False)

    if validation_results.empty:
        print("No regression model completed successfully.")
        return validation_results, pd.DataFrame(), None

    # Lowest validation MAE wins; test data is not used for selection.
    best_name = validation_results.sort_values(
        "validation_MAE"
    ).iloc[0]["model"]

    print(f"\nSelected using validation MAE: {best_name}")

    if best_name == "Mean Baseline":
        final_predictions = np.full(len(y_test), y_train.mean())
    else:
        final_model = candidates[best_name]
        final_pipeline = make_pipeline(final_model, X_train)
        final_pipeline.fit(X_train, y_train)

        final_predictions = np.maximum(
            final_pipeline.predict(X_test), 0
        )

    test_mae = mean_absolute_error(y_test, final_predictions)
    test_rmse = np.sqrt(
        mean_squared_error(y_test, final_predictions)
    )
    test_r2 = r2_score(y_test, final_predictions)

    test_results = pd.DataFrame([{
        "selected_model": best_name,
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
        "test_R2": test_r2,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }])

    print("\nFinal untouched test results:")
    print(
        f"{best_name}: MAE={test_mae:.4f}, "
        f"RMSE={test_rmse:.4f}, R2={test_r2:.4f}"
    )

    test_path = OUTPUT_DIR / "2024_only_temporal_regression_results.csv"
    test_results.to_csv(test_path, index=False)

    return validation_results, test_results, best_name


# ============================================================
# 8. CHURN VALIDATION AND FINAL TEST
# ============================================================

def evaluate_classification(
    X_fit,
    y_fit,
    X_val,
    y_val,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print("\n" + "=" * 68)
    print("CLASSIFICATION: CHURN WITHIN 90 DAYS")
    print("=" * 68)

    print("Fit class counts:", y_fit.value_counts().to_dict())
    print("Validation class counts:", y_val.value_counts().to_dict())
    print("Training class counts:", y_train.value_counts().to_dict())
    print("Test class counts:", y_test.value_counts().to_dict())

    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        print(
            "Cannot validate churn models: fit and validation "
            "subsets must each contain both classes."
        )
        return pd.DataFrame(), None

    rows = []
    candidates = get_classification_models(y_fit)
    threshold_grid = np.arange(0.05, 1.00, 0.05)

    # Select model and threshold on validation data.
    for name, model in candidates.items():
        try:
            pipeline = make_pipeline(model, X_fit)
            pipeline.fit(X_fit, y_fit)

            val_probabilities = pipeline.predict_proba(X_val)[:, 1]

            best_threshold = 0.50
            best_f1 = -1.0
            best_precision = 0.0
            best_recall = 0.0

            for threshold in threshold_grid:
                val_predictions = (
                    val_probabilities >= threshold
                ).astype(int)

                val_f1 = f1_score(
                    y_val,
                    val_predictions,
                    zero_division=0,
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_threshold = float(threshold)
                    best_precision = precision_score(
                        y_val,
                        val_predictions,
                        zero_division=0,
                    )
                    best_recall = recall_score(
                        y_val,
                        val_predictions,
                        zero_division=0,
                    )

            rows.append({
                "model": name,
                "threshold": best_threshold,
                "validation_F1": best_f1,
                "validation_precision": best_precision,
                "validation_recall": best_recall,
            })

            print(
                f"{name}: threshold={best_threshold:.2f}, "
                f"validation F1={best_f1:.4f}, "
                f"precision={best_precision:.4f}, "
                f"recall={best_recall:.4f}"
            )

        except Exception as exc:
            print(f"{name} validation failed: {exc}")

    validation_results = pd.DataFrame(rows)

    validation_path = OUTPUT_DIR / "2024_only_temporal_churn_validation_results.csv"
    validation_results.to_csv(validation_path, index=False)

    if validation_results.empty:
        print("No churn model completed validation successfully.")
        return validation_results, None

    # Pick the model based on validation F1, never test F1.
    best_row = validation_results.sort_values(
        ["validation_F1", "validation_recall"],
        ascending=[False, False],
    ).iloc[0]

    best_name = best_row["model"]
    best_threshold = float(best_row["threshold"])

    print(
        f"\nSelected using validation F1: {best_name}; "
        f"threshold={best_threshold:.2f}"
    )

    # Refit the selected model on all training data.
    final_pipeline = make_pipeline(
        candidates[best_name],
        X_train,
    )
    final_pipeline.fit(X_train, y_train)

    # Evaluate once on the untouched test snapshot.
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )

    auc = (
        roc_auc_score(y_test, test_probabilities)
        if y_test.nunique() == 2
        else np.nan
    )

    test_results = pd.DataFrame([{
        "selected_model": best_name,
        "threshold": best_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }])

    print("\nFinal untouched test results:")
    print(
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}, AUC={auc:.4f}"
    )

    test_path = OUTPUT_DIR / "2024_only_temporal_churn_results.csv"
    test_results.to_csv(test_path, index=False)

    return validation_results, test_results


# ============================================================
# 9. MAIN WORKFLOW
# ============================================================

def main():
    print("=" * 68)
    print("CUSTOMER LIFETIME VALUE - TEMPORAL VALIDATION")
    print("=" * 68)
    print("Training cutoff:", TRAIN_CUTOFF.date())
    print("Training target window ends:", TRAIN_HORIZON_END.date())
    print("Testing cutoff:", TEST_CUTOFF.date())
    print("Testing target window ends:", TEST_HORIZON_END.date())

    data = load_data()

    train = build_snapshot(data, TRAIN_CUTOFF)
    test = build_snapshot(data, TEST_CUTOFF)

    train_path = OUTPUT_DIR / "2024_only_temporal_train_snapshot_2023-12-31.csv"
    test_path = OUTPUT_DIR / "2024_only_temporal_test_snapshot_2024-06-30.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("\nSaved snapshots:")
    print(" ", train_path)
    print(" ", test_path)

    train_ids = set(train["customer_id"].dropna())
    test_ids = set(test["customer_id"].dropna())
    overlap = len(train_ids.intersection(test_ids))

    print("\nCustomer overlap between snapshots:", overlap)
    print(
        "The snapshots can contain the same customer at different dates. "
        "This is temporal evaluation, not customer-independent evaluation."
    )

    # Only use features available in both snapshots.
    feature_columns = [
        column
        for column in get_feature_columns(train)
        if column in test.columns
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    # Drop any remaining datetime columns.
    datetime_columns = [
        column
        for column in X_train.columns
        if (
            pd.api.types.is_datetime64_any_dtype(X_train[column])
            or pd.api.types.is_datetime64_any_dtype(X_test[column])
        )
    ]

    X_train = X_train.drop(columns=datetime_columns)
    X_test = X_test.drop(columns=datetime_columns)

    # Ensure both feature matrices have the same columns/order.
    X_test = X_test.reindex(columns=X_train.columns)

    print("\nNumber of features:", len(X_train.columns))
    print("Features:")
    for column in X_train.columns:
        print(" -", column)

    # Regression targets.
    y_train_reg = pd.to_numeric(
        train[TARGET_REGRESSION],
        errors="coerce",
    ).fillna(0)

    y_test_reg = pd.to_numeric(
        test[TARGET_REGRESSION],
        errors="coerce",
    ).fillna(0)

    # Churn targets.
    y_train_clf = train[TARGET_CLASSIFICATION].astype(int)
    y_test_clf = test[TARGET_CLASSIFICATION].astype(int)

    # Validation split comes ONLY from the training snapshot.
    # This is a random holdout because a single snapshot does not
    # contain separate feature-observation dates for its customers.
    indices = np.arange(len(X_train))

    if len(indices) < 10:
        raise ValueError(
            "Too few training rows for reliable model validation."
        )

    fit_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    X_fit = X_train.iloc[fit_indices].copy()
    X_val = X_train.iloc[val_indices].copy()

    y_fit_reg = y_train_reg.iloc[fit_indices].copy()
    y_val_reg = y_train_reg.iloc[val_indices].copy()

    y_fit_clf = y_train_clf.iloc[fit_indices].copy()
    y_val_clf = y_train_clf.iloc[val_indices].copy()

    print("\nValidation split:")
    print("Fit rows:", len(X_fit))
    print("Validation rows:", len(X_val))
    print("Final test rows:", len(X_test))

    # Regression: choose model on validation, test once.
    reg_validation, reg_test, selected_regression_model = (
        evaluate_regression(
            X_fit,
            y_fit_reg,
            X_val,
            y_val_reg,
            X_train,
            y_train_reg,
            X_test,
            y_test_reg,
        )
    )

    # Classification: choose model and threshold on validation.
    clf_validation, clf_test = evaluate_classification(
        X_fit,
        y_fit_clf,
        X_val,
        y_val_clf,
        X_train,
        y_train_clf,
        X_test,
        y_test_clf,
    )

    # Save a readable summary.
    summary_path = OUTPUT_DIR / "2024_only_temporal_validation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("TEMPORAL VALIDATION SUMMARY\n")
        file.write("=" * 65 + "\n")
        file.write(f"Training cutoff: {TRAIN_CUTOFF.date()}\n")
        file.write(
            f"Training target window ends: {TRAIN_HORIZON_END.date()}\n"
        )
        file.write(f"Test cutoff: {TEST_CUTOFF.date()}\n")
        file.write(
            f"Test target window ends: {TEST_HORIZON_END.date()}\n"
        )
        file.write(f"Training snapshot rows: {len(train)}\n")
        file.write(f"Test snapshot rows: {len(test)}\n")
        file.write(f"Shared customer IDs: {overlap}\n")
        file.write(f"Number of features: {len(X_train.columns)}\n")
        file.write(f"Fit rows: {len(X_fit)}\n")
        file.write(f"Validation rows: {len(X_val)}\n\n")

        file.write(
            "Revenue target: total successful-payment revenue in the "
            "90 days after the cutoff. This is a short-term revenue proxy, "
            "not true lifetime value or profit.\n\n"
        )
        file.write(
            "Churn target: a recorded churn event after the cutoff and "
            "on or before the 90-day horizon.\n\n"
        )
        file.write(
            "Customers with a recorded churn event on/before the cutoff "
            "are excluded from that snapshot.\n\n"
        )
        file.write(
            "Validation method: random holdout from the training snapshot. "
            "The final 2024 snapshot is not used for model selection. "
            "Customers may appear in both snapshots, so this is temporal "
            "evaluation rather than customer-independent evaluation.\n\n"
        )
        file.write(
            "Dataset limitation: annual-plan invoice/payment records "
            "may be absent in 2025. This can distort future revenue "
            "targets and limits the interpretation of model performance. "
            "The dataset appears synthetic and should not be treated as "
            "evidence of real-world generalization.\n\n"
        )

        file.write("REGRESSION VALIDATION RESULTS\n")
        file.write("-" * 65 + "\n")
        if reg_validation.empty:
            file.write("No regression model validated successfully.\n")
        else:
            file.write(reg_validation.to_string(index=False) + "\n")

        file.write("\n\nREGRESSION FINAL TEST RESULTS\n")
        file.write("-" * 65 + "\n")
        if reg_test.empty:
            file.write("No final regression result available.\n")
        else:
            file.write(reg_test.to_string(index=False) + "\n")

        file.write("\n\nCHURN VALIDATION RESULTS\n")
        file.write("-" * 65 + "\n")
        if clf_validation.empty:
            file.write("No churn model validated successfully.\n")
        else:
            file.write(clf_validation.to_string(index=False) + "\n")

        file.write("\n\nCHURN FINAL TEST RESULTS\n")
        file.write("-" * 65 + "\n")
        if clf_test is None or clf_test.empty:
            file.write("No final churn result available.\n")
        else:
            file.write(clf_test.to_string(index=False) + "\n")

    print("\n" + "=" * 68)
    print("TEMPORAL VALIDATION FINISHED")
    print("=" * 68)
    print("Training snapshot:", train_path)
    print("Test snapshot:", test_path)
    print("Regression validation:", OUTPUT_DIR / "2024_only_temporal_regression_validation_results.csv")
    print("Regression test:", OUTPUT_DIR / "2024_only_temporal_regression_results.csv")
    print("Churn validation:", OUTPUT_DIR / "2024_only_temporal_churn_validation_results.csv")
    print("Churn test:", OUTPUT_DIR / "2024_only_temporal_churn_results.csv")
    print("Summary:", summary_path)
    print("\nReview the results and data limitations before choosing a final model.")


if __name__ == "__main__":
    main()
