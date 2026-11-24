
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

TRAIN_FILE = OUT / "temporal_train_snapshot_2024-06-30.csv"
TEST_FILE = OUT / "temporal_test_snapshot_2025-06-30.csv"

TARGET = "target_future_90d_revenue"

IMPORTANCE_FILE = OUT / "xgboost_feature_importance.csv"
AUDIT_FILE = OUT / "xgboost_feature_availability_audit.csv"


# ============================================================
# 1. LOAD MONTHLY-PLAN DATA
# ============================================================

def load_monthly(path, label):
    df = pd.read_csv(path, low_memory=False)

    period = (
        df["billing_period_at_cutoff"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df = df.loc[period.eq("monthly")].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["plan_price_at_cutoff"] = pd.to_numeric(
        df["plan_price_at_cutoff"], errors="coerce"
    )

    df = df.dropna(subset=[TARGET, "plan_price_at_cutoff"])
    df = df[df[TARGET] >= 0].copy()

    print(f"{label}: {len(df):,} monthly customers")
    return df


# ============================================================
# 2. SELECT FEATURES
# ============================================================

def select_features(train, test):
    excluded_exact = {
        # Targets
        TARGET,
        "target_churn_within_90d",

        # Identifiers
        "customer_id",
        "subscription_id",
        "invoice_id",
        "payment_id",
        "event_id",
        "change_id",

        # Snapshot metadata
        "snapshot_date",
        "cutoff_date",
        "snapshot_cutoff_date",
        "target_horizon_end_date",

        # Raw plan identifiers: exclude these from the model
        "plan_id",
        "plan_id_at_cutoff",
        "plan_id_lookup",

        # Raw dates and final subscription outcomes
        "signup_date",
        "start_date",
        "end_date",
        "invoice_date",
        "due_date",
        "payment_date",
        "churn_date",
        "change_date",
        "last_successful_payment_date",
        "status",
        "subscription_status",
        "reason",
        "churn_reason",
    }

    features = []

    for col in train.columns:
        name = col.lower()

        if col in excluded_exact:
            continue

        if name.startswith("target_"):
            continue

        if name.endswith("_date") or name.endswith("_datetime"):
            continue

        if col not in test.columns:
            continue

        features.append(col)

    return features


# ============================================================
# 3. MAIN AUDIT
# ============================================================

def main():
    train = load_monthly(TRAIN_FILE, "Training")
    test = load_monthly(TEST_FILE, "Test")

    feature_columns = select_features(train, test)

    if not feature_columns:
        raise ValueError("No usable features were found.")

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    # Identify numerical and categorical features.
    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in feature_columns
        if col not in numeric_columns
    ]

    # Convert numeric values and handle infinities.
    for col in numeric_columns:
        X_train[col] = pd.to_numeric(
            X_train[col], errors="coerce"
        )
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )

        X_train[col] = X_train[col].replace(
            [np.inf, -np.inf], np.nan
        )
        X_test[col] = X_test[col].replace(
            [np.inf, -np.inf], np.nan
        )

    transformers = []

    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_columns,
        ))

    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # XGBoost model.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = np.clip(model.predict(X_test), 0, None)

    y_true = y_test.to_numpy()

    mae = float(np.mean(np.abs(y_true - predictions)))
    rmse = float(np.sqrt(np.mean((y_true - predictions) ** 2)))

    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_residual = float(np.sum((y_true - predictions) ** 2))

    r2 = (
        1 - ss_residual / ss_total
        if ss_total > 0
        else float("nan")
    )

    # ========================================================
    # 4. FEATURE IMPORTANCE
    # ========================================================

    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_xgb = model.named_steps["model"]

    encoded_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_xgb.feature_importances_

    importance_df = pd.DataFrame({
        "encoded_feature": encoded_names,
        "importance": importances,
    })

    # Map encoded features back to their original columns.
    def original_column(encoded_name):
        clean_name = encoded_name

        if clean_name.startswith("numeric__"):
            clean_name = clean_name[len("numeric__"):]
        elif clean_name.startswith("categorical__"):
            clean_name = clean_name[len("categorical__"):]

        matches = [
            col for col in feature_columns
            if clean_name == col or clean_name.startswith(col + "_")
        ]

        return max(matches, key=len) if matches else clean_name

    importance_df["original_feature"] = (
        importance_df["encoded_feature"].map(original_column)
    )

    grouped_importance = (
        importance_df.groupby("original_feature", as_index=False)
        ["importance"].sum()
        .sort_values("importance", ascending=False)
    )

    total_importance = grouped_importance["importance"].sum()

    if total_importance > 0:
        grouped_importance["importance_pct"] = (
            grouped_importance["importance"] / total_importance * 100
        )
    else:
        grouped_importance["importance_pct"] = 0.0

    grouped_importance.to_csv(IMPORTANCE_FILE, index=False)

    # ========================================================
    # 5. FEATURE AVAILABILITY REVIEW
    # ========================================================

    # This is a name/non-null audit only. It cannot prove that a
    # feature was available at the prediction cutoff.
    audit_rows = []

    for col in feature_columns:
        name = col.lower()

        if any(word in name for word in [
            "future", "target", "after_cutoff", "post_cutoff"
        ]):
            flag = "REVIEW: possible future/target information"

        elif name in {
            "revenue_30d",
            "revenue_60d",
            "revenue_90d",
            "payment_count_30d",
            "payment_count_60d",
            "payment_count_90d",
            "historical_successful_revenue",
            "historical_successful_payment_count",
            "plan_price_at_cutoff",
            "billing_period_at_cutoff",
            "tenure_days",
            "subscription_age_days",
        }:
            flag = "CHECK feature construction and cutoff"

        else:
            flag = "CHECK source and cutoff availability"

        audit_rows.append({
            "feature": col,
            "data_type": str(train[col].dtype),
            "training_non_null_pct": round(
                train[col].notna().mean() * 100, 2
            ),
            "test_non_null_pct": round(
                test[col].notna().mean() * 100, 2
            ),
            "availability_review": flag,
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_FILE, index=False)

    # ========================================================
    # 6. PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("CORRECTED XGBOOST FEATURE AUDIT")
    print("=" * 70)

    print(f"Features before encoding: {len(feature_columns)}")
    print(f"Encoded features: {len(encoded_names)}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")

    print("\nTOP 15 ORIGINAL FEATURES")
    print(
        grouped_importance.head(15)
        .round(4)
        .to_string(index=False)
    )

    print("\nFEATURES USED BY THE MODEL")
    for col in feature_columns:
        print(f"- {col}")

    print("\nAvailability review saved to:")
    print(AUDIT_FILE)

    print("\nFeature importance saved to:")
    print(IMPORTANCE_FILE)

    print(
        "\nIMPORTANT: Raw plan identifiers, subscription status, and "
        "churn-reason fields are excluded to avoid outcome leakage. "
        "Feature importance does not prove causation or establish "
        "that every remaining feature is available at prediction time."
    )


if __name__ == "__main__":
    main()
