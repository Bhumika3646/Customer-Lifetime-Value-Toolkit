from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor, DummyClassifier
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
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "customer_snapshot_2025-06-30.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

REVENUE_TARGET = "target_future_90d_revenue"
CHURN_TARGET = "target_churn_within_90d"

# --------------------------------------------------
# 2. LOAD AND CHECK THE SNAPSHOT
# --------------------------------------------------

print("=" * 68)
print("CUSTOMER LIFETIME VALUE AND CHURN MODEL TRAINING")
print("=" * 68)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Snapshot not found: {INPUT_FILE}\n"
        "Run 02_build_customer_snapshot.py first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    REVENUE_TARGET,
    CHURN_TARGET,
    "customer_id",
    "subscription_id",
]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Required columns missing: {missing_columns}")

if df[required_columns].isna().any().any():
    raise ValueError("A required ID or target column contains missing values.")

if not set(df[CHURN_TARGET].unique()).issubset({0, 1}):
    raise ValueError("Churn target must contain only 0 and 1.")

if (df[REVENUE_TARGET] < 0).any():
    raise ValueError("Revenue target contains negative values.")

print(f"\nDataset shape: {df.shape}")
print(f"Customers: {len(df):,}")
print(f"Churn cases: {int(df[CHURN_TARGET].sum()):,}")
print(f"Mean future 90-day revenue: {df[REVENUE_TARGET].mean():.2f}")

# --------------------------------------------------
# 3. FEATURE ENGINEERING AND LEAKAGE PREVENTION
# --------------------------------------------------

for date_col in ["signup_date", "start_date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

cutoff = pd.Timestamp("2025-06-30")

if "signup_date" in df.columns:
    df["customer_tenure_days"] = (cutoff - df["signup_date"]).dt.days

if "start_date" in df.columns:
    df["subscription_tenure_days"] = (cutoff - df["start_date"]).dt.days

# IDs, raw dates, and target columns are excluded from the model inputs.
excluded_columns = [
    "customer_id",
    "subscription_id",
    "snapshot_cutoff_date",
    "signup_date",
    "start_date",
    "end_date",
    "churn_date",
    "target_horizon_end_date",
    # Final subscription status/reason can encode outcomes unavailable
    # at the prediction cutoff; always exclude them defensively.
    "status",
    "subscription_status",
    "reason",
    "churn_reason",
    REVENUE_TARGET,
    CHURN_TARGET,
]

X = df.drop(columns=[c for c in excluded_columns if c in df.columns]).copy()
y_revenue = df[REVENUE_TARGET].astype(float).copy()
y_churn = df[CHURN_TARGET].astype(int).copy()

# The audit showed paired missingness in recent payment count/revenue
# windows. Treat a paired missing window as zero recorded successful
# payments/revenue. Other missing values are handled by the imputers.
for count_col, revenue_col in [
    ("successful_payments_30d", "revenue_30d"),
    ("successful_payments_60d", "revenue_60d"),
    ("successful_payments_90d", "revenue_90d"),
]:
    if count_col in X.columns and revenue_col in X.columns:
        both_missing = X[count_col].isna() & X[revenue_col].isna()
        X.loc[both_missing, [count_col, revenue_col]] = 0

# Drop any unexpected raw datetime columns.
for col in list(X.columns):
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X = X.drop(columns=[col])

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"\nFeature count: {X.shape[1]}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

if X.shape[1] == 0:
    raise ValueError("No features available for training.")

# --------------------------------------------------
# 4. TRAIN/TEST SPLIT
# --------------------------------------------------
# This is a random held-out split from one fixed snapshot, not a
# temporal validation. Further date-based validation is recommended.

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_churn,
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_rev_train = y_revenue.iloc[train_idx]
y_rev_test = y_revenue.iloc[test_idx]
y_churn_train = y_churn.iloc[train_idx]
y_churn_test = y_churn.iloc[test_idx]

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# --------------------------------------------------
# 6. REGRESSION: FUTURE 90-DAY SUCCESSFUL-PAYMENT REVENUE
# --------------------------------------------------

regression_models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    ),
}

regression_results = []
regression_predictions = {}
regression_pipelines = {}

print("\n" + "=" * 68)
print("REGRESSION: FUTURE 90-DAY REVENUE")
print("=" * 68)

for name, model in regression_models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_rev_train)
    predictions = np.maximum(pipeline.predict(X_test), 0)

    mae = mean_absolute_error(y_rev_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_rev_test, predictions))
    r2 = r2_score(y_rev_test, predictions)

    regression_results.append(
        {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}
    )
    regression_predictions[name] = predictions
    regression_pipelines[name] = pipeline

    print(
        f"{name:20s} | MAE: {mae:.4f} | "
        f"RMSE: {rmse:.4f} | R2: {r2:.4f}"
    )

regression_df = pd.DataFrame(regression_results).sort_values("MAE")
regression_df.to_csv(OUTPUT_DIR / "regression_model_results.csv", index=False)

best_regression_name = regression_df.iloc[0]["model"]
pd.DataFrame(
    {
        "actual_future_90d_revenue": y_rev_test.to_numpy(),
        "predicted_future_90d_revenue": regression_predictions[best_regression_name],
    }
).to_csv(OUTPUT_DIR / "best_regression_test_predictions.csv", index=False)

# --------------------------------------------------
# 7. CLASSIFICATION: CHURN WITHIN 90 DAYS
# --------------------------------------------------

# scale_pos_weight can help XGBoost account for the minority churn class.
negative_count = int((y_churn_train == 0).sum())
positive_count = int((y_churn_train == 1).sum())
scale_pos_weight = negative_count / max(positive_count, 1)

classification_models = {
    "Majority Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    ),
}

classification_results = []
classification_details = {}

print("\n" + "=" * 68)
print("CLASSIFICATION: CHURN WITHIN 90 DAYS")
print("=" * 68)

for name, model in classification_models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_churn_train)

    predicted_labels = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        predicted_probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        predicted_probabilities = predicted_labels.astype(float)

    accuracy = accuracy_score(y_churn_test, predicted_labels)
    precision = precision_score(y_churn_test, predicted_labels, zero_division=0)
    recall = recall_score(y_churn_test, predicted_labels, zero_division=0)
    f1 = f1_score(y_churn_test, predicted_labels, zero_division=0)
    auc = (
        roc_auc_score(y_churn_test, predicted_probabilities)
        if len(np.unique(y_churn_test)) == 2
        else np.nan
    )

    classification_results.append(
        {
            "model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": auc,
        }
    )
    classification_details[name] = {
        "labels": predicted_labels,
        "probabilities": predicted_probabilities,
        "pipeline": pipeline,
    }

    print(
        f"{name:22s} | Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
    )

classification_df = pd.DataFrame(classification_results).sort_values(
    "F1", ascending=False
)
classification_df.to_csv(OUTPUT_DIR / "churn_model_results.csv", index=False)

best_churn_name = classification_df.iloc[0]["model"]
best_churn = classification_details[best_churn_name]

pd.DataFrame(
    {
        "actual_churn": y_churn_test.to_numpy(),
        "predicted_churn": best_churn["labels"],
        "predicted_churn_probability": best_churn["probabilities"],
    }
).to_csv(OUTPUT_DIR / "best_churn_test_predictions.csv", index=False)

print("\nBest regression model by MAE:", best_regression_name)
print("Best churn model by F1:", best_churn_name)
print("\nConfusion matrix for best churn model (rows=actual, columns=predicted):")
print(confusion_matrix(y_churn_test, best_churn["labels"]))
print("\nClassification report for best churn model:")
print(
    classification_report(
        y_churn_test,
        best_churn["labels"],
        zero_division=0,
    )
)

# --------------------------------------------------
# 8. SAVE SUMMARY
# --------------------------------------------------

summary_file = OUTPUT_DIR / "model_training_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:
    file.write("MODEL TRAINING SUMMARY\n")
    file.write("=" * 60 + "\n\n")
    file.write(f"Snapshot rows: {len(df)}\n")
    file.write(f"Features used: {X.shape[1]}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(
        "Split: random 80/20 split stratified by churn target. "
        "This is not temporal validation.\n\n"
    )
    file.write("REGRESSION RESULTS\n")
    file.write(regression_df.to_string(index=False))
    file.write("\n\nCHURN CLASSIFICATION RESULTS\n")
    file.write(classification_df.to_string(index=False))
    file.write("\n\n")
    file.write(f"Best regression model by MAE: {best_regression_name}\n")
    file.write(f"Best churn model by F1: {best_churn_name}\n\n")
    file.write(
        "Limitations:\n"
        "- The dataset is synthetic.\n"
        "- Evaluation uses one fixed-cutoff snapshot and a random split.\n"
        "- Performance across future dates has not been established.\n"
        "- Future 90-day successful-payment revenue is a short-term "
        "customer-value proxy, not lifetime profit.\n"
        "- Model comparisons on this test set are preliminary; use a "
        "separate validation strategy for final model selection.\n"
    )

print("\n" + "=" * 68)
print("TRAINING COMPLETED")
print("=" * 68)
print("\nSaved files:")
for filename in [
    "regression_model_results.csv",
    "churn_model_results.csv",
    "best_regression_test_predictions.csv",
    "best_churn_test_predictions.csv",
    "model_training_summary.txt",
]:
    print(OUTPUT_DIR / filename)

print("\nNext: review results and investigate model validity before final selection.")
