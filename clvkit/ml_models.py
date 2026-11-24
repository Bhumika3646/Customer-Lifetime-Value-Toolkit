"""Machine-learning comparators (gradient boosting) for revenue and churn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from . import config as C
from .features import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, assert_no_leakage


def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
         ("num", "passthrough", NUMERIC)])
    n_cat = len(CATEGORICAL)
    estimator.set_params(categorical_features=[True] * n_cat + [False] * len(NUMERIC))
    return Pipeline([("pre", pre), ("model", estimator)])


def _X(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage(FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return X


def fit_predict_revenue(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    est = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4,
                                        random_state=C.RANDOM_STATE)
    pipe = _pipeline(est).fit(_X(train), train[target])
    return np.clip(pipe.predict(_X(test)), 0, None)


def fit_predict_churn(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    est = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                         random_state=C.RANDOM_STATE)
    pipe = _pipeline(est).fit(_X(train), train[target])
    return pipe.predict_proba(_X(test))[:, 1]


# Model families required by the original project objectives. These use a shared
# preprocessing definition and the same customer-grouped holdout for fair comparison.
REGRESSION_MODEL_NAMES = ("Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost")
CLASSIFICATION_MODEL_NAMES = ("Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost")


def _comparison_preprocessor() -> ColumnTransformer:
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("cat", cat, CATEGORICAL), ("num", num, NUMERIC)],
        remainder="drop", verbose_feature_names_out=True,
    )


def comparison_pipeline(model_name: str, task: str) -> Pipeline:
    """Create a reproducible regression/classification model pipeline.

    `task` must be `regression` or `classification`. Categorical features are
    one-hot encoded with unseen values handled safely; numeric missing values
    are imputed from the training set only.
    """
    task = task.casefold().strip()
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")

    if task == "regression":
        estimators = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=120, min_samples_leaf=3, n_jobs=-1, random_state=C.RANDOM_STATE
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=140, learning_rate=0.04, max_depth=2, loss="squared_error",
                random_state=C.RANDOM_STATE,
            ),
        }
    else:
        estimators = {
            "Logistic Regression": LogisticRegression(
                max_iter=1200, solver="lbfgs", random_state=C.RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120, min_samples_leaf=3,
                n_jobs=-1, random_state=C.RANDOM_STATE,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=140, learning_rate=0.04, max_depth=2, random_state=C.RANDOM_STATE,
            ),
        }
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise ImportError("XGBoost comparison requires xgboost; install requirements.txt") from exc

    if task == "regression":
        estimators["XGBoost"] = XGBRegressor(
            n_estimators=180, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            objective="reg:squarederror", eval_metric="rmse",
            n_jobs=4, random_state=C.RANDOM_STATE,
        )
    else:
        estimators["XGBoost"] = XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=4, random_state=C.RANDOM_STATE,
        )

    if model_name not in estimators:
        raise ValueError(f"Unknown {task} model {model_name!r}; choose from {list(estimators)}")
    return Pipeline([("preprocess", _comparison_preprocessor()), ("model", estimators[model_name])])


def fit_comparison_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task: str,
) -> dict[str, dict]:
    """Fit all four objective-aligned models and return their predictions/pipelines."""
    assert_no_leakage(FEATURE_COLUMNS)
    model_names = REGRESSION_MODEL_NAMES if task == "regression" else CLASSIFICATION_MODEL_NAMES
    X_train = _X(train)
    X_test = _X(test)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy()
    fitted = {}
    for name in model_names:
        pipeline = comparison_pipeline(name, task)
        pipeline.fit(X_train, y_train)
        if task == "regression":
            prediction = np.clip(pipeline.predict(X_test), 0, None)
        else:
            prediction = np.clip(pipeline.predict_proba(X_test)[:, 1], 0.0, 1.0)
        fitted[name] = {"pipeline": pipeline, "prediction": prediction}
    return fitted


def score_regression_predictions(y_true, prediction) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else float("nan"),
        "portfolio_bias_pct": float(100.0 * (p.sum() / y.sum() - 1.0)) if y.sum() > 0 else float("nan"),
    }


def score_classification_predictions(y_true, probability) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if len(np.unique(y)) < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
    return {
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Brier": float(brier_score_loss(y, p)),
        "calibration_gap": float(p.mean() - y.mean()),
    }
