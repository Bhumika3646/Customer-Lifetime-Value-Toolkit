"""SHAP summaries for supervised revenue-proxy and churn models."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURE_COLUMNS
from . import ml_models


def _original_feature(transformed_name: str) -> str | None:
    name = str(transformed_name)
    if "__" in name:
        name = name.split("__", 1)[1]
    if name.startswith("missingindicator_"):
        name = name[len("missingindicator_"):]
    # Match categorical source names before interpreting the remainder as a numeric feature.
    for col in sorted(CATEGORICAL, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name if name in FEATURE_COLUMNS else None


def _normalise_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        # Binary classifiers commonly return one matrix per class; explain positive class.
        values = values[1] if len(values) > 1 else values[0]
    if hasattr(values, "values") and not isinstance(values, np.ndarray):
        values = values.values
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Newer SHAP versions may append a class dimension.
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape {arr.shape}")
    return arr


def explain_fitted_model(
    pipeline,
    test_frame: pd.DataFrame,
    task: str,
    output_dir: Path,
    sample_size: int = 160,
    local_examples: int = 5,
    local_top_features: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write grouped global SHAP and a few local example explanations.

    Revenue SHAP values are in the 90-day revenue target's units (currency units
    in this dataset); binary churn SHAP values are contributions to model log-odds.
    Customer IDs are deliberately not written to the local explanation file.
    """
    import shap

    if test_frame.empty:
        raise ValueError(f"No held-out records available for SHAP task={task!r}")
    n = min(int(sample_size), len(test_frame))
    sample = test_frame.sample(n=n, random_state=ml_models.C.RANDOM_STATE).reset_index(drop=True)
    X_sample = ml_models._X(sample)
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocess.transform(X_sample)
    transformed_names = preprocess.get_feature_names_out()
    explainer = shap.TreeExplainer(estimator)
    values = _normalise_shap_values(explainer.shap_values(transformed, check_additivity=False))
    if values.shape[1] != len(transformed_names):
        raise ValueError(f"SHAP feature count {values.shape[1]} != transformed feature count {len(transformed_names)}")

    grouped: dict[str, list[int]] = {}
    for idx, feature_name in enumerate(transformed_names):
        original = _original_feature(feature_name)
        if original is not None:
            grouped.setdefault(original, []).append(idx)
    if not grouped:
        raise ValueError("Could not map transformed SHAP features back to original feature names")

    grouped_values = {}
    for original, indices in grouped.items():
        # Add contributions from one-hot columns to recover source-feature contribution.
        grouped_values[original] = values[:, indices].sum(axis=1)

    scale_note = ("90-day future revenue prediction units; SHAP explains the supervised revenue proxy, not the survival CLV calculation directly."
                  if task == "90-day revenue / CLV proxy" else
                  "Log-odds contribution for the supervised 3-month churn classifier; not a direct probability change.")
    global_rows = []
    for feature, contribution in grouped_values.items():
        global_rows.append({
            "task": task, "model": "XGBoost", "source_feature": feature,
            "mean_abs_shap": float(np.mean(np.abs(contribution))),
            "mean_signed_shap": float(np.mean(contribution)),
            "sample_rows": int(n), "interpretation_scale": scale_note,
        })
    global_df = pd.DataFrame(global_rows).sort_values("mean_abs_shap", ascending=False)

    local_rows = []
    for row_idx in range(min(int(local_examples), n)):
        ranking = sorted(grouped_values, key=lambda f: abs(grouped_values[f][row_idx]), reverse=True)
        for feature in ranking[:int(local_top_features)]:
            local_rows.append({
                "task": task,
                "example_number": row_idx + 1,
                "source_feature": feature,
                "feature_value": str(sample.iloc[row_idx].get(feature, "")),
                "shap_value": float(grouped_values[feature][row_idx]),
                "interpretation_scale": scale_note,
            })
    return global_df, pd.DataFrame(local_rows)


def write_shap_outputs(model_and_data: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Build both SHAP outputs; report failures honestly rather than hide them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, locals_ = [], []
    failures = []
    for task_key, task_label in (("revenue", "90-day revenue / CLV proxy"), ("churn", "3-month churn risk")):
        item = model_and_data.get(task_key)
        if not item:
            failures.append(f"{task_label}: XGBoost model/test data unavailable")
            continue
        try:
            g, l = explain_fitted_model(item["pipeline"], item["test_frame"], task_label, output_dir)
            summaries.append(g)
            locals_.append(l)
        except Exception as exc:  # preserve model evaluation even if SHAP is incompatible
            failures.append(f"{task_label}: {type(exc).__name__}: {exc}")
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    local = pd.concat(locals_, ignore_index=True) if locals_ else pd.DataFrame()
    summary_path = output_dir / "08_shap_global_summary.csv"
    local_path = output_dir / "08_shap_local_examples.csv"
    status_path = output_dir / "08_shap_status.txt"
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    else:
        summary_path.unlink(missing_ok=True)
    if not local.empty:
        local.to_csv(local_path, index=False)
    else:
        local_path.unlink(missing_ok=True)
    status = "SHAP generated for both models." if not failures else "SHAP incomplete: " + " | ".join(failures)
    status_path.write_text(status + "\n", encoding="utf-8")
    return summary, local, status
