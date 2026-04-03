"""Evaluation utilities for imbalanced node classification.

This module centralizes metric computation and visualization so the team can
reuse the same logic across baseline and ablation experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]


def _to_numpy(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert list-like input to a 1D NumPy array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}.")
    return arr


def _safe_metric(metric_fn, *args, default: float = 0.0, **kwargs) -> float:
    """Return metric value or a safe default when sklearn raises edge-case errors."""
    try:
        value = metric_fn(*args, **kwargs)
    except ValueError:
        value = default
    return float(value)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Optional[Sequence[float] | np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics with emphasis on imbalance-sensitive scores.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    y_prob:
        Positive-class probabilities or scores. Needed for PR-AUC and ROC-AUC.
    """
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    metrics.update(
        {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "support": int(len(y_true_arr)),
            "positive_rate": float(np.mean(y_true_arr == 1)),
            "predicted_positive_rate": float(np.mean(y_pred_arr == 1)),
        }
    )

    if y_prob is None:
        metrics["pr_auc"] = np.nan
        metrics["roc_auc"] = np.nan
    else:
        y_prob_arr = _to_numpy(y_prob)
        if y_prob_arr.shape != y_true_arr.shape:
            raise ValueError("y_prob must have the same shape as y_true.")
        metrics["pr_auc"] = _safe_metric(average_precision_score, y_true_arr, y_prob_arr, default=np.nan)
        metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true_arr, y_prob_arr, default=np.nan)

    return metrics


def metrics_to_dataframe(metrics_dict: Dict[str, float], model_name: str = "model") -> pd.DataFrame:
    """Convert a metrics dictionary into a single-row DataFrame."""
    row = {"model": model_name, **metrics_dict}
    return pd.DataFrame([row])


def save_metrics_json(metrics_dict: Dict[str, float], output_path: str | Path) -> None:
    """Persist metrics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics_dict).to_json(output_path, indent=2)


def save_metrics_csv(metrics_dict: Dict[str, float], output_path: str | Path, model_name: str = "model") -> None:
    """Persist metrics as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(metrics_dict, model_name=model_name).to_csv(output_path, index=False)


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix heatmap-like figure using pure matplotlib."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(_to_numpy(y_true), _to_numpy(y_pred), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    save_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot and save a precision-recall curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(_to_numpy(y_true), _to_numpy(y_prob))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(
    summary_df: pd.DataFrame,
    save_path: str | Path,
    metrics: Optional[Iterable[str]] = None,
    title: str = "Model Comparison",
) -> None:
    """Plot grouped bar chart for multiple models and metrics."""
    metrics = list(metrics or DEFAULT_METRICS)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {"model", *metrics}
    missing_cols = required_cols - set(summary_df.columns)
    if missing_cols:
        raise ValueError(f"summary_df is missing columns: {sorted(missing_cols)}")

    plot_df = summary_df[["model", *metrics]].copy()
    x = np.arange(len(metrics))
    n_models = len(plot_df)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        offsets = x - 0.4 + width / 2 + idx * width
        ax.bar(offsets, row[metrics].astype(float).values, width=width, label=row["model"])

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m.endswith("auc") else m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radar_chart(
    summary_df: pd.DataFrame,
    save_path: str | Path,
    metrics: Optional[Iterable[str]] = None,
    title: str = "Radar Comparison",
) -> None:
    """Plot a radar chart comparing multiple models on normalized metrics."""
    metrics = list(metrics or DEFAULT_METRICS)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {"model", *metrics}
    missing_cols = required_cols - set(summary_df.columns)
    if missing_cols:
        raise ValueError(f"summary_df is missing columns: {sorted(missing_cols)}")

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for _, row in summary_df.iterrows():
        values = row[metrics].astype(float).fillna(0.0).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() if m.endswith("auc") else m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_prediction_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a prediction CSV and validate the minimum required columns.

    Expected columns:
        - y_true
        - y_pred or y_prob
    Optional columns:
        - y_prob
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required_any = {"y_true"}
    if not required_any.issubset(df.columns):
        raise ValueError("Prediction CSV must contain 'y_true'.")
    if "y_pred" not in df.columns and "y_prob" not in df.columns:
        raise ValueError("Prediction CSV must contain either 'y_pred' or 'y_prob'.")
    return df


def summarize_models_from_csv(summary_csv: str | Path) -> pd.DataFrame:
    """Load a model-summary CSV for bar/radar visualization."""
    summary_csv = Path(summary_csv)
    df = pd.read_csv(summary_csv)
    required_cols = {"model", *DEFAULT_METRICS}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing_cols)}")
    return df
