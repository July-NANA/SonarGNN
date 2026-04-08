
"""Evaluation metrics for imbalanced binary node classification.

This module avoids heavy third-party dependencies so it can run in a light
project environment. It supports the SonarGNN setting where labels can contain
-1 for unknown nodes that should be ignored during evaluation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np


ArrayLike = Sequence[int] | Sequence[float] | np.ndarray


def _to_numpy(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape={arr.shape}")
    return arr


def filter_valid_labels(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    y_score: Optional[ArrayLike] = None,
    ignore_label: int = -1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Remove samples whose true label equals ``ignore_label``."""
    true_arr = _to_numpy(y_true).astype(int)
    valid_mask = true_arr != ignore_label
    true_arr = true_arr[valid_mask]

    pred_arr = None
    if y_pred is not None:
        pred_arr = _to_numpy(y_pred).astype(int)
        if pred_arr.shape[0] != valid_mask.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")
        pred_arr = pred_arr[valid_mask]

    score_arr = None
    if y_score is not None:
        score_arr = _to_numpy(y_score).astype(float)
        if score_arr.shape[0] != valid_mask.shape[0]:
            raise ValueError("y_true and y_score must have the same length")
        score_arr = score_arr[valid_mask]

    return true_arr, pred_arr, score_arr


def confusion_counts(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    positive_label: int = 1,
    ignore_label: int = -1,
) -> Dict[str, int]:
    """Return TP, FP, TN, FN counts for binary classification."""
    true_arr, pred_arr, _ = filter_valid_labels(y_true, y_pred=y_pred, ignore_label=ignore_label)
    if pred_arr is None:
        raise ValueError("y_pred is required")

    pos = positive_label
    neg_mask_true = true_arr != pos
    neg_mask_pred = pred_arr != pos

    tp = int(np.sum((true_arr == pos) & (pred_arr == pos)))
    fp = int(np.sum(neg_mask_true & (pred_arr == pos)))
    tn = int(np.sum(neg_mask_true & neg_mask_pred))
    fn = int(np.sum((true_arr == pos) & neg_mask_pred))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def accuracy_score_from_counts(counts: Dict[str, int]) -> float:
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    return _safe_divide(counts["tp"] + counts["tn"], total)


def precision_score_from_counts(counts: Dict[str, int]) -> float:
    return _safe_divide(counts["tp"], counts["tp"] + counts["fp"])


def recall_score_from_counts(counts: Dict[str, int]) -> float:
    return _safe_divide(counts["tp"], counts["tp"] + counts["fn"])


def specificity_score_from_counts(counts: Dict[str, int]) -> float:
    return _safe_divide(counts["tn"], counts["tn"] + counts["fp"])


def f1_score_from_counts(counts: Dict[str, int]) -> float:
    precision = precision_score_from_counts(counts)
    recall = recall_score_from_counts(counts)
    return _safe_divide(2 * precision * recall, precision + recall)


def balanced_accuracy_from_counts(counts: Dict[str, int]) -> float:
    recall = recall_score_from_counts(counts)
    specificity = specificity_score_from_counts(counts)
    return 0.5 * (recall + specificity)


def _binary_labels(y_true: np.ndarray, positive_label: int) -> np.ndarray:
    return (y_true == positive_label).astype(int)


def roc_curve_points(
    y_true: ArrayLike,
    y_score: ArrayLike,
    positive_label: int = 1,
    ignore_label: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return FPR and TPR points for a manual ROC curve."""
    true_arr, _, score_arr = filter_valid_labels(y_true, y_score=y_score, ignore_label=ignore_label)
    if score_arr is None:
        raise ValueError("y_score is required for ROC-AUC")

    y_bin = _binary_labels(true_arr, positive_label)
    pos_total = int(np.sum(y_bin == 1))
    neg_total = int(np.sum(y_bin == 0))
    if pos_total == 0 or neg_total == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-score_arr)
    y_sorted = y_bin[order]
    score_sorted = score_arr[order]

    tps = 0
    fps = 0
    fpr_points: List[float] = [0.0]
    tpr_points: List[float] = [0.0]

    for idx, label in enumerate(y_sorted):
        if label == 1:
            tps += 1
        else:
            fps += 1

        next_is_new_threshold = idx == len(score_sorted) - 1 or score_sorted[idx + 1] != score_sorted[idx]
        if next_is_new_threshold:
            fpr_points.append(fps / neg_total)
            tpr_points.append(tps / pos_total)

    if fpr_points[-1] != 1.0 or tpr_points[-1] != 1.0:
        fpr_points.append(1.0)
        tpr_points.append(1.0)

    return np.asarray(fpr_points), np.asarray(tpr_points)


def pr_curve_points(
    y_true: ArrayLike,
    y_score: ArrayLike,
    positive_label: int = 1,
    ignore_label: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return recall and precision points for a manual PR curve."""
    true_arr, _, score_arr = filter_valid_labels(y_true, y_score=y_score, ignore_label=ignore_label)
    if score_arr is None:
        raise ValueError("y_score is required for PR-AUC")

    y_bin = _binary_labels(true_arr, positive_label)
    pos_total = int(np.sum(y_bin == 1))
    if pos_total == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    order = np.argsort(-score_arr)
    y_sorted = y_bin[order]
    score_sorted = score_arr[order]

    tps = 0
    fps = 0
    recall_points: List[float] = [0.0]
    precision_points: List[float] = [1.0]

    for idx, label in enumerate(y_sorted):
        if label == 1:
            tps += 1
        else:
            fps += 1

        next_is_new_threshold = idx == len(score_sorted) - 1 or score_sorted[idx + 1] != score_sorted[idx]
        if next_is_new_threshold:
            recall = tps / pos_total
            precision = _safe_divide(tps, tps + fps)
            recall_points.append(recall)
            precision_points.append(precision)

    if recall_points[-1] != 1.0:
        recall_points.append(1.0)
        precision_points.append(precision_points[-1])

    return np.asarray(recall_points), np.asarray(precision_points)


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def roc_auc_score_manual(
    y_true: ArrayLike,
    y_score: ArrayLike,
    positive_label: int = 1,
    ignore_label: int = -1,
) -> Optional[float]:
    fpr, tpr = roc_curve_points(y_true, y_score, positive_label=positive_label, ignore_label=ignore_label)
    if len(fpr) < 2:
        return None
    return auc_trapezoid(fpr, tpr)


def pr_auc_score_manual(
    y_true: ArrayLike,
    y_score: ArrayLike,
    positive_label: int = 1,
    ignore_label: int = -1,
) -> Optional[float]:
    recall, precision = pr_curve_points(y_true, y_score, positive_label=positive_label, ignore_label=ignore_label)
    if len(recall) < 2:
        return None
    return auc_trapezoid(recall, precision)


def compute_binary_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: Optional[ArrayLike] = None,
    positive_label: int = 1,
    ignore_label: int = -1,
) -> Dict[str, float | int | None]:
    """Compute a metric bundle used in SonarGNN reports.

    Returned metrics are computed after dropping unknown labels (default: -1).
    """
    true_arr, pred_arr, score_arr = filter_valid_labels(
        y_true, y_pred=y_pred, y_score=y_score, ignore_label=ignore_label
    )
    if pred_arr is None:
        raise ValueError("y_pred is required")

    counts = confusion_counts(true_arr, pred_arr, positive_label=positive_label, ignore_label=999999)
    metrics: Dict[str, float | int | None] = {
        "num_samples": int(true_arr.shape[0]),
        "positive_support": int(np.sum(true_arr == positive_label)),
        "negative_support": int(np.sum(true_arr != positive_label)),
        "accuracy": round(accuracy_score_from_counts(counts), 6),
        "precision": round(precision_score_from_counts(counts), 6),
        "recall": round(recall_score_from_counts(counts), 6),
        "f1_score": round(f1_score_from_counts(counts), 6),
        "specificity": round(specificity_score_from_counts(counts), 6),
        "balanced_accuracy": round(balanced_accuracy_from_counts(counts), 6),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "tn": counts["tn"],
        "fn": counts["fn"],
    }

    if score_arr is not None:
        roc_auc = roc_auc_score_manual(true_arr, score_arr, positive_label=positive_label, ignore_label=999999)
        pr_auc = pr_auc_score_manual(true_arr, score_arr, positive_label=positive_label, ignore_label=999999)
        metrics["roc_auc"] = None if roc_auc is None else round(roc_auc, 6)
        metrics["pr_auc"] = None if pr_auc is None else round(pr_auc, 6)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics
