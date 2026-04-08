
import numpy as np

from src.evaluation.metrics import compute_binary_classification_metrics, confusion_counts


def test_confusion_counts_ignore_unknown():
    y_true = np.array([1, 0, 1, 0, -1])
    y_pred = np.array([1, 0, 0, 1, 1])
    counts = confusion_counts(y_true, y_pred, positive_label=1, ignore_label=-1)
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_compute_binary_classification_metrics_basic():
    y_true = np.array([1, 1, 0, 0, -1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_score = np.array([0.95, 0.40, 0.10, 0.20, 0.99])
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score=y_score)

    assert metrics["num_samples"] == 4
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert abs(metrics["accuracy"] - 0.75) < 1e-6
    assert abs(metrics["precision"] - 1.0) < 1e-6
    assert abs(metrics["recall"] - 0.5) < 1e-6
    assert abs(metrics["f1_score"] - 0.666667) < 1e-5
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None
