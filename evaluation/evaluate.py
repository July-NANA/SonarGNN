"""Standalone evaluation entry point for SonarGNN.

Usage examples
--------------
1) Run with dummy data to verify the evaluation pipeline:
   python evaluate.py --mode dummy

2) Evaluate one model from a CSV of predictions:
   python evaluate.py --mode single --predictions outputs/predictions.csv --model-name baseline

   CSV columns:
   y_true,y_pred,y_prob
   1,1,0.91
   0,0,0.12
   ...

3) Visualize multiple experiment results:
   python evaluate.py --mode summary --summary-csv outputs/model_summary.csv

   CSV columns:
   model,accuracy,precision,recall,f1,pr_auc,roc_auc
   baseline,0.94,0.61,0.43,0.50,0.58,0.89
   weighted_ce,0.93,0.56,0.68,0.61,0.65,0.90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    DEFAULT_METRICS,
    compute_classification_metrics,
    load_prediction_csv,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_pr_curve,
    plot_radar_chart,
    save_metrics_csv,
    save_metrics_json,
    summarize_models_from_csv,
)


def generate_dummy_predictions(
    n_samples: int = 1000,
    positive_rate: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate imbalanced dummy data for pipeline testing.

    Returns
    -------
    single_df:
        Dummy predictions for a single model.
    summary_df:
        Dummy comparison table for several model variants.
    """
    rng = np.random.default_rng(seed)

    y_true = (rng.random(n_samples) < positive_rate).astype(int)

    # Pretend this is the team's final best model.
    logits = rng.normal(loc=-1.8, scale=1.1, size=n_samples)
    logits += y_true * 2.4
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.50).astype(int)

    single_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})

    # Dummy multi-model summary that mimics likely experiment outcomes.
    summary_df = pd.DataFrame(
        [
            {"model": "baseline", "accuracy": 0.938, "precision": 0.600, "recall": 0.410, "f1": 0.487, "pr_auc": 0.561, "roc_auc": 0.865},
            {"model": "weighted_ce", "accuracy": 0.931, "precision": 0.545, "recall": 0.670, "f1": 0.601, "pr_auc": 0.648, "roc_auc": 0.886},
            {"model": "best_lr", "accuracy": 0.936, "precision": 0.587, "recall": 0.640, "f1": 0.612, "pr_auc": 0.672, "roc_auc": 0.894},
            {"model": "neighbor_loader", "accuracy": 0.934, "precision": 0.571, "recall": 0.690, "f1": 0.625, "pr_auc": 0.689, "roc_auc": 0.901},
            {"model": "final_model", "accuracy": 0.942, "precision": 0.622, "recall": 0.702, "f1": 0.659, "pr_auc": 0.718, "roc_auc": 0.913},
        ]
    )

    return single_df, summary_df


def evaluate_single(predictions_path: Path, output_dir: Path, model_name: str) -> None:
    """Evaluate one set of predictions and save artifacts."""
    df = load_prediction_csv(predictions_path)

    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy() if "y_prob" in df.columns else None
    if "y_pred" in df.columns:
        y_pred = df["y_pred"].to_numpy()
    elif y_prob is not None:
        y_pred = (y_prob >= 0.50).astype(int)
    else:
        raise ValueError("Need y_pred or y_prob to derive predictions.")

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_dir / f"{model_name}_metrics.json")
    save_metrics_csv(metrics, output_dir / f"{model_name}_metrics.csv", model_name=model_name)
    plot_confusion_matrix(y_true, y_pred, output_dir / f"{model_name}_confusion_matrix.png", title=f"{model_name} Confusion Matrix")
    if y_prob is not None:
        plot_pr_curve(y_true, y_prob, output_dir / f"{model_name}_pr_curve.png", title=f"{model_name} PR Curve")

    print(f"[Done] Saved single-model evaluation artifacts to: {output_dir}")
    print(pd.DataFrame([{"model": model_name, **metrics}]).round(4).to_string(index=False))


def evaluate_summary(summary_csv: Path, output_dir: Path) -> None:
    """Create summary plots across multiple models."""
    summary_df = summarize_models_from_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "model_summary_verified.csv", index=False)
    plot_metric_bars(summary_df, output_dir / "model_comparison_bar.png", metrics=DEFAULT_METRICS, title="SonarGNN Model Comparison")
    plot_radar_chart(summary_df, output_dir / "model_comparison_radar.png", metrics=DEFAULT_METRICS, title="SonarGNN Radar Comparison")

    print(f"[Done] Saved summary comparison artifacts to: {output_dir}")
    print(summary_df.round(4).to_string(index=False))


def run_dummy(output_dir: Path) -> None:
    """Run a fully self-contained dummy evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    single_df, summary_df = generate_dummy_predictions()

    dummy_predictions_path = output_dir / "dummy_predictions.csv"
    dummy_summary_path = output_dir / "dummy_model_summary.csv"
    single_df.to_csv(dummy_predictions_path, index=False)
    summary_df.to_csv(dummy_summary_path, index=False)

    evaluate_single(dummy_predictions_path, output_dir, model_name="dummy_final_model")
    evaluate_summary(dummy_summary_path, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SonarGNN predictions and compare experiments.")
    parser.add_argument("--mode", choices=["dummy", "single", "summary"], default="dummy")
    parser.add_argument("--predictions", type=Path, help="CSV file containing y_true/y_pred/y_prob columns.")
    parser.add_argument("--summary-csv", type=Path, help="CSV file containing one row per model with metric columns.")
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "dummy":
        run_dummy(args.output_dir)
    elif args.mode == "single":
        if args.predictions is None:
            raise ValueError("--predictions is required in single mode.")
        evaluate_single(args.predictions, args.output_dir, model_name=args.model_name)
    elif args.mode == "summary":
        if args.summary_csv is None:
            raise ValueError("--summary-csv is required in summary mode.")
        evaluate_summary(args.summary_csv, args.output_dir)


if __name__ == "__main__":
    main()
