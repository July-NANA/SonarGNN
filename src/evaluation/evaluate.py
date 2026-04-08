
"""CLI evaluation entrypoint for SonarGNN.

Usage examples:
    python -m src.evaluation.evaluate --input results/predictions.csv --output-dir results/eval
    python -m src.evaluation.evaluate --input results/predictions.npz --output-dir results/eval
    python -m src.evaluation.evaluate --summary-json data/summary/memberF_current_summary.json --output-dir docs/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Support both:
#   1) python -m src.evaluation.evaluate
#   2) python src/evaluation/evaluate.py
if __package__ is None or __package__ == "":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_binary_classification_metrics


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_predictions(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        y_score = data["y_score"] if "y_score" in data.files else None
        return y_true, y_pred, y_score

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        y_true = np.asarray(payload["y_true"])
        y_pred = np.asarray(payload["y_pred"])
        y_score = np.asarray(payload["y_score"]) if "y_score" in payload else None
        return y_true, y_pred, y_score

    if suffix == ".csv":
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[float] = []
        has_score = False
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [name.strip().lstrip("\ufeff") for name in (reader.fieldnames or [])]
            reader.fieldnames = fieldnames
            required = {"y_true", "y_pred"}
            if not required.issubset(fieldnames):
                raise ValueError("CSV must contain columns: y_true, y_pred, and optionally y_score")
            has_score = "y_score" in fieldnames
            for row in reader:
                y_true.append(int(float(row["y_true"])))
                y_pred.append(int(float(row["y_pred"])))
                if has_score:
                    y_score.append(float(row["y_score"]))
        return np.asarray(y_true), np.asarray(y_pred), (np.asarray(y_score) if has_score else None)

    raise ValueError(f"Unsupported file format: {suffix}")


def save_metrics(metrics: Dict[str, object], output_dir: str) -> None:
    ensure_dir(output_dir)
    out_dir = Path(output_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def plot_metrics_bar(metrics: Dict[str, object], output_dir: str, title: str = "Evaluation Metrics") -> None:
    ensure_dir(output_dir)
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]
    labels = []
    values = []
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        labels.append(key)
        values.append(float(value))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "metrics_bar.png", dpi=150)
    plt.close()


def _radar_angles(num_vars: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    return np.concatenate([angles, [angles[0]]])


def plot_summary_comparison(summary: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Create a bar chart and radar chart from a manual summary JSON.

    The JSON format should be:
    {
      "shared_metrics": ["accuracy", "f1_score", "pr_auc"],
      "experiments": {
         "Baseline": {"accuracy": 0.93, "f1_score": 0.48, "pr_auc": 0.12},
         "Weighted CE": {...}
      }
    }
    """
    ensure_dir(output_dir)
    shared_metrics = summary["shared_metrics"]
    experiments = summary["experiments"]

    # grouped bar chart
    exp_names = list(experiments.keys())
    x = np.arange(len(shared_metrics))
    width = 0.8 / max(len(exp_names), 1)

    plt.figure(figsize=(10, 5))
    for idx, exp_name in enumerate(exp_names):
        vals = [experiments[exp_name].get(metric, np.nan) for metric in shared_metrics]
        plt.bar(x + idx * width, vals, width=width, label=exp_name)

    plt.xticks(x + width * (len(exp_names) - 1) / 2, shared_metrics, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title(summary.get("title", "Experiment Comparison"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "summary_grouped_bar.png", dpi=150)
    plt.close()

    # radar chart
    angles = _radar_angles(len(shared_metrics))
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    for exp_name in exp_names:
        vals = [experiments[exp_name].get(metric, 0.0) for metric in shared_metrics]
        vals = np.asarray(vals + [vals[0]])
        ax.plot(angles, vals, linewidth=2, label=exp_name)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(shared_metrics)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(summary.get("title", "Experiment Comparison"), pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "summary_radar.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SonarGNN prediction outputs.")
    parser.add_argument("--input", type=str, default=None, help="CSV/JSON/NPZ file with y_true, y_pred, and optional y_score")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON for grouped comparison charts")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Directory to save metrics and plots")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--title", type=str, default="Evaluation Metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.summary_json:
        summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
        plot_summary_comparison(summary, args.output_dir)
        print(f"Saved summary comparison charts to: {args.output_dir}")

    if args.input:
        y_true, y_pred, y_score = load_predictions(args.input)
        metrics = compute_binary_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            positive_label=args.positive_label,
            ignore_label=args.ignore_label,
        )
        save_metrics(metrics, args.output_dir)
        plot_metrics_bar(metrics, args.output_dir, title=args.title)
        print(json.dumps(metrics, indent=2))
        print(f"Saved metrics and plots to: {args.output_dir}")

    if not args.input and not args.summary_json:
        raise SystemExit("Nothing to do. Provide --input and/or --summary-json.")


if __name__ == "__main__":
    main()
