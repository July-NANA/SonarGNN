
"""Generate Member F report figures from currently available teammate results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


CURRENT_LOSS_RESULTS = {
    "CE": {
        "accuracy": 0.9723,
        "macro_f1": 0.4930,
        "illicit_precision": 0.0000,
        "illicit_recall": 0.0000,
        "illicit_f1": 0.0000,
    },
    "Weighted CE": {
        "accuracy": 0.9297,
        "macro_f1": 0.4860,
        "illicit_precision": 0.0066,
        "illicit_recall": 0.0118,
        "illicit_f1": 0.0084,
    },
}

CURRENT_BATCH_RESULTS = {
    512: {"val_accuracy": 0.9394, "seconds_per_batch": 0.97},
    1024: {"val_accuracy": 0.9304, "seconds_per_batch": 0.51},
    2048: {"val_accuracy": 0.9372, "seconds_per_batch": 0.61},
    4096: {"val_accuracy": 0.9317, "seconds_per_batch": 0.65},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_ce_vs_wce(output_dir: Path) -> None:
    metrics = ["accuracy", "macro_f1", "illicit_precision", "illicit_recall", "illicit_f1"]
    labels = ["Accuracy", "Macro F1", "Illicit Prec", "Illicit Recall", "Illicit F1"]
    x = range(len(metrics))
    width = 0.35

    ce_vals = [CURRENT_LOSS_RESULTS["CE"][m] for m in metrics]
    wce_vals = [CURRENT_LOSS_RESULTS["Weighted CE"][m] for m in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], ce_vals, width=width, label="CE")
    plt.bar([i + width/2 for i in x], wce_vals, width=width, label="Weighted CE")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("CE vs Weighted CE: Imbalance-Sensitive Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "memberF_ce_vs_wce_bar.png", dpi=150)
    plt.close()


def plot_batch_tradeoff(output_dir: Path) -> None:
    batch_sizes = sorted(CURRENT_BATCH_RESULTS.keys())
    times = [CURRENT_BATCH_RESULTS[b]["seconds_per_batch"] for b in batch_sizes]
    accs = [CURRENT_BATCH_RESULTS[b]["val_accuracy"] for b in batch_sizes]

    plt.figure(figsize=(8, 5))
    plt.scatter(times, accs, s=120)
    for batch, time_s, acc in zip(batch_sizes, times, accs):
        plt.annotate(f"bs={batch}", (time_s, acc), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Average Seconds per Batch")
    plt.ylabel("Validation Accuracy")
    plt.title("Batch Size Trade-off: Speed vs Validation Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "memberF_batch_tradeoff_scatter.png", dpi=150)
    plt.close()


def save_current_summary_json(output_dir: Path) -> None:
    payload = {
        "title": "Current Available Comparison",
        "shared_metrics": ["accuracy", "macro_f1", "illicit_recall", "illicit_f1"],
        "experiments": CURRENT_LOSS_RESULTS,
        "notes": {
            "batch_results": CURRENT_BATCH_RESULTS,
            "lr_recommendation": "0.01",
            "lr_note": "Moderate LR=0.01 showed the best stability and convergence according to Member D.",
        },
    }
    (output_dir / "memberF_current_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    output_dir = Path("docs/generated")
    ensure_dir(output_dir)
    plot_ce_vs_wce(output_dir)
    plot_batch_tradeoff(output_dir)
    save_current_summary_json(output_dir)
    print(f"Saved current report figures to {output_dir}")


if __name__ == "__main__":
    main()
