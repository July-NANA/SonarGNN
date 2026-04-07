import os
import matplotlib.pyplot as plt


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_training_curves(metrics, save_path, title_prefix):
    """
    Plot training/validation loss and accuracy for a single run.

    metrics: dict with keys train_loss, val_loss, train_acc, val_acc
    """
    _ensure_dir(save_path)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{title_prefix} Loss")

    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc"], label="Train Acc")
    plt.plot(metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{title_prefix} Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_loss_comparison(ce_metrics, wce_metrics, save_path):
    """
    Compare CE vs Weighted CE on validation curves.
    """
    _ensure_dir(save_path)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ce_metrics["val_loss"], label="CE Val Loss")
    plt.plot(wce_metrics["val_loss"], label="Weighted CE Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Validation Loss Comparison")

    plt.subplot(1, 2, 2)
    plt.plot(ce_metrics["val_acc"], label="CE Val Acc")
    plt.plot(wce_metrics["val_acc"], label="Weighted CE Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy Comparison")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
