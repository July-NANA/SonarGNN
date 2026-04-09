import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set a unified plotting style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["savefig.dpi"] = 150


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_training_curves(*args, **kwargs):
    """
    Supports two calling styles:
    1) plot_training_curves(metrics_dict, save_path=..., title_prefix=...)
    2) plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=...)
    """
    if args and isinstance(args[0], dict):
        metrics = args[0]
        save_path = kwargs.get("save_path", "training_curves.png")
        title_prefix = kwargs.get("title_prefix", "Training")

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
        return

    if len(args) < 4:
        raise ValueError("Expected either a metrics dict or 4 arrays for losses/accs")

    train_losses, val_losses, train_accs, val_accs = args[:4]
    save_path = kwargs.get("save_path", "training_curves.png")

    _ensure_dir(save_path)
    plt.figure(figsize=(12, 4))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy", color="#1f77b4", linewidth=2)
    plt.plot(val_accs, label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

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


def plot_lr_comparison(results_dict, save_path="lr_comparison.png"):
    """Compare training effects of different learning rates"""
    _ensure_dir(save_path)
    plt.figure(figsize=(12, 4))

    # Loss comparison
    plt.subplot(1, 2, 1)
    for lr, metrics in results_dict.items():
        plt.plot(metrics["train_losses"], label=f"LR={lr}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Comparison")

    # Accuracy comparison
    plt.subplot(1, 2, 2)
    for lr, metrics in results_dict.items():
        plt.plot(metrics["val_accs"], label=f"LR={lr}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.title("Validation Accuracy Comparison")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
