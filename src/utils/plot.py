import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a unified plotting style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["savefig.dpi"] = 150


def plot_training_curves(
    train_losses, val_losses, train_accs, val_accs, save_path="training_curves.png"
):
    """Plot training and validation loss/accuracy curves"""
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
    plt.savefig(save_path)
    plt.close()
    print(f"Curves saved to {save_path}")


def plot_lr_comparison(results_dict, save_path="lr_comparison.png"):
    """Compare training effects of different learning rates"""
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
    plt.savefig(save_path)
    plt.close()
    print(f"LR comparison saved to {save_path}")
