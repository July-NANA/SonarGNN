import argparse
import csv
import os
import sys

import torch
import torch.optim as optim
import yaml

# Add project root to path (for direct script execution)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model import GCN
from src.data.dataset import EllipticDataset
from src.training.loss import compute_class_weights, nll_loss_with_optional_weights
from src.utils.plot import plot_training_curves, plot_loss_comparison

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    print("Loading Elliptic dataset...")
    dataset = EllipticDataset(root="data/Elliptic")
    data = dataset[0]
    print(
        f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges, "
        f"{data.num_node_features} features, {data.num_classes} classes"
    )
    return data


def run_training(data, epochs, lr, hidden_channels, class_weights=None):
    in_channels = data.num_node_features
    out_channels = data.num_classes
    model = GCN(in_channels, hidden_channels, out_channels, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = nll_loss_with_optional_weights(
            out[data.train_mask],
            data.y[data.train_mask],
            class_weights=class_weights,
        )
        loss.backward()
        optimizer.step()
        metrics["train_loss"].append(loss.item())

        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        metrics["train_acc"].append(train_acc)

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = nll_loss_with_optional_weights(
                val_out[data.val_mask],
                data.y[data.val_mask],
                class_weights=class_weights,
            ).item()
            metrics["val_loss"].append(val_loss)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            metrics["val_acc"].append(val_acc)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    return model, metrics


def save_test_comparison_table(data, model_ce, model_wce, save_path, max_rows=100):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_ce.eval()
    model_wce.eval()
    with torch.no_grad():
        out_ce = model_ce(data.x, data.edge_index)
        out_wce = model_wce(data.x, data.edge_index)

    test_indices = torch.nonzero(data.test_mask, as_tuple=False).view(-1)[:max_rows]
    rows = []
    for idx in test_indices:
        idx = idx.item()
        true_label = int(data.y[idx].item())
        pred_ce = int(out_ce[idx].argmax(dim=0).item())
        pred_wce = int(out_wce[idx].argmax(dim=0).item())
        rows.append(
            {
                "node_index": idx,
                "true_label": true_label,
                "pred_ce": pred_ce,
                "pred_weighted_ce": pred_wce,
                "correct_ce": int(pred_ce == true_label),
                "correct_weighted_ce": int(pred_wce == true_label),
            }
        )

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_index",
                "true_label",
                "pred_ce",
                "pred_weighted_ce",
                "correct_ce",
                "correct_weighted_ce",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def train_loss_comparison(epochs, lr, hidden_channels, output_dir):
    data = load_data()
    set_seed(42)

    class_weights = compute_class_weights(
        data.y, num_classes=data.num_classes, mask=data.train_mask
    )
    class_weights = class_weights.to(data.x.device)
    print(f"Class weights: {class_weights.tolist()}")

    print("\nTraining with standard CE...")
    model_ce, ce_metrics = run_training(
        data, epochs=epochs, lr=lr, hidden_channels=hidden_channels, class_weights=None
    )

    print("\nTraining with weighted CE...")
    model_wce, wce_metrics = run_training(
        data, epochs=epochs, lr=lr, hidden_channels=hidden_channels, class_weights=class_weights
    )

    plot_training_curves(
        ce_metrics,
        save_path=os.path.join(output_dir, "ce_curves.png"),
        title_prefix="CE",
    )
    plot_training_curves(
        wce_metrics,
        save_path=os.path.join(output_dir, "weighted_ce_curves.png"),
        title_prefix="Weighted CE",
    )
    plot_loss_comparison(
        ce_metrics,
        wce_metrics,
        save_path=os.path.join(output_dir, "ce_vs_weighted_comparison.png"),
    )
    print(f"Curves saved under {output_dir}")

    save_test_comparison_table(
        data,
        model_ce,
        model_wce,
        save_path=os.path.join(output_dir, "test_first_100_comparison.csv"),
        max_rows=100,
    )
    print(f"Test comparison table saved to {output_dir}/test_first_100_comparison.csv")




def train_baseline(epochs, lr, hidden_channels, output_dir):
    data = load_data()
    set_seed(42)
    model, metrics = run_training(
        data, epochs=epochs, lr=lr, hidden_channels=hidden_channels, class_weights=None
    )
    plot_training_curves(
        metrics,
        save_path=os.path.join(output_dir, "baseline_curves.png"),
        title_prefix="Baseline",
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "baseline_model.pth"))
    print(f"Baseline curves and model saved under {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCN and compare loss functions.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", choices=["baseline", "loss_compare"], default="loss_compare")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        for key in ["mode", "epochs", "lr", "hidden_channels", "output_dir"]:
            if key in config:
                setattr(args, key, config[key])

    if args.mode == "baseline":
        train_baseline(args.epochs, args.lr, args.hidden_channels, args.output_dir)
    else:
        train_loss_comparison(args.epochs, args.lr, args.hidden_channels, args.output_dir)
