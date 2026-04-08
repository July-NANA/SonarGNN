"""Export SonarGNN predictions for downstream evaluation.

This script bridges the gap between training outputs and the evaluation CLI.
It loads a trained checkpoint, runs inference on a chosen split, and exports
sample-level predictions to both CSV and NPZ formats.

Typical usage
-------------
1) Export test-set predictions from a trained checkpoint:
   python scripts/export_predictions.py \
       --checkpoint results/checkpoints/baseline_model.pth \
       --dataset-root data/Elliptic \
       --split test \
       --output-dir results/predictions

2) Then evaluate them:
   python -m src.evaluation.evaluate \
       --input results/predictions/predictions_test.csv \
       --output-dir results/eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Support both `python scripts/export_predictions.py` and module-style execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_runtime_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "export_predictions.py requires torch and torch_geometric. "
            "Please install the training/runtime dependencies first."
        ) from exc


def load_dataset(dataset_root: str):
    from src.data.dataset import EllipticDataset

    dataset = EllipticDataset(root=dataset_root)
    return dataset[0]


def get_model_class():
    import torch
    import torch.nn.functional as F

    try:
        from src.model.model import GCN  # type: ignore
        return GCN
    except Exception:  # pragma: no cover - fallback for branches without src/model/model.py
        from torch_geometric.nn import GCNConv

        class GCN(torch.nn.Module):
            """Fallback two-layer GCN used when the project model module is absent."""

            def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)

        return GCN


def get_split_mask(data, split: str):
    split = split.lower()
    if split == "train":
        return data.train_mask
    if split == "val":
        return data.val_mask
    if split == "test":
        return data.test_mask
    if split == "all":
        return data.y != -1
    raise ValueError("split must be one of: train, val, test, all")


def infer_hidden_channels(state_dict: dict, default: int = 64) -> int:
    for key in ("conv1.bias", "conv1.lin.weight", "conv1.weight"):
        tensor = state_dict.get(key)
        if tensor is None:
            continue
        if hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            return int(tensor.shape[0])
    return default


def build_model(data, checkpoint: Path, hidden_channels: int | None = None, dropout: float = 0.5):
    import torch

    GCN = get_model_class()
    raw = torch.load(checkpoint, map_location="cpu")
    state_dict = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    inferred_hidden = hidden_channels or infer_hidden_channels(state_dict)
    model = GCN(
        in_channels=int(data.num_node_features),
        hidden_channels=int(inferred_hidden),
        out_channels=int(getattr(data, "num_classes", 2)),
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_inference(model, data, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    mask = get_split_mask(data, split)
    node_indices = torch.where(mask)[0]

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        if out.dim() != 2:
            raise ValueError("Model output must be a 2D tensor of shape [num_nodes, num_classes]")

        probs_exp = torch.exp(out)
        if torch.allclose(probs_exp.sum(dim=1).mean(), torch.tensor(1.0), atol=1e-3):
            probs = probs_exp
        else:
            probs = torch.softmax(out, dim=1)

        y_true = data.y[node_indices].cpu().numpy()
        y_score = probs[node_indices, 1].cpu().numpy()
        y_pred = probs[node_indices].argmax(dim=1).cpu().numpy()
        node_indices_np = node_indices.cpu().numpy()

    return node_indices_np, y_true, y_pred, y_score


def save_csv(output_path: Path, node_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "y_true", "y_pred", "y_score"])
        for row in zip(node_indices, y_true, y_pred, y_score):
            writer.writerow(row)


def save_npz(output_path: Path, node_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> None:
    np.savez(
        output_path,
        node_index=node_indices,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )


def save_metadata(output_path: Path, args: argparse.Namespace, data, count: int) -> None:
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_root": args.dataset_root,
        "split": args.split,
        "num_exported_samples": int(count),
        "num_total_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_node_features": int(data.num_node_features),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SonarGNN predictions to CSV and NPZ for evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint (.pth)")
    parser.add_argument("--dataset-root", default="data/Elliptic", help="Root directory passed to EllipticDataset")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output-dir", default="results/predictions", help="Directory to save exported predictions")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Optional manual hidden size override")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout used when rebuilding the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset_root)
    model = build_model(data, Path(args.checkpoint), hidden_channels=args.hidden_channels, dropout=args.dropout)
    node_indices, y_true, y_pred, y_score = run_inference(model, data, args.split)

    csv_path = output_dir / f"predictions_{args.split}.csv"
    npz_path = output_dir / f"predictions_{args.split}.npz"
    meta_path = output_dir / f"predictions_{args.split}_meta.json"

    save_csv(csv_path, node_indices, y_true, y_pred, y_score)
    save_npz(npz_path, node_indices, y_true, y_pred, y_score)
    save_metadata(meta_path, args, data, len(y_true))

    print(f"Saved CSV predictions to: {csv_path}")
    print(f"Saved NPZ predictions to: {npz_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Exported {len(y_true)} samples from split='{args.split}'.")


if __name__ == "__main__":
    main()
