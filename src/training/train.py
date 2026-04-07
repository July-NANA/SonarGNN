import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

# Add project root to path (for direct script execution)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model import GCN
from src.data.dataset import EllipticDataset
from src.utils.plot import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    return parser.parse_args()


def train_baseline():
    args = parse_args()

    # -------------------- 1. Load data --------------------
    print("Loading Elliptic dataset...")
    dataset = EllipticDataset(root='data/Elliptic')
    data = dataset[0]
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_node_features} features, {data.num_classes} classes")

    # -------------------- 2. Model initialization --------------------
    in_channels = data.num_node_features
    hidden_channels = 64      # Can be reduced if out-of-memory
    out_channels = data.num_classes
    model = GCN(in_channels, hidden_channels, out_channels, dropout=0.5)
    print(model)

    # -------------------- 3. Optimizer --------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -------------------- 4. Training loop --------------------
    epochs = 200
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Training accuracy
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        train_accs.append(train_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.nll_loss(val_out[data.val_mask], data.y[data.val_mask]).item()
            val_losses.append(val_loss)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            val_accs.append(val_acc)

        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # -------------------- 5. Save model --------------------
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path)

    model_path = os.path.join(args.save_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # -------------------- 6. Save training metrics --------------------
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, 'metrics.pkl')
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_baseline()