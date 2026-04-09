import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path (for direct script execution)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model import GCN
from src.data.dataset import EllipticDataset

def train_baseline():
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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

    # -------------------- 5. Plot and save curves --------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('baseline_curves.png', dpi=150)
    print("Baseline curves saved as 'baseline_curves.png'")
    plt.show()

    # -------------------- 6. Save model (optional) --------------------
    torch.save(model.state_dict(), 'baseline_model.pth')
    print("Baseline model saved as 'baseline_model.pth'")

if __name__ == "__main__":
    train_baseline()

#added a test line to check if the script runs without errors. The line will be removed after confirming functionality.

def run_experiment():
    """运行 Batch Size 对比实验的入口"""
    import time
    
    batch_sizes = [512, 1024, 2048]
    
    for bs in batch_sizes:
        print(f"\n{'='*50}")
        print(f"开始实验: Batch Size = {bs}")
        print('='*50)
        
        start = time.time()
   
        # train_with_sampling(batch_size=bs, epochs=50)
        end = time.time()
        
        print(f"耗时: {(end-start)/60:.2f} 分钟")

if __name__ == "__main__":

    train_baseline()
    

    # run_experiment()