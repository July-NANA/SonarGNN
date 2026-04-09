import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import sys
import os
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model import GCN
from src.data.dataset import EllipticDataset


def train_with_sampling(batch_size=1024, epochs=200, num_neighbors=[25, 10]):
    """
     Use NeighborLoader to train GCN with different batch sizes and neighbor sampling.
    """
    # -------------------- 1. Load data --------------------
    print(f"\n{'='*60}")
    print(f"实验配置: Batch Size = {batch_size}, num_neighbors = {num_neighbors}")
    print('='*60)
    
    print("加载数据...")
    dataset = EllipticDataset(root='data/raw')
    data = dataset[0]
    print(f"数据集: {data.num_nodes} 节点, {data.num_edges} 边")
    
    # -------------------- 2. Create NeighborLoader --------------------
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=0,
    )
    
    # -------------------- 3. Model --------------------
    model = GCN(data.num_node_features, 64, data.num_classes, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # -------------------- 4. Training --------------------
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # Calculate loss only on the nodes in the current batch (batch.batch_size gives the number of target nodes)
            loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pred = out[:batch.batch_size].argmax(dim=1)
            correct += (pred == batch.y[:batch.batch_size]).sum().item()
            total += batch.batch_size
        
        avg_loss = total_loss / num_batches
        train_acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        
        # Validation 
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.nll_loss(val_out[data.val_mask], data.y[data.val_mask]).item()
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
    
    # -------------------- 5. Results --------------------
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\nresults (batch_size={batch_size}):")
    print(f"  average time: {avg_epoch_time:.2f} 秒")
    print(f"  final validation accuracy: {val_accs[-1]:.4f}")
    print(f"  final validation loss: {val_losses[-1]:.4f}")
    
    # Save curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'Loss Curves (batch_size={batch_size})')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title(f'Accuracy Curves (batch_size={batch_size})')
    
    plt.tight_layout()
    plt.savefig(f'docs/training_curves_bs{batch_size}.png', dpi=150)
    print(f" curves saved: docs/training_curves_bs{batch_size}.png")
    
    return {
        'batch_size': batch_size,
        'avg_epoch_time': avg_epoch_time,
        'final_val_acc': val_accs[-1],
        'final_val_loss': val_losses[-1],
        'train_accs': train_accs,
        'val_accs': val_accs,
    }


def run_all_experiments():
    """运行所有 Batch Size 对比实验"""
    batch_sizes = [512, 1024, 2048, 4096]
    results = []
    
    for bs in batch_sizes:
        print(f"\n{'#'*60}")
        print(f"开始实验: Batch Size = {bs}")
        print('#'*60)
        
        result = train_with_sampling(batch_size=bs, epochs=100)  # 先用100轮快速测试
        results.append(result)
    
  
    print("\n" + "="*60)
    print("Summary of Results:")
    print("="*60)
    print(f"{'Batch Size':<12} {'Average Time (s)':<18} {'Final Val Acc':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['batch_size']:<12} {r['avg_epoch_time']:<18.2f} {r['final_val_acc']:<15.4f}")
    

def plot_comparison(results):

    batch_sizes = [r['batch_size'] for r in results]
    times = [r['avg_epoch_time'] for r in results]
    accs = [r['final_val_acc'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(batch_sizes, times, color='steelblue')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Average Time (s)')
    axes[0].set_title('Training Speed Comparison')
    

    axes[1].plot(batch_sizes, accs, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Model Performance Comparison')
    axes[1].set_ylim([0.85, 0.95])
    
    plt.tight_layout()
    plt.savefig('docs/batch_size_comparison.png', dpi=150)
    print("\n对比图已保存: docs/batch_size_comparison.png")
    #plt.show()
 
 #visualize_results(results)   
def generate_summary_chart():

    import matplotlib.pyplot as plt
    import matplotlib
    
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    

    batch_sizes = [512, 1024, 2048, 4096]
    times = [0.97, 0.51, 0.61, 0.65]           
    accuracies = [93.94, 93.04, 93.72, 93.17]  
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    

    bars = axes[0].bar(batch_sizes, times, color='#dbddef', edgecolor='black', width=200)
    axes[0].set_xlabel('Batch Size', fontsize=12)
    axes[0].set_ylabel('Average Time (s)', fontsize=12)
    axes[0].set_title('different Batch Size Training Speed Comparison', fontsize=14)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_xticks(batch_sizes)
    
    # present time values above the bars
    for bar, v in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.2f}s', 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Line Chart
    axes[1].plot(batch_sizes, accuracies, 'o-', color='#c1d8e9', linewidth=2, markersize=8,
                 markerfacecolor='#c1d8e9', markeredgecolor='black', markeredgewidth=1)
    axes[1].set_xlabel('Batch Size', fontsize=12)
    axes[1].set_ylabel('Accuracy Validation (%)', fontsize=12)
    axes[1].set_title('Different Batch Size performance Comparison', fontsize=14)
    axes[1].set_ylim(92.5, 94.5)
    axes[1].set_xticks(batch_sizes)
    axes[1].grid(True, linestyle='--', alpha=0.6, color='#d4d4d4')
    
    # present accuracy values above the points
    for x, y in zip(batch_sizes, accuracies):
        axes[1].text(x+50, y + 0.12, f'{y:.2f}%', ha='right', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/batch_size_experiment_summary.png', dpi=200, bbox_inches='tight')
    print("Summarychart saved: docs/batch_size_experiment_summary.png")
    plt.show()


if __name__ == "__main__":
    # single experiment 
    # train_with_sampling(batch_size=1024, epochs=100)

    # All experiments
    # run_all_experiments()
    
    # Generate summary chart with results from experiments
    generate_summary_chart()