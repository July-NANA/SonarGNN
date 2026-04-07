import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import matplotlib.pyplot as plt

class EllipticDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root: directory where the dataset should be stored. This folder will contain
              `raw` and `processed` directories.
        """
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['elliptic_txs_features.csv', 'elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Data is assumed to be manually placed in self.raw_dir
        pass

    def process(self):
        print("Processing features...")
        features_path = os.path.join(self.raw_dir, 'elliptic_txs_features.csv')
        classes_path = os.path.join(self.raw_dir, 'elliptic_txs_classes.csv')
        edges_path = os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv')

        # Load features
        df_features = pd.read_csv(features_path, header=None)
        
        # Load classes
        print("Processing classes...")
        df_classes = pd.read_csv(classes_path)
        
        # Merge to ensure consistent ordering
        # Features: col 0 is txId, col 1 is time step
        tx_ids = df_features[0].values
        
        # Map txId to contiguous index
        tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
        
        # Process node features
        # Columns 2 to end are features
        x = torch.tensor(df_features.iloc[:, 2:].values, dtype=torch.float)
        
        # Process time steps
        time_steps = torch.tensor(df_features[1].values, dtype=torch.long)

        # Process labels
        # Class "1" -> Illicit, "2" -> Licit, "unknown" -> Unknown
        # We map: Illicit -> 1, Licit -> 0, Unknown -> -1
        class_mapping = {'1': 1, '2': 0, 'unknown': -1}
        
        # Create a label array matching the order of tx_ids
        df_classes_indexed = df_classes.set_index('txId')
        labels = df_classes_indexed.loc[tx_ids, 'class'].astype(str).map(class_mapping).fillna(-1).values
        y = torch.tensor(labels, dtype=torch.long)
        
        # Train/Val/Test masks based on time step
        # Time steps 1-34 are train, 35-42 are val, 43-49 are test
        train_mask = (time_steps <= 34) & (y != -1)
        val_mask = (time_steps > 34) & (time_steps <= 42) & (y != -1)
        test_mask = (time_steps > 42) & (y != -1)

        # Process edges
        print("Processing edges...")
        df_edges = pd.read_csv(edges_path)
        
        # Only keep edges where both nodes are in the feature set
        valid_edges = df_edges[df_edges['txId1'].isin(tx_id_to_idx) & df_edges['txId2'].isin(tx_id_to_idx)]
        
        # Map edge IDs to indices
        src = valid_edges['txId1'].map(tx_id_to_idx).values
        dst = valid_edges['txId2'].map(tx_id_to_idx).values
        
        edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.time_steps = time_steps
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.num_classes = 2

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print("Data processing completed.")

def plot_class_distribution(y_tensor, save_path="data/class_distribution.png"):
    """
    Plots a pie chart for the class distribution (Licit vs Illicit) and saves it.
    """
    y_numpy = y_tensor.numpy()
    
    illicit_count = np.sum(y_numpy == 1)
    licit_count = np.sum(y_numpy == 0)
    
    labels = ['Licit', 'Illicit']
    sizes = [licit_count, illicit_count]
    colors = ['#66b3ff', '#ff9999']
    explode = (0, 0.1)  # explode the illicit slice
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Distribution of Licit vs Illicit Nodes')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Pie chart saved to {save_path}")

if __name__ == '__main__':
    # Load dataset
    dataset = EllipticDataset(root='data/Elliptic')
    data = dataset[0]
    
    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    
    # Plot distribution
    plot_class_distribution(data.y, save_path="docs/class_distribution.png")
