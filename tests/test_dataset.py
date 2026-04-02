import os
import torch
import pytest
from src.data.dataset import EllipticDataset, plot_class_distribution

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    # Create dummy data in a temporary directory
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Dummy features (txId, time_step, feature1, feature2)
    features_csv = raw_dir / "elliptic_txs_features.csv"
    features_csv.write_text("1,1,0.5,0.6\n2,1,0.1,0.2\n3,2,0.9,0.8\n4,2,0.3,0.4\n")

    # Dummy classes
    classes_csv = raw_dir / "elliptic_txs_classes.csv"
    classes_csv.write_text("txId,class\n1,1\n2,2\n3,unknown\n4,1\n")

    # Dummy edges
    edges_csv = raw_dir / "elliptic_txs_edgelist.csv"
    edges_csv.write_text("txId1,txId2\n1,2\n2,3\n3,4\n")

    return str(tmp_path)

def test_elliptic_dataset(dummy_dataset_dir):
    dataset = EllipticDataset(root=dummy_dataset_dir)
    
    assert len(dataset) == 1
    data = dataset[0]
    
    # We expect 4 nodes
    assert data.num_nodes == 4
    
    # We expect 3 edges
    assert data.num_edges == 3
    
    # Features
    assert data.num_node_features == 2
    
    # Labels: 1 -> 1, 2 -> 0, 3 -> -1, 4 -> 1
    expected_y = torch.tensor([1, 0, -1, 1], dtype=torch.long)
    assert torch.equal(data.y, expected_y)
    
    # Time steps: 1, 1, 2, 2
    expected_time_steps = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    assert torch.equal(data.time_steps, expected_time_steps)
    
    # Train/val/test masks logic check (in dummy dataset all are <=34, so train)
    # Valid labels are for nodes 0, 1, 3 (txIds 1, 2, 4)
    expected_train_mask = torch.tensor([True, True, False, True], dtype=torch.bool)
    assert torch.equal(data.train_mask, expected_train_mask)

def test_plot_class_distribution(tmp_path):
    y = torch.tensor([1, 0, 1, 1, 0, -1, -1])
    save_path = tmp_path / "plot.png"
    plot_class_distribution(y, str(save_path))
    
    assert os.path.exists(str(save_path))
