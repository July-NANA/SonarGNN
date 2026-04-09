# SonarGNN: Web3 Anti-Money Laundering Radar 🕸️🔍

![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=for-the-badge)

> A Graph Convolutional Network (GCN) designed to detect illicit transactions (e.g., money laundering, wash trading) within Web3 cryptocurrency networks.
> 
> Developed as part of the **CDS 525 Group Project (Deep Learning)**.

## 📌 Project Overview
In decentralized financial networks, malicious actors often hide illicit activities within complex webs of transactions. **SonarGNN** leverages the topological structure of the network. Instead of analyzing transactions in isolation, it uses a Graph Neural Network (GNN) to aggregate features from a transaction's "neighborhood," effectively acting as a radar to identify high-risk nodes in a highly imbalanced environment.

## 📊 Dataset Context
The project utilizes the **Elliptic Data Set**, a graph-structured dataset of Bitcoin transactions.

* **Source:** [Elliptic Data Set on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
* **Total Transactions (Nodes):** > 200,000 
* **Transaction Flows (Edges):** > 230,000
* **Node Features:** 165 dimensions (anonymized temporal and structural metrics)
* **Classes:** Illicit (High Risk) vs. Licit (Normal) 

*(Note: The dataset exhibits a significant class imbalance, which is a primary challenge addressed in the model's design).*

## ⚙️ Core Architecture & Pipeline
The project is structured around a complete deep learning pipeline:

1.  **Data Ingestion & Graph Construction:** Automated retrieval of raw transaction data and mapping into a structured graph topology.
2.  **GCN Model Design:** A lightweight Graph Convolutional Network optimized for classifying node anomalies.
3.  **Imbalance Handling:** Implementation of specialized loss functions (e.g., Weighted Cross-Entropy) to prevent the model from ignoring the minority "illicit" class.

## 📈 Evaluation & Experiments (CDS 525 Scope)
To fulfill the core requirements of the CDS 525 curriculum, this project conducts extensive hyperparameter tuning and generates performance visualizations:

* **Training Dynamics:** Tracking Training vs. Test Loss and Accuracy curves across multiple Epochs.
* **Loss Function Ablation:** Comparing standard Cross-Entropy against Weighted approaches to handle data imbalance.
* **Learning Rate Optimization:** Evaluating model convergence scaling across multiple learning rate configurations (e.g., 0.1, 0.01, 0.001).
* **Batch Size & Subgraph Sampling:** Comparing the effects of different batch sizes on training stability and speed.

---
*Built with ❤️ by the CDS 525 Group.*


## 🚀 Quick Start: Train → Export Predictions → Evaluate

The evaluation script is a **post-processing tool**. It does **not** train the model by itself.
The expected workflow is:

1. Prepare the Elliptic dataset under `data/Elliptic/raw/`
2. Train or load a model checkpoint (`.pth`)
3. Export sample-level predictions to `predictions.csv` / `predictions.npz`
4. Run the evaluation CLI to compute metrics and generate figures

### 1) Required raw files
Place the following files in `data/Elliptic/raw/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

### 2) Training output expected by member F's evaluation
`src/evaluation/evaluate.py` expects a predictions file generated **after inference on a train/val/test split**.
The exported file should contain at least:

- `y_true`: ground-truth label
- `y_pred`: predicted class label
- `y_score` (recommended): model score or probability for the positive / illicit class

Supported formats:

#### CSV format
```csv
node_index,y_true,y_pred,y_score
1001,1,1,0.95
1002,0,0,0.08
1003,1,0,0.41
```

#### NPZ format
The `.npz` file should contain arrays named:

- `node_index` (optional)
- `y_true`
- `y_pred`
- `y_score` (recommended)

### 3) Export predictions automatically from a trained checkpoint
Use the new helper script below to bridge training outputs and evaluation:

```bash
python scripts/export_predictions.py   --checkpoint results/checkpoints/baseline_model.pth   --dataset-root data/Elliptic   --split test   --output-dir results/predictions
```

Generated files:

- `results/predictions/predictions_test.csv`
- `results/predictions/predictions_test.npz`
- `results/predictions/predictions_test_meta.json`

This script rebuilds the model, runs inference on the selected split, and exports
sample-level predictions that can be consumed directly by `evaluate.py`.

### 4) Run evaluation
#### Module-style execution (recommended)
```bash
python -m src.evaluation.evaluate   --input results/predictions/predictions_test.csv   --output-dir results/eval
```

#### Direct-script execution (also supported)
```bash
python src/evaluation/evaluate.py   --input results/predictions/predictions_test.csv   --output-dir results/eval
```

### 5) Optional summary charts for report / PPT
If you already have experiment-level metrics from members B/C/D/E, you can generate grouped bar and radar charts with:

```bash
python -m src.evaluation.evaluate   --summary-json docs/summary/memberF_current_summary.json   --output-dir docs/generated
```

## 🧭 What a new teammate needs to run the project

If someone new clones the repository and wants to reproduce the evaluation pipeline, the minimum steps are:

1. Install project dependencies
2. Put the Elliptic raw CSV files into `data/Elliptic/raw/`
3. Obtain a trained checkpoint (`.pth`) from baseline / weighted-loss / LR / batch experiments
4. Run `scripts/export_predictions.py` to generate `predictions.csv` or `predictions.npz`
5. Run `src/evaluation/evaluate.py` to compute F1, PR-AUC, ROC-AUC, confusion-matrix-based metrics, and report-ready plots

## 📝 Notes on current repository status

- `evaluate.py` is the unified evaluation entrypoint for member F's deliverables.
- `export_predictions.py` is the bridge from trained models to evaluation artifacts.
- If a branch does not yet include a full training pipeline, predictions can still be exported as long as a valid checkpoint and dataset are available.
- The recommended positive class is `1` (illicit), while label `-1` is treated as unknown and ignored during evaluation.
