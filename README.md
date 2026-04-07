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
