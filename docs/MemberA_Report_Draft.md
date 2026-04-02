# SonarGNN:


## 1. Introduction
With the rapid proliferation of cryptocurrencies, detecting illicit activities within financial networks has become a critical security challenge. Traditional machine learning approaches often fail to capture the complex, interdependent nature of transaction flows. Recently, Graph Neural Networks (GNNs) have emerged as the state-of-the-art technique for such tasks, as they can effectively model both node features and topological structures [1]. This project, SonarGNN, aims to develop a robust GNN-based system for anti-money laundering detection. My primary responsibility is to establish the foundational data pipeline, conduct background exploration, and define the core problem.

## 2. Dataset Description and Analysis
The project utilizes the Elliptic dataset, a large-scale graph representing Bitcoin transactions. To construct the data object, I developed `dataset.py` leveraging the PyTorch Geometric (PyG) framework. The methodology involved parsing raw CSV files to extract 203,769 nodes and 234,355 edges, processing 165-dimensional node features, and mapping categorical labels. Nodes were chronologically divided into training, validation, and test sets using time-step masks to simulate real-world financial forecasting.

A major challenge encountered during data exploration was the severe class imbalance. By implementing a visualization module to plot the class distribution, I quantitatively demonstrated the extreme disparity between "Licit" and "Illicit" nodes. Illicit transactions constitute a critically small minority, which inherently biases standard deep learning classifiers toward the majority class. Recognizing this pain point is essential, as it provides strong data support and directly justifies the team's subsequent architectural decisions, including the adoption of Weighted Cross-Entropy loss and sub-graph sampling techniques.

## 3. Conclusion
In summary, I successfully implemented the end-to-end data construction module, overcoming the complexities of large-scale graph processing. The generated PyG `Data` object and the distribution analysis fulfill the project's foundational requirements while providing critical data-driven evidence for our model design. This contribution successfully unblocks downstream tasks, seamlessly bridging data preparation with the team's baseline model training and comprehensive evaluation.

## References
[1] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.
