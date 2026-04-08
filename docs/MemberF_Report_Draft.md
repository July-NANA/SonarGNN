
# Member F Report Draft: Comprehensive Evaluation, Conclusion, and Future Work

## 1. Role and Objective
As Member F, my responsibility is to integrate experimental evidence from all teammates, evaluate the model with imbalance-sensitive metrics, summarize the final performance, and present the project limitations and future directions. In this project, plain accuracy is not sufficient because the Elliptic dataset is severely imbalanced: a model can achieve high overall accuracy while still failing to detect illicit transactions. Therefore, the evaluation layer must include **F1-score, recall, precision, PR-AUC, ROC-AUC, balanced accuracy, and confusion-matrix counts**.

To support this role, I implemented an evaluation module (`src/evaluation/evaluate.py` and `src/evaluation/metrics.py`) that can read prediction files and automatically generate metric tables, JSON/CSV summaries, and visual comparison charts.

## 2. Comprehensive Evaluation Framework
The final evaluation framework uses the following indicators:

- **Accuracy**: useful as a global summary, but insufficient on its own.
- **Precision (Illicit class)**: among predicted illicit nodes, how many are truly illicit.
- **Recall (Illicit class)**: among all true illicit nodes, how many are successfully detected.
- **F1-score**: harmonic mean of precision and recall; important under class imbalance.
- **Specificity**: ability to correctly identify licit transactions.
- **Balanced Accuracy**: average of recall and specificity.
- **ROC-AUC**: threshold-free separability between classes.
- **PR-AUC**: especially important for rare positive classes because it focuses on precision-recall trade-offs.

This metric bundle is more aligned with the anti-money-laundering objective than headline accuracy alone.

## 3. Current Integrated Findings from Teammates

### 3.1 Baseline Model (Member B)
From the baseline training curves, the model converges stably and achieves strong overall validation accuracy (approximately **93%+**) after sufficient epochs. However, the growing train-validation gap suggests mild overfitting near the end of training. This baseline is therefore a solid reference point, but it should not be treated as the final answer because it was mainly optimized for general classification performance rather than minority-class recovery.

### 3.2 Weighted Cross-Entropy (Member C)
Member C provided the most important imbalance-oriented evidence. The test-set summary is shown below:

| Metric | CE | Weighted CE |
| --- | ---: | ---: |
| Accuracy | 0.9723 | 0.9297 |
| Macro F1 | 0.4930 | 0.4860 |
| Illicit Precision | 0.0000 | 0.0066 |
| Illicit Recall | 0.0000 | 0.0118 |
| Illicit F1 | 0.0000 | 0.0084 |

Interpretation:
- Standard CE achieves the best **headline accuracy**, but it completely misses the illicit class on the reported test set.
- Weighted CE lowers overall accuracy, yet it begins to recover a small number of illicit cases.
- For an AML task, this direction is important: a model that never detects illicit nodes is operationally weak, even if its accuracy looks high.

### 3.3 Learning Rate Tuning (Member D)
Member D’s experiments show that **LR = 0.01** is the most balanced setting:
- **LR = 0.1** converges but shows unstable early oscillation.
- **LR = 0.001** is stable but too slow within the fixed epoch budget.
- **LR = 0.01** offers the best compromise between stability, convergence speed, and final validation performance.

Therefore, the integrated recommendation from the optimization perspective is to keep **0.01 as the default learning rate** for subsequent experiments.

### 3.4 Batch Size / Neighbor Sampling (Member E)
From Member E’s summary chart, the batch-size results are:

| Batch Size | Validation Accuracy | Avg. Seconds / Batch |
| --- | ---: | ---: |
| 512 | 93.94% | 0.97 s |
| 1024 | 93.04% | 0.51 s |
| 2048 | 93.72% | 0.61 s |
| 4096 | 93.17% | 0.65 s |

Interpretation:
- **Batch size 512** achieves the best validation accuracy.
- **Batch size 1024** is the fastest option.
- **Batch size 2048** is a reasonable trade-off, with accuracy close to the best result and better throughput than 512.
- **Batch size 4096** does not provide a speed advantage large enough to justify its lower accuracy.

From a performance-first perspective, **512** is preferable. From a throughput-first perspective, **1024** is preferable. From a balanced deployment perspective, **2048** is a defensible compromise.

## 4. Integrated Conclusion
The current project evidence leads to four key conclusions.

First, the SonarGNN pipeline is technically valid: the team successfully built the graph data pipeline, trained a GCN baseline, and completed targeted experiments on imbalance handling, learning rate selection, and mini-batch training.

Second, the model achieves **strong overall validation accuracy**, but this alone is not enough for an anti-money-laundering problem. The weighted-loss experiment demonstrates that the main bottleneck is still **minority-class detection**.

Third, among the tested optimization settings, **learning rate 0.01** remains the most reliable default choice, while **batch size 512 / 2048** provides the best performance trade-offs depending on whether accuracy or efficiency is prioritized.

Fourth, from a product and risk perspective, the project is promising but **not yet production-ready**. The current results show that the team can build an effective graph-based detector, but the illicit-class recall is still too low to support real-world screening without further threshold tuning, sampling strategies, or model enhancement.

## 5. Future Work
Several directions can directly improve the project:

1. **Threshold tuning instead of argmax-only evaluation**  
   The current classification decisions are likely based on a fixed argmax rule. For rare-event detection, threshold tuning on validation PR curves can often improve recall at acceptable precision cost.

2. **Better imbalance strategies**  
   Weighted CE is a useful first step, but it is not sufficient. Future work should explore focal loss, class-balanced loss, positive-class oversampling, or hard-negative mining.

3. **Neighbor-sampling + imbalance learning integration**  
   A particularly promising next experiment is to combine Member C’s weighted-loss setting with Member E’s mini-batch sampling pipeline, then compare both speed and illicit recall.

4. **Early stopping and model checkpoint selection**  
   Several curves show overfitting behavior. Early stopping on validation F1 or PR-AUC would likely produce a better checkpoint than simply using the final epoch.

5. **Richer evaluation protocol**  
   The final report should include confusion matrices, PR curves, ROC curves, and per-class metrics rather than only loss/accuracy curves.

## 6. QA / Integration Notes
As PM and QA, I also identified several integration points that the team should standardize before final submission:

- All members should define metrics consistently: for example, whether validation accuracy means **best epoch**, **last epoch**, or **average across epochs**.
- All result files should use a common export format such as `y_true`, `y_pred`, and optional `y_score` so that the evaluation script can aggregate them automatically.
- The final model selection criterion should be declared explicitly. For this project, selecting by **validation PR-AUC or illicit F1** is more defensible than selecting by raw accuracy.

## 7. Deliverables Prepared by Member F
- `src/evaluation/metrics.py`: imbalance-aware metric computation.
- `src/evaluation/evaluate.py`: CLI evaluation script.
- `src/evaluation/report_figures.py`: figure generation from current teammate summaries.
- `tests/test_evaluation.py`: unit tests for the evaluation module.
- `docs/generated/memberF_ce_vs_wce_bar.png`: current CE vs Weighted-CE summary figure.
- `docs/generated/memberF_batch_tradeoff_scatter.png`: batch-size trade-off figure.
