# Member C Report Draft: Weighted Cross-Entropy (Imbalance Handling)

## 1. Experimental Goal
Evaluate whether Weighted Cross-Entropy (WCE) improves detection of the minority "Illicit" class compared to standard Cross-Entropy (CE), under the same GCN architecture and training settings.

## 2. Setup
Model: 2-layer GCN (hidden=64, dropout=0.5).  
Training: 200 epochs, Adam, lr=0.01.  
Data split: Elliptic dataset time-based masks (train/val/test).  
Loss variants:
1. CE (standard NLL loss)
2. WCE (class weights computed from training set inverse frequency)

Artifacts produced:
1. CE curves: `outputs/ce_curves.png`
2. WCE curves: `outputs/weighted_ce_curves.png`
3. CE vs WCE comparison: `outputs/ce_vs_weighted_comparison.png`
4. First 100 test samples comparison table: `outputs/test_first_100_comparison.csv`

## 3. Key Results (Full Test Set)
Test set size: 6,687 nodes  
Class distribution: 6,518 Licit, 169 Illicit

Metrics summary:

| Metric | CE | WCE |
| --- | --- | --- |
| Accuracy | 0.9723 | 0.9297 |
| Macro F1 | 0.4930 | 0.4860 |
| Illicit Precision | 0.0000 | 0.0066 |
| Illicit Recall | 0.0000 | 0.0118 |
| Illicit F1 | 0.0000 | 0.0084 |

## 4. Interpretation
1. CE achieves higher overall accuracy and macro F1, but completely fails to detect any Illicit nodes (precision/recall/F1 = 0 for class 1).
2. WCE lowers overall accuracy but starts to recover a small number of Illicit cases (non-zero precision and recall).
3. This confirms the expected tradeoff in imbalanced learning: WCE sacrifices majority-class performance to improve minority-class sensitivity.

## 5. Result Direction (Is It Positive?)
The result is **partially positive**:
1. Positive for the minority class: WCE improves Illicit recall from 0 to 0.0118.
2. Negative for overall accuracy and macro F1: both metrics slightly drop.

Conclusion: WCE shows early signs of recovering minority class detection, but the gains are still small and not yet strong enough to improve global metrics. Further tuning (class weights, thresholding, or sampling) is needed to make the improvement practically meaningful.

## 6. Note on the First-100 Test Sample Table
The first 100 test samples are all Licit. CE gets 100/100 correct, while WCE gets 97/100.  
This subset reflects the majority-class tradeoff and does not measure minority-class improvement.

## 7. Change Log (Member C)
1. Added Weighted CE support via class-weight computation and optional weighted NLL loss.
2. Integrated CE vs WCE comparison workflow with consistent plotting utilities.
3. Generated required artifacts:
   - CE/WCE training curves
   - CE vs WCE comparison plot
   - First-100 test sample comparison table

## 8. Project Structure (Relevant Modules)
1. `src/data/`  
   - `dataset.py`: Elliptic dataset loading, masks, and preprocessing.
2. `src/model/`  
   - `model.py`: GCN architecture definition (no training logic).
3. `src/training/`  
   - `loss.py`: Weighted CE utilities.  
   - `train.py`: Training loop and CE/WCE comparison pipeline.
4. `src/utils/`  
   - `plot.py`: Unified plotting utilities for consistent figures.
5. `configs/`  
   - `baseline.yaml`: Baseline settings.  
   - `experiment_loss.yaml`: CE vs WCE experiment settings.
6. `outputs/`  
   - Generated plots and CSV tables for reporting.
