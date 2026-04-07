#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/lr_experiments

learning_rates=(0.1 0.01 0.001)

for lr in "${learning_rates[@]}"; do
  echo "Running experiment with learning rate: ${lr}"
  python src/training/train.py --lr "${lr}" --save_dir "results/lr_experiments/lr_${lr}" --epochs 200
done

echo "All lr experiments completed"
