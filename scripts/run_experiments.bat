@echo off
setlocal enabledelayedexpansion

REM Create results directory (ignore error if exists)
mkdir results\lr_experiments 2>nul

set learning_rates=0.1 0.01 0.001

for %%r in (%learning_rates%) do (
    echo Running experiment with learning rate: %%r
    python src/training/train.py --lr %%r --save_dir results/lr_experiments/lr_%%r --epochs 200
)

echo All lr experiments completed
endlocal