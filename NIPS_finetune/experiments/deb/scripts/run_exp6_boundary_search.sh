#!/bin/bash
################################################################################
# Experiment 6: True Partial FT Boundary Search
#
# Compares baseline vs selector across 4 fine-tuning regimes:
#   frozen, top1, top2, top4
#
# Usage:
#   bash experiments/deb/scripts/run_exp6_boundary_search.sh [MODEL] [DATASET] [SEED] [CUDA]
#
# Examples:
#   bash experiments/deb/scripts/run_exp6_boundary_search.sh codebrain TUAB 3407 0
#   bash experiments/deb/scripts/run_exp6_boundary_search.sh codebrain TUAB all 0  # 5 seeds
################################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SEED_ARG=${3:-3407}
CUDA=${4:-0}

EPOCHS=100
BATCH_SIZE=64
LR_HEAD=1e-3
LR_RATIO=10
PATIENCE=15
SAVE_DIR="checkpoints_selector"
WANDB_PROJECT="eeg_selector_exp6"

REGIMES=(frozen top1 top2 top4)
MODES=(baseline selector)

# Determine seeds
if [ "$SEED_ARG" = "all" ]; then
    SEEDS=(42 1234 2025 3407 7777)
else
    SEEDS=("$SEED_ARG")
fi

echo "=============================================="
echo "  Experiment 6: True Partial FT Boundary Search"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Regimes: ${REGIMES[*]}"
echo "  Modes: ${MODES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
    for REGIME in "${REGIMES[@]}"; do
        for MODE in "${MODES[@]}"; do
            TAG="Exp6_${MODE}_${MODEL}_${DATASET}_${REGIME}_s${SEED}"
            echo ""
            echo ">>> ${TAG}"

            python experiments/deb/scripts/train_partial_ft.py \
                --mode "$MODE" \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --regime "$REGIME" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr_head "$LR_HEAD" \
                --lr_ratio "$LR_RATIO" \
                --patience "$PATIENCE" \
                --seed "$SEED" \
                --cuda "$CUDA" \
                --save_dir "$SAVE_DIR" \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_run_name "$TAG" \
                --split_strategy subject \
                --eval_test_every_epoch
        done
    done
done

echo ""
echo "Experiment 6 complete."
