#!/bin/bash
################################################################################
# Experiment 6A: Frozen Baseline Re-run — Local Sequential Runner
#
# Runs frozen baseline across all seeds locally (no SLURM).
#
# Usage:
#   bash experiments/deb/scripts/run_exp6a_frozen_baseline_local.sh [MODEL] [DATASET] [SEED] [CUDA]
#
# Examples:
#   bash experiments/deb/scripts/run_exp6a_frozen_baseline_local.sh codebrain TUAB 3407 0
#   bash experiments/deb/scripts/run_exp6a_frozen_baseline_local.sh codebrain TUAB all 0
################################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SEED_ARG=${3:-3407}
CUDA=${4:-0}

SAVE_DIR="checkpoints_selector/exp6a_frozen_baseline"
WANDB_PROJECT="eeg_selector_exp6a"
MODE="baseline"
REGIME="frozen"

# Determine seeds
if [ "$SEED_ARG" = "all" ]; then
    SEEDS=(42 1234 2025 3407 7777)
else
    SEEDS=("$SEED_ARG")
fi

echo "=============================================="
echo "  Experiment 6A: Frozen Baseline Re-run (Local)"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Mode: ${MODE}  Regime: ${REGIME}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
    TAG="Exp6A_${MODE}_${MODEL}_${DATASET}_${REGIME}_s${SEED}"
    echo ""
    echo ">>> ${TAG}"

    python experiments/deb/scripts/train_partial_ft.py \
        --mode "$MODE" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --regime "$REGIME" \
        --epochs 50 \
        --batch_size 64 \
        --lr_head 1e-3 \
        --lr_backbone 0.0 \
        --lr_ratio 10 \
        --patience 12 \
        --warmup_epochs 3 \
        --scheduler cosine \
        --clip_value 1.0 \
        --seed "$SEED" \
        --cuda "$CUDA" \
        --save_dir "$SAVE_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$TAG" \
        --split_strategy subject \
        --eval_test_every_epoch \
        --resume "$SAVE_DIR"
done

echo ""
echo "Experiment 6A complete."
