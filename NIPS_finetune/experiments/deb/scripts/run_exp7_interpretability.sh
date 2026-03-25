#!/bin/bash
################################################################################
# Experiment 7: Selector Interpretability Enhancement
#
# Runs selector with sparse / consistency / both on the best regime.
# The regime should be determined from Exp 6 results.
#
# Usage:
#   bash experiments/deb/scripts/run_exp7_interpretability.sh [MODEL] [DATASET] [REGIME] [SEED] [CUDA]
#
# Examples:
#   # All 3 variants with best regime
#   bash experiments/deb/scripts/run_exp7_interpretability.sh codebrain TUAB top2 3407 0
#
#   # All seeds
#   bash experiments/deb/scripts/run_exp7_interpretability.sh codebrain TUAB top2 all 0
################################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
REGIME=${3:-top2}
SEED_ARG=${4:-3407}
CUDA=${5:-0}

EPOCHS=100
BATCH_SIZE=64
LR_HEAD=1e-3
LR_RATIO=10
PATIENCE=15
SAVE_DIR="checkpoints_selector"
WANDB_PROJECT="eeg_selector_exp7"

if [ "$SEED_ARG" = "all" ]; then
    SEEDS=(42 1234 2025 3407 7777)
else
    SEEDS=("$SEED_ARG")
fi

COMMON_ARGS=(
    --mode selector
    --dataset "$DATASET"
    --model "$MODEL"
    --regime "$REGIME"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr_head "$LR_HEAD"
    --lr_ratio "$LR_RATIO"
    --patience "$PATIENCE"
    --cuda "$CUDA"
    --save_dir "$SAVE_DIR"
    --split_strategy subject
    --eval_test_every_epoch
)

echo "=============================================="
echo "  Experiment 7: Selector Interpretability"
echo "  Model: ${MODEL}  Dataset: ${DATASET}  Regime: ${REGIME}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
    # ── Variant 1: Selector + Sparse ──
    TAG="Exp7_sparse_${MODEL}_${DATASET}_${REGIME}_s${SEED}"
    echo ""
    echo ">>> ${TAG}"
    python experiments/deb/scripts/train_partial_ft.py \
        "${COMMON_ARGS[@]}" \
        --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$TAG" \
        --enable_sparse \
        --sparse_lambda 1e-3 \
        --sparse_type l1

    # ── Variant 2: Selector + Consistency ──
    TAG="Exp7_consist_${MODEL}_${DATASET}_${REGIME}_s${SEED}"
    echo ""
    echo ">>> ${TAG}"
    python experiments/deb/scripts/train_partial_ft.py \
        "${COMMON_ARGS[@]}" \
        --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$TAG" \
        --enable_consistency \
        --consistency_lambda 1e-2 \
        --consistency_type l2

    # ── Variant 3: Selector + Sparse + Consistency ──
    TAG="Exp7_sp_cons_${MODEL}_${DATASET}_${REGIME}_s${SEED}"
    echo ""
    echo ">>> ${TAG}"
    python experiments/deb/scripts/train_partial_ft.py \
        "${COMMON_ARGS[@]}" \
        --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$TAG" \
        --enable_sparse \
        --sparse_lambda 1e-3 \
        --sparse_type l1 \
        --enable_consistency \
        --consistency_lambda 1e-2 \
        --consistency_type l2
done

echo ""
echo "Experiment 7 complete."
