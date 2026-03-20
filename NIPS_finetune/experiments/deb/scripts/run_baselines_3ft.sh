#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# 3 Baseline Fine-tune Strategies (plain head) under eeg_deb protocol
#
# Runs:
#   1. frozen   + plain head (pool)
#   2. partial  + plain head (pool)  — backbone LR = lr_head / lr_ratio
#   3. full     + plain head (pool)  — all params at lr_head
#
# Usage:
#   sbatch experiments/deb/scripts/run_baselines_3ft.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
#
# Or run a single strategy (with optional seed):
#   sbatch experiments/deb/scripts/run_baselines_3ft.sh codebrain TUAB frozen 42
################################################################################

set -e

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg2025

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SINGLE_FT=${3:-all}   # frozen | partial | full | all
SEED_ARG=${4:-3407}    # seed (default: 3407)

# ── Shared hyperparameters ────────────────────────────────────────────────────
SEED=$SEED_ARG
EPOCHS=100
BATCH_SIZE=64
LR_HEAD=1e-3
LR_RATIO=10           # partial FT: backbone_lr = lr_head / lr_ratio = 1e-4
PATIENCE=15
CLIP=5.0
HEAD_TYPE=pool
HEAD_HIDDEN=512
SPLIT=subject
SAVE_DIR="checkpoints_deb"
WANDB_PROJECT="eeg_deb"
NUM_WORKERS=0

# ── Common arguments ─────────────────────────────────────────────────────────
COMMON_ARGS=(
    --mode baseline
    --dataset "$DATASET"
    --model "$MODEL"
    --head_type "$HEAD_TYPE"
    --head_hidden "$HEAD_HIDDEN"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr_head "$LR_HEAD"
    --patience "$PATIENCE"
    --clip_value "$CLIP"
    --split_strategy "$SPLIT"
    --seed "$SEED"
    --cuda 0
    --save_dir "$SAVE_DIR"
    --num_workers "$NUM_WORKERS"
    --wandb_project "$WANDB_PROJECT"
)

# ── Helper ────────────────────────────────────────────────────────────────────
run_experiment() {
    local FT_MODE=$1
    local TAG="Baseline_${MODEL}_${DATASET}_${FT_MODE}_pool_s${SEED}_$(date +%Y%m%d_%H%M%S)"

    echo ""
    echo "=============================================="
    echo "  Baseline: ${MODEL} | ${DATASET} | finetune=${FT_MODE} | head=${HEAD_TYPE}"
    echo "=============================================="

    local EXTRA_ARGS=()
    if [ "$FT_MODE" = "partial" ]; then
        # partial: unfreeze backbone with lower LR (lr_head / lr_ratio)
        EXTRA_ARGS+=(--lr_ratio "$LR_RATIO")
    fi

    python experiments/deb/scripts/train.py \
        "${COMMON_ARGS[@]}" \
        --finetune "$FT_MODE" \
        --wandb_run_name "$TAG" \
        "${EXTRA_ARGS[@]}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
if [ "$SINGLE_FT" = "all" ]; then
    run_experiment frozen
    run_experiment partial
    run_experiment full
else
    run_experiment "$SINGLE_FT"
fi

echo ""
echo "All done."
