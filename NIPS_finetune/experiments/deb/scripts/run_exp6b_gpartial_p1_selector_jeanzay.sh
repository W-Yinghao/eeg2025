#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -A ifd@a100
#SBATCH -p gpu_p5
#SBATCH -C a100

################################################################################
# Exp 6B P1 — Gentle Partial Selector (lr_backbone=1e-5) — Jean Zay H100
#
# Regime: top1 (last 1 block unfrozen), freeze_patch_embed=True
# lr_head=1e-3, lr_backbone=1e-5, cosine scheduler, warmup=3, max_epochs=12
# early_stop_patience=4
#
# Usage:
#   sbatch run_exp6b_gpartial_p1_selector_jeanzay.sh [MODEL] [DATASET] [SEED]
#
# Defaults: MODEL=codebrain  DATASET=TUAB  SEED=3407
################################################################################

set -e

module purge
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/.local_pt260
export WANDB_MODE=offline

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SEED=${3:-3407}

SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/exp6b_gpartial_p1_selector"
mkdir -p "$SAVE_DIR"

TAG="Exp6B_P1_selector_${MODEL}_${DATASET}_top1_lrbb1e-5_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 6B P1: Gentle Partial Selector (lr_bb=1e-5)"
echo "  ${MODEL} | ${DATASET} | regime=top1 | seed=${SEED}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Save: ${SAVE_DIR}"
echo "=============================================="

python experiments/deb/scripts/train_partial_ft.py \
    --mode selector \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --regime top1 \
    --freeze_patch_embed \
    --epochs 12 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --lr_backbone 1e-5 \
    --scheduler cosine \
    --warmup_epochs 3 \
    --patience 4 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "$SAVE_DIR" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp6b \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    --amp
