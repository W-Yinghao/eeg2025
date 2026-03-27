#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00
#SBATCH -A ifd@v100
#SBATCH --qos=qos_gpu-t4
#SBATCH -C v100-32g

################################################################################
# Exp 6B P3e2 — Staged Partial Selector (stage1=2 ep) — Jean Zay H100
#
# Stage 1: top1 (last 1 block unfrozen), lr_head=1e-3, lr_backbone=1e-5, 2 ep
# Stage 2: backbone frozen, lr_head=5e-4, 20 ep, warmup=2
#
# Usage:
#   sbatch run_exp6b_gpartial_p3e2_selector_jeanzay.sh [MODEL] [DATASET] [SEED]
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

SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/exp6b_gpartial_p3e2_selector"
mkdir -p "$SAVE_DIR"

TAG="Exp6B_P3e2_staged_selector_${MODEL}_${DATASET}_s1ep2_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 6B P3e2: Staged Partial Selector (stage1=2 ep)"
echo "  ${MODEL} | ${DATASET} | seed=${SEED}"
echo "  Stage1: top1, lr_bb=1e-5, 2 ep"
echo "  Stage2: frozen, lr_head=5e-4, 20 ep"
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
    --staged_partial \
    --stage1_epochs 2 \
    --stage1_lr_head 1e-3 \
    --stage1_lr_backbone 1e-5 \
    --stage1_warmup_epochs 1 \
    --stage2_epochs 20 \
    --stage2_lr_head 5e-4 \
    --stage2_warmup_epochs 2 \
    --stage2_patience 6 \
    --batch_size 64 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "$SAVE_DIR" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp6b \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    --amp
