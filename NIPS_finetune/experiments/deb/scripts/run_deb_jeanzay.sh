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
# DEB (Disease Evidence Bottleneck) Training — Jean Zay version
#
# Usage:
#   sbatch experiments/deb/scripts/run_deb_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
################################################################################

set -e

# ── Environment setup ────────────────────────────────────────────────────────
module purge
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/.local_pt260
export WANDB_MODE=offline

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SEED=${3:-3407}

echo "=============================================="
echo "  DEB Training: ${MODEL} on ${DATASET}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

python experiments/deb/scripts/train.py \
    --mode deb \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --finetune frozen \
    --epochs 100 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --patience 15 \
    --clip_value 5.0 \
    --split_strategy subject \
    --deb_latent_dim 64 \
    --deb_gate_hidden 64 \
    --deb_fusion concat \
    --beta 1e-4 \
    --beta_warmup_epochs 5 \
    --sparse_lambda 1e-3 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "checkpoints_deb" \
    --num_workers 4 \
    --wandb_project eeg_deb \
    --wandb_run_name "DEB_${MODEL}_${DATASET}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
