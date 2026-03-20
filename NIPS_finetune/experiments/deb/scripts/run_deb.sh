#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# DEB (Disease Evidence Bottleneck) Training
#
# Usage:
#   sbatch experiments/deb/scripts/run_deb.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUEV)
################################################################################

set -e

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg2025

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

echo "=============================================="
echo "  DEB Training: ${MODEL} on ${DATASET}"
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
    --seed 3407 \
    --cuda 0 \
    --save_dir "checkpoints_deb" \
    --num_workers 0 \
    --wandb_project eeg_deb \
    --wandb_run_name "DEB_${MODEL}_${DATASET}_$(date +%Y%m%d_%H%M%S)"
