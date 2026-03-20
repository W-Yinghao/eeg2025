#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# DEB Baseline Training
#
# Usage:
#   sbatch experiments/deb/scripts/run_baseline.sh [MODEL] [DATASET] [HEAD_TYPE]
#   MODEL:     codebrain | cbramod | luna   (default: codebrain)
#   DATASET:   TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
#   HEAD_TYPE: pool | flatten  (default: pool)
################################################################################

set -e

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg2025

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
HEAD_TYPE=${3:-pool}

echo "=============================================="
echo "  DEB Baseline: ${MODEL} on ${DATASET} (head=${HEAD_TYPE})"
echo "=============================================="

python experiments/deb/scripts/train.py \
    --mode baseline \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --head_type "$HEAD_TYPE" \
    --finetune frozen \
    --epochs 100 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --patience 15 \
    --clip_value 5.0 \
    --split_strategy subject \
    --head_hidden 512 \
    --seed 3407 \
    --cuda 0 \
    --save_dir "checkpoints_deb" \
    --num_workers 0 \
    --wandb_project eeg_deb \
    --wandb_run_name "Baseline_${MODEL}_${DATASET}_${HEAD_TYPE}_$(date +%Y%m%d_%H%M%S)"
