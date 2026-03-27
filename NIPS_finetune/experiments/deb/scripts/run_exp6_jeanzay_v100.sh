#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -A ifd@v100
#SBATCH -C v100-32g

################################################################################
# Experiment 6 — single (regime, mode, seed) run — Jean Zay V100
#
# Usage:
#   sbatch run_exp6_jeanzay_v100.sh [MODEL] [DATASET] [REGIME] [MODE] [SEED]
#
# Examples:
#   sbatch run_exp6_jeanzay_v100.sh codebrain TUAB frozen selector 3407
#   sbatch run_exp6_jeanzay_v100.sh codebrain TUAB top2 baseline 42
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
REGIME=${3:-frozen}
MODE=${4:-selector}
SEED=${5:-3407}

TAG="Exp6_${MODE}_${MODEL}_${DATASET}_${REGIME}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 6: ${MODE} | ${MODEL} | ${DATASET} | regime=${REGIME} | seed=${SEED}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

python experiments/deb/scripts/train_partial_ft.py \
    --mode "$MODE" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --regime "$REGIME" \
    --epochs 100 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --lr_ratio 10 \
    --patience 15 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "checkpoints_selector" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp6 \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch
