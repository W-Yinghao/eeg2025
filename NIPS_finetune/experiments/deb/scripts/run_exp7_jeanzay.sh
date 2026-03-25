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
# Experiment 7 — single variant run — for Jean Zay SLURM
#
# Usage:
#   sbatch run_exp7_jeanzay.sh [MODEL] [DATASET] [REGIME] [VARIANT] [SEED]
#   VARIANT: sparse | consistency | both
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
REGIME=${3:-top2}
VARIANT=${4:-sparse}
SEED=${5:-3407}

TAG="Exp7_${VARIANT}_${MODEL}_${DATASET}_${REGIME}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 7: selector+${VARIANT} | ${MODEL} | ${DATASET} | ${REGIME} | seed=${SEED}"
echo "=============================================="

VARIANT_ARGS=()
if [ "$VARIANT" = "sparse" ] || [ "$VARIANT" = "both" ]; then
    VARIANT_ARGS+=(--enable_sparse --sparse_lambda 1e-3 --sparse_type l1)
fi
if [ "$VARIANT" = "consistency" ] || [ "$VARIANT" = "both" ]; then
    VARIANT_ARGS+=(--enable_consistency --consistency_lambda 1e-2 --consistency_type l2)
fi

python experiments/deb/scripts/train_partial_ft.py \
    --mode selector \
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
    --wandb_project eeg_selector_exp7 \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    "${VARIANT_ARGS[@]}"
