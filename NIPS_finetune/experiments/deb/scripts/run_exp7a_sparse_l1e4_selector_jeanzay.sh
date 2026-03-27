#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -A ifd@v100
#SBATCH -C v100-32g
#SBATCH --signal=B:USR1@180

################################################################################
# Experiment 7A — Frozen Selector + L1 Sparse (lambda=1e-4)
#
# Usage:
#   sbatch run_exp7a_sparse_l1e4_selector_jeanzay.sh [MODEL] [DATASET] [SEED]
################################################################################

set -uo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/.local_pt260
export WANDB_MODE=offline

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SEED=${3:-3407}

MODE="selector"
REGIME="frozen"
SPARSE_LAMBDA=1e-4

SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/exp7a_sparse_l1e4_selector"
LOG_DIR="${PROJECT_DIR}/deb_log/exp7a_sparse_l1e4_selector"
SCRIPT_PATH="${PROJECT_DIR}/experiments/deb/scripts/run_exp7a_sparse_l1e4_selector_jeanzay.sh"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TAG="Exp7A_sparse_l1e4_${MODE}_${MODEL}_${DATASET}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 7A: Frozen Selector + L1 Sparse"
echo "  lambda_sparse=${SPARSE_LAMBDA}"
echo "  ${MODE} | ${MODEL} | ${DATASET} | regime=${REGIME} | seed=${SEED}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

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
    --enable_sparse \
    --sparse_lambda "$SPARSE_LAMBDA" \
    --sparse_type l1 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "$SAVE_DIR" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp7a \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    --resume "$SAVE_DIR" \
    && EXIT_CODE=0 || EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 124 ]; then
    echo ""
    echo "[REQUEUE] Training preempted. Resubmitting..."
    NEW_JOB=$(sbatch \
        -o "${LOG_DIR}/%j_selector_frozen_sparse_s${SEED}.out" \
        -e "${LOG_DIR}/%j_selector_frozen_sparse_s${SEED}.err" \
        --job-name="E7A_l1e4_s${SEED}" \
        "$SCRIPT_PATH" "$MODEL" "$DATASET" "$SEED" \
        | awk '{print $NF}')
    echo "[REQUEUE] Submitted continuation job: ${NEW_JOB}"
    exit 0
fi

exit $EXIT_CODE
