#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180

################################################################################
# Experiment 7C-control — Light Sparse + Consistency (gpu-gw)
#
# Identical to 7C-main EXCEPT sparse_lambda = 3e-4 (vs 1e-3 in main).
# Purpose: test whether light sparse + consistency is already sufficient.
#
# Loss: L = L_ce + 3e-4 * L_sparse(temporal+frequency) + 3e-3 * L_consistency
#
# Usage:
#   sbatch run_exp7c_control_selector.sh [MODEL] [DATASET] [SEED]
################################################################################

set -uo pipefail

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate eeg2025

export WANDB_MODE=offline

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
SEED=${3:-3407}

MODE="selector"
REGIME="frozen"

# Joint regularization — light sparse
SPARSE_LAMBDA=3e-4
CONS_LAMBDA=3e-3

EXP_NAME="exp7c_control_selector"
SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/${EXP_NAME}"
LOG_DIR="${PROJECT_DIR}/deb_log/${EXP_NAME}"
SCRIPT_PATH="${PROJECT_DIR}/experiments/deb/scripts/run_exp7c_control_selector.sh"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TAG="Exp7C_ctrl_${MODE}_${MODEL}_${DATASET}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 7C-control: Light Sparse + Consistency"
echo "  lambda_sparse=${SPARSE_LAMBDA} (both gates)  lambda_cons=${CONS_LAMBDA}"
echo "  ${MODE} | ${MODEL} | ${DATASET} | regime=${REGIME} | seed=${SEED}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Save: ${SAVE_DIR}"
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
    --enable_consistency \
    --consistency_lambda "$CONS_LAMBDA" \
    --consistency_type l2 \
    --aug_jitter_std 0.03 \
    --aug_mask_ratio 0.08 \
    --aug_time_shift_max 1 \
    --aug_p_time_shift 0.5 \
    --aug_p_jitter 0.5 \
    --aug_p_mask 0.3 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "$SAVE_DIR" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp7c \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    --resume "$SAVE_DIR" \
    && EXIT_CODE=0 || EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 124 ]; then
    echo ""
    echo "[REQUEUE] Training preempted. Resubmitting..."
    NEW_JOB=$(sbatch \
        -o "${LOG_DIR}/%j_selector_frozen_7c_ctrl_s${SEED}.out" \
        -e "${LOG_DIR}/%j_selector_frozen_7c_ctrl_s${SEED}.err" \
        --job-name="E7C_ctrl_s${SEED}" \
        "$SCRIPT_PATH" "$MODEL" "$DATASET" "$SEED" \
        | awk '{print $NF}')
    echo "[REQUEUE] Submitted continuation job: ${NEW_JOB}"
    exit 0
fi

exit $EXIT_CODE
