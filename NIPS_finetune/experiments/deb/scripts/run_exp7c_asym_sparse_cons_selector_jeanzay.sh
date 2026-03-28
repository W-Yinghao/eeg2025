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
# Experiment 7C — Asymmetric Sparse + Consistency (Jean Zay)
#
# Design:
#   - Base: Exp7B best (consistency λ=3e-3, L2, augmented views)
#   - Added: temporal-only L1 sparse (λ_t=3e-4, λ_f=0.0)
#   - Frequency gate receives NO sparse penalty
#
# Loss: L = L_ce + 3e-3 * L_consistency + 3e-4 * L_sparse_temporal
#
# Usage:
#   sbatch run_exp7c_asym_sparse_cons_selector_jeanzay.sh [MODEL] [DATASET] [SEED]
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

# Consistency (from Exp7B best)
CONS_LAMBDA=3e-3

# Branch-aware sparse: temporal only
SPARSE_LAMBDA_T=3e-4
SPARSE_LAMBDA_F=0.0

EXP_NAME="exp7c_asym_sparse_cons_selector"
SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/${EXP_NAME}"
LOG_DIR="${PROJECT_DIR}/deb_log/${EXP_NAME}"
SCRIPT_PATH="${PROJECT_DIR}/experiments/deb/scripts/run_exp7c_asym_sparse_cons_selector_jeanzay.sh"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TAG="Exp7C_asym_${MODE}_${MODEL}_${DATASET}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 7C: Asymmetric Sparse + Consistency"
echo "  lambda_cons=${CONS_LAMBDA}  lambda_sparse_t=${SPARSE_LAMBDA_T}  lambda_sparse_f=${SPARSE_LAMBDA_F}"
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
    --sparse_type l1 \
    --sparse_lambda_temporal "$SPARSE_LAMBDA_T" \
    --sparse_lambda_frequency "$SPARSE_LAMBDA_F" \
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
        -o "${LOG_DIR}/%j_selector_frozen_7c_asym_s${SEED}.out" \
        -e "${LOG_DIR}/%j_selector_frozen_7c_asym_s${SEED}.err" \
        --job-name="E7C_asym_s${SEED}" \
        "$SCRIPT_PATH" "$MODEL" "$DATASET" "$SEED" \
        | awk '{print $NF}')
    echo "[REQUEUE] Submitted continuation job: ${NEW_JOB}"
    exit 0
fi

exit $EXIT_CODE
