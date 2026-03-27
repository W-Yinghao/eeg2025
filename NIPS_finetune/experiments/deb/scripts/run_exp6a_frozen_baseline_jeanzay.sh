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
# Experiment 6A — Frozen Baseline Re-run (head-only, auto-requeue)
#
# Purpose:
#   Complete the frozen baseline results that were cancelled in Exp6.
#   Uses SIGUSR1 + auto-requeue: SLURM sends USR1 3 min before timeout,
#   the trainer saves a resume checkpoint and exits with code 124.
#   This script then resubmits itself to continue training.
#
# Usage:
#   sbatch run_exp6a_frozen_baseline_jeanzay.sh [MODEL] [DATASET] [SEED]
#
# Examples:
#   sbatch run_exp6a_frozen_baseline_jeanzay.sh codebrain TUAB 3407
#   sbatch run_exp6a_frozen_baseline_jeanzay.sh codebrain TUAB 42
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

# Fixed config for Exp6A
MODE="baseline"
REGIME="frozen"

SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/exp6a_frozen_baseline"
LOG_DIR="${PROJECT_DIR}/deb_log/exp6a_frozen_baseline"
SCRIPT_PATH="${PROJECT_DIR}/experiments/deb/scripts/run_exp6a_frozen_baseline_jeanzay.sh"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TAG="Exp6A_${MODE}_${MODEL}_${DATASET}_${REGIME}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 6A: Frozen Baseline (auto-requeue)"
echo "  ${MODE} | ${MODEL} | ${DATASET} | regime=${REGIME} | seed=${SEED}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Save: ${SAVE_DIR}"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Run training — pass --resume to auto-detect resume checkpoint in save_dir
# Capture exit code without triggering pipefail on 124 (preempt)
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
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "$SAVE_DIR" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp6a \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    --resume "$SAVE_DIR" \
    && EXIT_CODE=0 || EXIT_CODE=$?

# Auto-requeue: if trainer exited with 124, it was preempted and saved a
# resume checkpoint. Resubmit this script with the same arguments.
if [ "$EXIT_CODE" -eq 124 ]; then
    echo ""
    echo "[REQUEUE] Training preempted. Resubmitting..."
    NEW_JOB=$(sbatch \
        -o "${LOG_DIR}/%j_baseline_frozen_s${SEED}.out" \
        -e "${LOG_DIR}/%j_baseline_frozen_s${SEED}.err" \
        --job-name="E6A_bl_frozen_s${SEED}" \
        "$SCRIPT_PATH" "$MODEL" "$DATASET" "$SEED" \
        | awk '{print $NF}')
    echo "[REQUEUE] Submitted continuation job: ${NEW_JOB}"
    exit 0
fi

exit $EXIT_CODE
