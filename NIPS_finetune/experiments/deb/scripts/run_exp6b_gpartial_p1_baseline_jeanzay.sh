#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -A ifd@v100
#SBATCH -C v100-32g
#SBATCH --signal=B:USR1@120

################################################################################
# Exp 6B — Gentle Partial FT Baseline: P1
#
# Config:
#   regime=top1_gentle (last 1 block unfrozen, patch_embed frozen)
#   lr_head=1e-3, lr_backbone=1e-5 (100x ratio)
#   cosine scheduler, warmup=3, epochs=12, patience=4
#   monitor: val_balanced_accuracy
#
# Supports: checkpoint + auto-requeue on SLURM preemption/timeout.
#
# Usage:
#   sbatch run_exp6b_gpartial_p1_baseline_jeanzay.sh [MODEL] [DATASET] [SEED]
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

MODE="baseline"
REGIME="top1_gentle"
EXP_TAG="exp6b_gpartial_p1_baseline"

SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/${EXP_TAG}"
LOG_DIR="${PROJECT_DIR}/deb_log/${EXP_TAG}"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TAG="Exp6B_P1_${MODE}_${MODEL}_${DATASET}_${REGIME}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 6B P1: Gentle Partial FT Baseline"
echo "  ${MODE} | ${MODEL} | ${DATASET} | regime=${REGIME} | seed=${SEED}"
echo "  lr_head=1e-3  lr_backbone=1e-5  warmup=3  epochs=12"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Save: ${SAVE_DIR}"
echo "=============================================="

# ── Signal forwarding for SLURM preemption ──
# SLURM sends SIGUSR1 to bash (--signal=B:USR1@120).
# We forward it to the Python child so it can checkpoint and exit(124).
TRAIN_PID=""
forward_signal() {
    if [ -n "$TRAIN_PID" ]; then
        echo "[SLURM] Forwarding SIGUSR1 to training process (PID=$TRAIN_PID)"
        kill -USR1 "$TRAIN_PID" 2>/dev/null
    fi
}
trap forward_signal USR1

# Run training in background (required for trap to work)
python experiments/deb/scripts/train_partial_ft.py \
    --mode "$MODE" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --regime "$REGIME" \
    --resume "$SAVE_DIR" \
    --epochs 12 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --lr_backbone 1e-5 \
    --warmup_epochs 3 \
    --scheduler cosine \
    --patience 4 \
    --clip_value 1.0 \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "$SAVE_DIR" \
    --num_workers 4 \
    --wandb_project eeg_selector_exp6b \
    --wandb_run_name "$TAG" \
    --split_strategy subject \
    --eval_test_every_epoch \
    &
TRAIN_PID=$!

# Wait for training (re-wait if interrupted by signal)
while kill -0 $TRAIN_PID 2>/dev/null; do
    wait $TRAIN_PID || true
done
wait $TRAIN_PID 2>/dev/null
EXIT_CODE=$?

# ── Auto-requeue on preemption (exit code 124) ──
if [ $EXIT_CODE -eq 124 ]; then
    echo "[SLURM] Preempted (exit 124) — resubmitting job"
    sbatch -o "${LOG_DIR}/%j_p1_s${SEED}.out" \
           -e "${LOG_DIR}/%j_p1_s${SEED}.err" \
           --job-name="E6B_p1_s${SEED}" \
           "$0" "$MODEL" "$DATASET" "$SEED"
    exit 0
fi

exit $EXIT_CODE
