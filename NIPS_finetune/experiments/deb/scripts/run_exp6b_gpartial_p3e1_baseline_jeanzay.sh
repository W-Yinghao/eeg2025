#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00
#SBATCH -A ifd@v100
#SBATCH --qos=qos_gpu-t4
#SBATCH -C v100-32g
#SBATCH --signal=B:USR1@120

################################################################################
# Exp 6B — Gentle Partial FT Baseline: P3 (stage1=1 epoch)
#
# Staged partial fine-tuning:
#   Stage 1: top1 unfrozen, 1 epoch, lr_bb=1e-5, lr_head=1e-3, warmup=1
#   Stage 2: backbone frozen, 20 epochs, lr_head=5e-4, warmup=2, patience=6
#
# Resume logic:
#   - If preempted during stage1 (cheap): re-runs stage1 from scratch
#   - If preempted during stage2: resumes from checkpoint
#   - Uses .stage1_done marker to skip stage1 on resume
#
# Usage:
#   sbatch run_exp6b_gpartial_p3e1_baseline_jeanzay.sh [MODEL] [DATASET] [SEED]
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
EXP_TAG="exp6b_gpartial_p3e1_baseline"

SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/${EXP_TAG}"
LOG_DIR="${PROJECT_DIR}/deb_log/${EXP_TAG}"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TAG="Exp6B_P3e1_${MODE}_${MODEL}_${DATASET}_staged_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Exp 6B P3e1: Staged Partial FT Baseline (stage1=1ep)"
echo "  ${MODE} | ${MODEL} | ${DATASET} | seed=${SEED}"
echo "  Stage1: top1, 1ep, lr_bb=1e-5, lr_head=1e-3"
echo "  Stage2: frozen, 20ep, lr_head=5e-4"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Save: ${SAVE_DIR}"
echo "=============================================="

# ── Signal forwarding for SLURM preemption ──
TRAIN_PID=""
forward_signal() {
    if [ -n "$TRAIN_PID" ]; then
        echo "[SLURM] Forwarding SIGUSR1 to training process (PID=$TRAIN_PID)"
        kill -USR1 "$TRAIN_PID" 2>/dev/null
    fi
}
trap forward_signal USR1

python experiments/deb/scripts/train_partial_ft.py \
    --mode "$MODE" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --regime top1 \
    --resume "$SAVE_DIR" \
    --staged_partial \
    --stage1_epochs 1 \
    --stage1_lr_head 1e-3 \
    --stage1_lr_backbone 1e-5 \
    --stage1_warmup_epochs 1 \
    --stage2_epochs 20 \
    --stage2_lr_head 5e-4 \
    --stage2_warmup_epochs 2 \
    --stage2_patience 6 \
    --batch_size 64 \
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

while kill -0 $TRAIN_PID 2>/dev/null; do
    wait $TRAIN_PID || true
done
wait $TRAIN_PID 2>/dev/null
EXIT_CODE=$?

# ── Auto-requeue on preemption ──
if [ $EXIT_CODE -eq 124 ]; then
    echo "[SLURM] Preempted (exit 124) — resubmitting job"
    sbatch -o "${LOG_DIR}/%j_p3e1_s${SEED}.out" \
           -e "${LOG_DIR}/%j_p3e1_s${SEED}.err" \
           --job-name="E6B_p3e1_s${SEED}" \
           "$0" "$MODEL" "$DATASET" "$SEED"
    exit 0
fi

exit $EXIT_CODE
