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
# Selector-only ablation (gates + fusion + classifier, NO VIB) — Jean Zay
#
# Usage:
#   sbatch run_selector_3ft_jeanzay.sh [MODEL] [DATASET] [FT_MODE] [SEED]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
#   FT_MODE: frozen | partial | full  (default: frozen)
#   SEED:    random seed  (default: 3407)
################################################################################

set -e

# ── Environment setup ────────────────────────────────────────────────────────
module purge
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/.local_pt260
export WANDB_MODE=offline

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
FT_MODE=${3:-frozen}
SEED=${4:-3407}

echo "=============================================="
echo "  Selector: ${MODEL} | ${DATASET} | ${FT_MODE} | seed=${SEED}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

EXTRA_ARGS=()
if [ "$FT_MODE" = "partial" ]; then
    EXTRA_ARGS+=(--lr_ratio 10)
fi

TAG="Selector_${MODEL}_${DATASET}_${FT_MODE}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

python experiments/deb/scripts/train_selector.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --finetune "$FT_MODE" \
    --epochs 100 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --patience 15 \
    --clip_value 5.0 \
    --split_strategy subject \
    --deb_gate_hidden 64 \
    --deb_fusion concat \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "checkpoints_deb" \
    --num_workers 4 \
    --wandb_project eeg_deb \
    --wandb_run_name "$TAG" \
    "${EXTRA_ARGS[@]}"
