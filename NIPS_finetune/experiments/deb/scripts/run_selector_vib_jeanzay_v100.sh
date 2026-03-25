#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -A ifd@v100
#SBATCH -C v100-32g

################################################################################
# Selector + local VIB ablation (gates + fusion + VIB, NO sparse reg)
# = DEB with sparse_lambda=0, sweeping beta
# V100 version (gpu_p13 partition)
#
# Usage:
#   sbatch run_selector_vib_jeanzay_v100.sh [MODEL] [DATASET] [FT_MODE] [SEED] [BETA]
################################################################################

# ── Environment setup ────────────────────────────────────────────────────────
module purge
module load pytorch-gpu/py3/2.6.0
export PYTHONUSERBASE=$WORK/.local_pt260
export WANDB_MODE=offline

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
FT_MODE=${3:-partial}
SEED=${4:-3407}
BETA=${5:-1e-4}

echo "=============================================="
echo "  Selector+VIB: ${MODEL} | ${DATASET} | ${FT_MODE} | seed=${SEED} | beta=${BETA}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

EXTRA_ARGS=()
if [ "$FT_MODE" = "partial" ]; then
    EXTRA_ARGS+=(--lr_ratio 10)
fi

TAG="SelVIB_${MODEL}_${DATASET}_${FT_MODE}_b${BETA}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

python experiments/deb/scripts/train.py \
    --mode deb \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --finetune "$FT_MODE" \
    --epochs 100 \
    --batch_size 64 \
    --lr_head 1e-3 \
    --patience 15 \
    --clip_value 5.0 \
    --split_strategy subject \
    --deb_latent_dim 64 \
    --deb_gate_hidden 64 \
    --deb_fusion concat \
    --beta "$BETA" \
    --beta_warmup_epochs 5 \
    --no_sparse_reg \
    --seed "$SEED" \
    --cuda 0 \
    --save_dir "checkpoints_deb" \
    --num_workers 4 \
    --wandb_project eeg_deb \
    --wandb_run_name "$TAG" \
    "${EXTRA_ARGS[@]}"
