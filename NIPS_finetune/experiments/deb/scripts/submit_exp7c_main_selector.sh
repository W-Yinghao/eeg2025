#!/bin/bash
################################################################################
# Submit Exp 7C-main — Joint Sparse + Consistency (gpu-gw)
#
# 3 seeds x 1 config = 3 jobs
#
# Usage:
#   bash experiments/deb/scripts/submit_exp7c_main_selector.sh [MODEL] [DATASET]
################################################################################

set -e

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 2025 3407)

EXP_NAME="exp7c_main_selector"
LOG_DIR="${PROJECT_DIR}/deb_log/${EXP_NAME}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  Submitting Exp 7C-main: Joint Sparse + Consistency (gpu-gw)"
echo "  lambda_sparse=1e-3 (both gates)  lambda_cons=3e-3"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for SEED in "${SEEDS[@]}"; do
    echo "Submitting: ${EXP_NAME} | seed=${SEED}"
    sbatch -o "${LOG_DIR}/%j_selector_frozen_7c_main_s${SEED}.out" \
           -e "${LOG_DIR}/%j_selector_frozen_7c_main_s${SEED}.err" \
           --job-name="E7C_main_s${SEED}" \
           "${SCRIPT_DIR}/run_exp7c_main_selector.sh" \
           "$MODEL" "$DATASET" "$SEED"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds)."
echo ""
echo "Log directory:  ${LOG_DIR}/"
echo "Checkpoint dir: ${PROJECT_DIR}/checkpoints_selector/${EXP_NAME}/"
