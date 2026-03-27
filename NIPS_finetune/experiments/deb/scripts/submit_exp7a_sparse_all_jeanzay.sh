#!/bin/bash
################################################################################
# Submit all Exp 7A — Frozen Selector + L1 Sparse (3 lambdas x 3 seeds)
#
# Usage:
#   bash experiments/deb/scripts/submit_exp7a_sparse_all_jeanzay.sh [MODEL] [DATASET]
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 2025 3407)

# Lambda tag -> script mapping
declare -A LAMBDA_TAGS
LAMBDA_TAGS[l1e4]="1e-4"
LAMBDA_TAGS[l3e4]="3e-4"
LAMBDA_TAGS[l1e3]="1e-3"

echo "=============================================="
echo "  Submitting Exp 7A: Frozen Selector + L1 Sparse"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Lambdas: ${!LAMBDA_TAGS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for TAG in l1e4 l3e4 l1e3; do
    LOG_DIR="${PROJECT_DIR}/deb_log/exp7a_sparse_${TAG}_selector"
    mkdir -p "$LOG_DIR"

    for SEED in "${SEEDS[@]}"; do
        echo "Submitting: sparse_${TAG} (lambda=${LAMBDA_TAGS[$TAG]}) | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j_selector_frozen_sparse_s${SEED}.out" \
               -e "${LOG_DIR}/%j_selector_frozen_sparse_s${SEED}.err" \
               --job-name="E7A_${TAG}_s${SEED}" \
               "${SCRIPT_DIR}/run_exp7a_sparse_${TAG}_selector_jeanzay.sh" \
               "$MODEL" "$DATASET" "$SEED"
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted ${COUNT} jobs (3 lambdas x ${#SEEDS[@]} seeds)."
echo ""
echo "Log directories:"
echo "  ${PROJECT_DIR}/deb_log/exp7a_sparse_l1e4_selector/"
echo "  ${PROJECT_DIR}/deb_log/exp7a_sparse_l3e4_selector/"
echo "  ${PROJECT_DIR}/deb_log/exp7a_sparse_l1e3_selector/"
