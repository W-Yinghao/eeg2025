#!/bin/bash
################################################################################
# Submit all Exp 7B — Frozen Selector + Consistency (3 lambdas x 3 seeds)
#
# Usage:
#   bash experiments/deb/scripts/submit_exp7b_cons_all_jeanzay.sh [MODEL] [DATASET]
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 2025 3407)

declare -A LAMBDA_TAGS
LAMBDA_TAGS[l1e3]="1e-3"
LAMBDA_TAGS[l3e3]="3e-3"
LAMBDA_TAGS[l1e2]="1e-2"

echo "=============================================="
echo "  Submitting Exp 7B: Frozen Selector + Consistency"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Lambdas: ${!LAMBDA_TAGS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for TAG in l1e3 l3e3 l1e2; do
    LOG_DIR="${PROJECT_DIR}/deb_log/exp7b_cons_${TAG}_selector"
    mkdir -p "$LOG_DIR"

    for SEED in "${SEEDS[@]}"; do
        echo "Submitting: cons_${TAG} (lambda=${LAMBDA_TAGS[$TAG]}) | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j_selector_frozen_cons_s${SEED}.out" \
               -e "${LOG_DIR}/%j_selector_frozen_cons_s${SEED}.err" \
               --job-name="E7B_${TAG}_s${SEED}" \
               "${SCRIPT_DIR}/run_exp7b_cons_${TAG}_selector_jeanzay.sh" \
               "$MODEL" "$DATASET" "$SEED"
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted ${COUNT} jobs (3 lambdas x ${#SEEDS[@]} seeds)."
echo ""
echo "Log directories:"
echo "  ${PROJECT_DIR}/deb_log/exp7b_cons_l1e3_selector/"
echo "  ${PROJECT_DIR}/deb_log/exp7b_cons_l3e3_selector/"
echo "  ${PROJECT_DIR}/deb_log/exp7b_cons_l1e2_selector/"
