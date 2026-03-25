#!/bin/bash
################################################################################
# Submit Experiment 6: all regime x mode x seed combinations
#
# Usage:
#   bash experiments/deb/scripts/submit_exp6_seeds.sh [MODEL] [DATASET]
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="${PROJECT_DIR}/deb_log/exp6"
mkdir -p "$LOG_DIR"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 1234 2025 3407 7777)
REGIMES=(frozen top1 top2 top4)
MODES=(baseline selector)

echo "=============================================="
echo "  Submitting Experiment 6: Boundary Search"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Regimes: ${REGIMES[*]}"
echo "  Modes: ${MODES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

count=0
for SEED in "${SEEDS[@]}"; do
    for REGIME in "${REGIMES[@]}"; do
        for MODE in "${MODES[@]}"; do
            JOB_NAME="E6_${MODE:0:3}_${REGIME}_s${SEED}"
            echo "Submitting: ${JOB_NAME}"
            sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
                --job-name="$JOB_NAME" \
                "${SCRIPT_DIR}/run_exp6_jeanzay.sh" \
                "$MODEL" "$DATASET" "$REGIME" "$MODE" "$SEED"
            count=$((count + 1))
        done
    done
done

echo ""
echo "Submitted ${count} jobs. Logs: ${LOG_DIR}/"
