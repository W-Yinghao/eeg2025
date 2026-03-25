#!/bin/bash
################################################################################
# Submit Experiment 7: sparse / consistency / both across seeds
#
# Usage:
#   bash experiments/deb/scripts/submit_exp7_seeds.sh [MODEL] [DATASET] [REGIME]
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="${PROJECT_DIR}/deb_log/exp7"
mkdir -p "$LOG_DIR"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}
REGIME=${3:-top2}

SEEDS=(42 1234 2025 3407 7777)
VARIANTS=(sparse consistency both)

echo "=============================================="
echo "  Submitting Experiment 7: Interpretability"
echo "  Model: ${MODEL}  Dataset: ${DATASET}  Regime: ${REGIME}"
echo "  Variants: ${VARIANTS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

count=0
for SEED in "${SEEDS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        JOB_NAME="E7_${VARIANT:0:4}_${REGIME}_s${SEED}"
        echo "Submitting: ${JOB_NAME}"
        sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
            --job-name="$JOB_NAME" \
            "${SCRIPT_DIR}/run_exp7_jeanzay.sh" \
            "$MODEL" "$DATASET" "$REGIME" "$VARIANT" "$SEED"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted ${count} jobs. Logs: ${LOG_DIR}/"
