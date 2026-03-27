#!/bin/bash
################################################################################
# Submit Exp 6A — Frozen Selector with 5 seeds on Jean Zay H100
#
# This is a dedicated submission script for the frozen selector supplementary
# experiment. Completely independent from the original Exp6 submission.
#
# Usage:
#   bash experiments/deb/scripts/submit_exp6a_frozen_selector_seeds_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUAB | TUSZ | DIAGNOSIS | ... (default: TUAB)
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="$WORK/yinghao/eeg2025/NIPS_finetune/deb_log/exp6a_frozen_selector"
SAVE_DIR="$WORK/yinghao/eeg2025/NIPS_finetune/checkpoints_selector/exp6a_frozen_selector"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 1234 2025 3407 7777)

echo "=============================================="
echo "  Submitting Exp 6A: Frozen Selector"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "  GPU:   H100 (priority)"
echo "  Log:   ${LOG_DIR}"
echo "  Save:  ${SAVE_DIR}"
echo "=============================================="

COUNT=0
for SEED in "${SEEDS[@]}"; do
    echo "Submitting: frozen selector | seed=${SEED}"
    sbatch -o "${LOG_DIR}/%j_frozen_selector_s${SEED}.out" \
           -e "${LOG_DIR}/%j_frozen_selector_s${SEED}.err" \
           --job-name="E6A_sel_frz_s${SEED}" \
           "${SCRIPT_DIR}/run_exp6a_frozen_selector_jeanzay.sh" \
           "$MODEL" "$DATASET" "$SEED"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds)."
echo "Logs in:        ${LOG_DIR}/"
echo "Checkpoints in: ${SAVE_DIR}/"
