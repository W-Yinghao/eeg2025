#!/bin/bash
################################################################################
# Submit partial + full fine-tune experiments (baseline & DEB) with 5 seeds
# on Jean Zay. Frozen already done — this covers the remaining two FT modes.
#
# Usage:
#   bash experiments/deb/scripts/submit_full_partial_seeds_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="${PROJECT_DIR}/deb_log"
mkdir -p "$LOG_DIR"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 1234 2025 3407 7777)
FT_MODES=(partial full)

echo "=============================================="
echo "  Submitting partial + full: baseline & DEB"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for SEED in "${SEEDS[@]}"; do
    for FT in "${FT_MODES[@]}"; do
        # ── Baseline ──
        echo "Submitting: Baseline ${FT} | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
            --job-name="BL_${FT}_${MODEL}_s${SEED}" \
            "${SCRIPT_DIR}/run_baselines_3ft_jeanzay.sh" "$MODEL" "$DATASET" "$FT" "$SEED"

        # ── DEB ──
        echo "Submitting: DEB ${FT} | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
            --job-name="DEB_${FT}_${MODEL}_s${SEED}" \
            "${SCRIPT_DIR}/run_deb_3ft_jeanzay.sh" "$MODEL" "$DATASET" "$FT" "$SEED"

        COUNT=$((COUNT + 2))
    done
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds x ${#FT_MODES[@]} FT modes x 2 methods)."
echo "Logs in: ${LOG_DIR}/"
