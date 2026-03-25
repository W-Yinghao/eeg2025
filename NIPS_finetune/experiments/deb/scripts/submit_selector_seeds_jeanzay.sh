#!/bin/bash
################################################################################
# Submit selector-only ablation: frozen + partial + full, 5 seeds each
# on Jean Zay (A100, gpu_p5).
#
# Usage:
#   bash experiments/deb/scripts/submit_selector_seeds_jeanzay.sh [MODEL] [DATASET]
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
FT_MODES=(frozen partial full)

echo "=============================================="
echo "  Submitting Selector-only ablation"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  FT modes: ${FT_MODES[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for SEED in "${SEEDS[@]}"; do
    for FT in "${FT_MODES[@]}"; do
        echo "Submitting: Selector ${FT} | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
            --job-name="SEL_${FT}_${MODEL}_s${SEED}" \
            "${SCRIPT_DIR}/run_selector_3ft_jeanzay.sh" "$MODEL" "$DATASET" "$FT" "$SEED"
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds x ${#FT_MODES[@]} FT modes)."
echo "Logs in: ${LOG_DIR}/"
