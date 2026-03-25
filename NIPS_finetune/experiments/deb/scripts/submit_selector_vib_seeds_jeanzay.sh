#!/bin/bash
################################################################################
# Submit Selector + local VIB ablation: partial FT, sweep beta, 5 seeds
# on Jean Zay (A100, gpu_p5).
#
# Experiment: gates + fusion + VIB (no sparse), sweep beta in {1e-4, 3e-4, 1e-3}
# Question: does local VIB bring real gains over selector-only?
#
# Usage:
#   bash experiments/deb/scripts/submit_selector_vib_seeds_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
################################################################################

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="${PROJECT_DIR}/deb_log"
mkdir -p "$LOG_DIR"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 1234 2025 3407 7777)
BETAS=(1e-4 3e-4 1e-3)
FT_MODE=partial

echo "=============================================="
echo "  Submitting Selector + VIB ablation"
echo "  Model: ${MODEL}  Dataset: ${DATASET}  FT: ${FT_MODE}"
echo "  Betas: ${BETAS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for BETA in "${BETAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting: SelVIB ${FT_MODE} | beta=${BETA} | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
            --job-name="SV_b${BETA}_s${SEED}" \
            "${SCRIPT_DIR}/run_selector_vib_jeanzay.sh" "$MODEL" "$DATASET" "$FT_MODE" "$SEED" "$BETA"
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted ${COUNT} jobs (${#BETAS[@]} betas x ${#SEEDS[@]} seeds)."
echo "Logs in: ${LOG_DIR}/"
