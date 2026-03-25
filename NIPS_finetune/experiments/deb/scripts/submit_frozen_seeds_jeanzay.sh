#!/bin/bash
################################################################################
# Submit frozen baseline + frozen DEB experiments with 5 random seeds
# on Jean Zay (A100, gpu_p5).
#
# Usage:
#   bash experiments/deb/scripts/submit_frozen_seeds_jeanzay.sh [MODEL] [DATASET]
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

echo "=============================================="
echo "  Submitting frozen baseline + DEB"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "=============================================="

for SEED in "${SEEDS[@]}"; do
    # ── Frozen Baseline ──
    echo "Submitting: Baseline frozen | seed=${SEED}"
    sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
        --job-name="BL_${MODEL}_${DATASET}_s${SEED}" \
        "${SCRIPT_DIR}/run_baselines_3ft_jeanzay.sh" "$MODEL" "$DATASET" frozen "$SEED"

    # ── Frozen DEB ──
    echo "Submitting: DEB frozen | seed=${SEED}"
    sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
        --job-name="DEB_${MODEL}_${DATASET}_s${SEED}" \
        "${SCRIPT_DIR}/run_deb_jeanzay.sh" "$MODEL" "$DATASET" "$SEED"
done

echo ""
echo "All ${#SEEDS[@]} x 2 = $((${#SEEDS[@]} * 2)) jobs submitted."
echo "Logs in: ${LOG_DIR}/"
