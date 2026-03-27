#!/bin/bash
################################################################################
# Submit ALL Exp 6B — Gentle Partial FT Baseline configs on Jean Zay V100
#
# Submits P1, P2, P3e1, P3e2 x 3 seeds = 12 jobs total.
#
# Usage:
#   bash experiments/deb/scripts/submit_exp6b_gpartial_baseline_all_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUAB | TUEV | TUSZ | ...     (default: TUAB)
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 2025 3407)

# Config → script mapping
declare -A CONFIGS
CONFIGS[p1]="run_exp6b_gpartial_p1_baseline_jeanzay.sh"
CONFIGS[p2]="run_exp6b_gpartial_p2_baseline_jeanzay.sh"
CONFIGS[p3e1]="run_exp6b_gpartial_p3e1_baseline_jeanzay.sh"
CONFIGS[p3e2]="run_exp6b_gpartial_p3e2_baseline_jeanzay.sh"

echo "=============================================="
echo "  Submitting Exp 6B: Gentle Partial FT Baseline"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Configs: ${!CONFIGS[*]}"
echo "=============================================="

COUNT=0
for CONFIG_NAME in p1 p2 p3e1 p3e2; do
    SCRIPT="${CONFIGS[$CONFIG_NAME]}"
    LOG_DIR="${PROJECT_DIR}/deb_log/exp6b_gpartial_${CONFIG_NAME}_baseline"
    mkdir -p "$LOG_DIR"

    for SEED in "${SEEDS[@]}"; do
        echo "Submitting: ${CONFIG_NAME} | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j_${CONFIG_NAME}_s${SEED}.out" \
               -e "${LOG_DIR}/%j_${CONFIG_NAME}_s${SEED}.err" \
               --job-name="E6B_${CONFIG_NAME}_s${SEED}" \
               "${SCRIPT_DIR}/${SCRIPT}" \
               "$MODEL" "$DATASET" "$SEED"
        COUNT=$((COUNT + 1))
    done
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds x 4 configs)."
echo ""
echo "Log directories:"
echo "  deb_log/exp6b_gpartial_p1_baseline/"
echo "  deb_log/exp6b_gpartial_p2_baseline/"
echo "  deb_log/exp6b_gpartial_p3e1_baseline/"
echo "  deb_log/exp6b_gpartial_p3e2_baseline/"
echo ""
echo "Checkpoint directories:"
echo "  checkpoints_selector/exp6b_gpartial_p1_baseline/"
echo "  checkpoints_selector/exp6b_gpartial_p2_baseline/"
echo "  checkpoints_selector/exp6b_gpartial_p3e1_baseline/"
echo "  checkpoints_selector/exp6b_gpartial_p3e2_baseline/"
