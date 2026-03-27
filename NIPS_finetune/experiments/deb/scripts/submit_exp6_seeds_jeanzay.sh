#!/bin/bash
################################################################################
# Submit Exp 6 — boundary search (regime x mode) with 5 seeds on Jean Zay V100
#
# Usage:
#   bash experiments/deb/scripts/submit_exp6_seeds_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
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
echo "  Submitting Exp 6: boundary search"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Regimes: ${REGIMES[*]}"
echo "  Modes:   ${MODES[*]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "=============================================="

COUNT=0
for SEED in "${SEEDS[@]}"; do
    for REGIME in "${REGIMES[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "Submitting: ${MODE} | regime=${REGIME} | seed=${SEED}"
            sbatch -o "${LOG_DIR}/%j_${MODE}_${REGIME}_s${SEED}.out" \
                   -e "${LOG_DIR}/%j_${MODE}_${REGIME}_s${SEED}.err" \
                   --job-name="E6_${MODE}_${REGIME}_s${SEED}" \
                   "${SCRIPT_DIR}/run_exp6_jeanzay_v100.sh" \
                   "$MODEL" "$DATASET" "$REGIME" "$MODE" "$SEED"
            COUNT=$((COUNT + 1))
        done
    done
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds x ${#REGIMES[@]} regimes x ${#MODES[@]} modes)."
echo "Logs in: ${LOG_DIR}/"
