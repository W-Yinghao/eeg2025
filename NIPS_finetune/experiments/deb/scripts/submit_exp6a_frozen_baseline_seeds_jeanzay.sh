#!/bin/bash
################################################################################
# Submit Exp 6A — Frozen Baseline Re-run with 5 seeds on Jean Zay V100
#
# Submits one job per seed, all using frozen regime + baseline mode.
# Logs and checkpoints are isolated from Exp6.
#
# Usage:
#   bash experiments/deb/scripts/submit_exp6a_frozen_baseline_seeds_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | ...  (default: TUAB)
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="${PROJECT_DIR}/deb_log/exp6a_frozen_baseline"
mkdir -p "$LOG_DIR"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 1234 2025 3407 7777)

echo "=============================================="
echo "  Submitting Exp 6A: Frozen Baseline Re-run"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Log dir: ${LOG_DIR}"
echo "=============================================="

COUNT=0
for SEED in "${SEEDS[@]}"; do
    echo "Submitting: baseline | regime=frozen | seed=${SEED}"
    sbatch -o "${LOG_DIR}/%j_baseline_frozen_s${SEED}.out" \
           -e "${LOG_DIR}/%j_baseline_frozen_s${SEED}.err" \
           --job-name="E6A_bl_frozen_s${SEED}" \
           "${SCRIPT_DIR}/run_exp6a_frozen_baseline_jeanzay.sh" \
           "$MODEL" "$DATASET" "$SEED"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted ${COUNT} jobs (${#SEEDS[@]} seeds x 1 regime x 1 mode)."
echo "Logs in: ${LOG_DIR}/"
echo "Checkpoints in: ${PROJECT_DIR}/checkpoints_selector/exp6a_frozen_baseline/"
