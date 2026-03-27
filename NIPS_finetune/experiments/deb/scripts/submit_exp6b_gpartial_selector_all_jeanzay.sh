#!/bin/bash
################################################################################
# Submit ALL Exp 6B — Gentle Partial Selector (P1 + P2 + P3e1 + P3e2)
# with 3 seeds each on Jean Zay H100
#
# Usage:
#   bash experiments/deb/scripts/submit_exp6b_gpartial_selector_all_jeanzay.sh [MODEL] [DATASET]
#   MODEL:   codebrain | cbramod | luna   (default: codebrain)
#   DATASET: TUAB | TUSZ | DIAGNOSIS | ... (default: TUAB)
#
# Total jobs: 4 configs x 3 seeds = 12
################################################################################

set -e

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"

MODEL=${1:-codebrain}
DATASET=${2:-TUAB}

SEEDS=(42 2025 3407)

# Format: script_name:log_subdir:job_prefix:description
CONFIGS=(
    "run_exp6b_gpartial_p1_selector_jeanzay.sh:exp6b_gpartial_p1_selector:E6B_P1:P1 lr_bb=1e-5"
    "run_exp6b_gpartial_p2_selector_jeanzay.sh:exp6b_gpartial_p2_selector:E6B_P2:P2 lr_bb=1e-6"
    "run_exp6b_gpartial_p3e1_selector_jeanzay.sh:exp6b_gpartial_p3e1_selector:E6B_P3e1:P3 staged s1=1ep"
    "run_exp6b_gpartial_p3e2_selector_jeanzay.sh:exp6b_gpartial_p3e2_selector:E6B_P3e2:P3 staged s1=2ep"
)

echo "=============================================="
echo "  Submitting Exp 6B: Gentle Partial Selector"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Configs: P1, P2, P3e1, P3e2"
echo "  GPU:   H100"
echo "=============================================="
echo ""

COUNT=0
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r SCRIPT LOG_NAME JOB_PREFIX DESC <<< "$CFG"
    LOG_DIR="${PROJECT_DIR}/deb_log/${LOG_NAME}"
    SAVE_DIR="${PROJECT_DIR}/checkpoints_selector/${LOG_NAME}"
    mkdir -p "$LOG_DIR" "$SAVE_DIR"

    echo "--- ${DESC} ---"
    for SEED in "${SEEDS[@]}"; do
        echo "  Submitting: ${JOB_PREFIX} | seed=${SEED}"
        sbatch -o "${LOG_DIR}/%j_s${SEED}.out" \
               -e "${LOG_DIR}/%j_s${SEED}.err" \
               --job-name="${JOB_PREFIX}_s${SEED}" \
               "${SCRIPT_DIR}/${SCRIPT}" \
               "$MODEL" "$DATASET" "$SEED"
        COUNT=$((COUNT + 1))
    done
    echo ""
done

echo "=============================================="
echo "Submitted ${COUNT} jobs (4 configs x ${#SEEDS[@]} seeds)."
echo ""
echo "Log directories:"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r _ LOG_NAME _ DESC <<< "$CFG"
    echo "  ${DESC}: ${PROJECT_DIR}/deb_log/${LOG_NAME}/"
done
echo ""
echo "Checkpoint directories:"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r _ LOG_NAME _ DESC <<< "$CFG"
    echo "  ${DESC}: ${PROJECT_DIR}/checkpoints_selector/${LOG_NAME}/"
done
echo "=============================================="
