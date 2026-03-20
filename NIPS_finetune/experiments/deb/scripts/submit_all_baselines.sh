#!/bin/bash
################################################################################
# Submit all baseline experiments to SLURM.
#
# 3 datasets × 3 finetune modes × 3 seeds = 27 independent jobs.
# Each job runs a SINGLE experiment to stay within GPU time limits.
#
# Usage:
#   bash experiments/deb/scripts/submit_all_baselines.sh [MODEL]
#   MODEL: codebrain | cbramod | luna  (default: codebrain)
################################################################################

set -e

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=${1:-codebrain}

DATASETS=(TUAB DIAGNOSIS CHB-MIT)
FT_MODES=(frozen partial full)
SEEDS=(3407 42 2024)

SCRIPT="experiments/deb/scripts/run_baselines_3ft.sh"
LOG_DIR="slurm_log/baselines_${MODEL}"
mkdir -p "${LOG_DIR}"

COUNT=0

for DATASET in "${DATASETS[@]}"; do
    for FT in "${FT_MODES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="bl_${MODEL}_${DATASET}_${FT}_s${SEED}"
            LOG_FILE="${LOG_DIR}/${JOB_NAME}.out"

            echo "Submitting: ${JOB_NAME}"

            sbatch \
                --job-name="${JOB_NAME}" \
                --output="${LOG_FILE}" \
                --export="ALL,WANDB_MODE=disabled" \
                "${SCRIPT}" "${MODEL}" "${DATASET}" "${FT}" "${SEED}"

            COUNT=$((COUNT + 1))
        done
    done
done

echo ""
echo "Submitted ${COUNT} jobs for model=${MODEL}."
echo "Log dir: ${LOG_DIR}/"
echo "Monitor with: squeue -u \$(whoami)"
