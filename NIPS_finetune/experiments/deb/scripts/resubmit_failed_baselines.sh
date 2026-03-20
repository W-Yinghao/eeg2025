#!/bin/bash
################################################################################
# Resubmit failed baseline experiments (CUDA OOM due to GPU sharing).
#
# 12 failed jobs identified from SLURM logs:
#   - All 9 CHB-MIT jobs (frozen/partial/full × 3 seeds)
#   - DIAGNOSIS: frozen s2024/s3407, partial s2024/s3407/s42, full s2024/s42
#   - TUAB: full s2024
################################################################################

set -e

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
cd "${PROJECT_DIR}"

MODEL=codebrain
SCRIPT="experiments/deb/scripts/run_baselines_3ft.sh"
LOG_DIR="slurm_log/baselines_${MODEL}"
mkdir -p "${LOG_DIR}"

COUNT=0

submit_job() {
    local DATASET=$1
    local FT=$2
    local SEED=$3

    JOB_NAME="bl_${MODEL}_${DATASET}_${FT}_s${SEED}"
    LOG_FILE="${LOG_DIR}/${JOB_NAME}.out"

    echo "Submitting: ${JOB_NAME}"

    sbatch \
        --job-name="${JOB_NAME}" \
        --output="${LOG_FILE}" \
        --export="ALL,WANDB_MODE=disabled" \
        "${SCRIPT}" "${MODEL}" "${DATASET}" "${FT}" "${SEED}"

    COUNT=$((COUNT + 1))
}

# ── CHB-MIT: all 9 jobs failed ───────────────────────────────────────────────
for FT in frozen partial full; do
    for SEED in 3407 42 2024; do
        submit_job CHB-MIT "$FT" "$SEED"
    done
done

# ── DIAGNOSIS: 7 jobs failed ────────────────────────────────────────────────
submit_job DIAGNOSIS frozen 2024
submit_job DIAGNOSIS frozen 3407
submit_job DIAGNOSIS partial 2024
submit_job DIAGNOSIS partial 3407
submit_job DIAGNOSIS partial 42
submit_job DIAGNOSIS full 2024
submit_job DIAGNOSIS full 42

# ── TUAB: 1 job failed ──────────────────────────────────────────────────────
submit_job TUAB full 2024

echo ""
echo "Resubmitted ${COUNT} failed jobs for model=${MODEL}."
echo "Log dir: ${LOG_DIR}/"
echo "Monitor with: squeue -u \$(whoami)"
