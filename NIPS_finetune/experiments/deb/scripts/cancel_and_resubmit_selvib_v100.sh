#!/bin/bash
################################################################################
# Cancel pending SelVIB A100 jobs and resubmit on V100
#
# Usage:
#   bash experiments/deb/scripts/cancel_and_resubmit_selvib_v100.sh
################################################################################

PROJECT_DIR="$WORK/yinghao/eeg2025/NIPS_finetune"
SCRIPT_DIR="${PROJECT_DIR}/experiments/deb/scripts"
LOG_DIR="${PROJECT_DIR}/deb_log"
mkdir -p "$LOG_DIR"

MODEL=codebrain
DATASET=TUAB
FT_MODE=partial

# ── Step 1: Cancel old A100 jobs ─────────────────────────────────────────────
OLD_JOBS=(1259967 1259968 1260022 1260023 1260024 1260025 1260026 1260027)

echo "=============================================="
echo "  Cancelling ${#OLD_JOBS[@]} old A100 jobs"
echo "=============================================="
for JOB in "${OLD_JOBS[@]}"; do
    echo "  scancel $JOB"
    scancel "$JOB"
done
echo ""

# ── Step 2: Resubmit on V100 ─────────────────────────────────────────────────
# Batch 1: beta=1e-3, seeds 3407 & 7777 (from first submission)
# Batch 2: all 3 betas x seeds 7 & 77 (from second submission)

JOBS_TO_SUBMIT=(
    # BETA   SEED
    "1e-3    3407"
    "1e-3    7777"
    "1e-4    7"
    "1e-4    77"
    "3e-4    7"
    "3e-4    77"
    "1e-3    7"
    "1e-3    77"
)

echo "=============================================="
echo "  Resubmitting on V100 (ifd@v100)"
echo "  Model: ${MODEL}  Dataset: ${DATASET}  FT: ${FT_MODE}"
echo "=============================================="

COUNT=0
for ENTRY in "${JOBS_TO_SUBMIT[@]}"; do
    read -r BETA SEED <<< "$ENTRY"
    echo "Submitting: SelVIB ${FT_MODE} | beta=${BETA} | seed=${SEED}"
    sbatch -o "${LOG_DIR}/%j.out" -e "${LOG_DIR}/%j.err" \
        --job-name="SV_b${BETA}_s${SEED}" \
        "${SCRIPT_DIR}/run_selector_vib_jeanzay_v100.sh" "$MODEL" "$DATASET" "$FT_MODE" "$SEED" "$BETA"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Cancelled ${#OLD_JOBS[@]} old jobs, submitted ${COUNT} new V100 jobs."
echo "Logs in: ${LOG_DIR}/"
