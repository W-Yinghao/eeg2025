#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# Full Fine-Tuning with CodeBrain (EEGSSM) Backbone
#
# Per the paper (Section 4.2 & Appendix H.3): "fine-tune the entire model end-to-end"
# The EEGSSM backbone includes PatchEmbedding + Residual Blocks (SGConv + SWA + Gate)
# + final conv + LayerNorm. ALL backbone + classifier parameters are trainable.
# Note: The TFDual-Tokenizer's Transformer is Stage 1 only, NOT part of finetuning.
# Single learning rate for all parameters (no multi_lr), matching Table 7 of the paper.
#
# Usage:
#   ./run_full_finetune.sh              # Run both TUEV and TUAB
#   ./run_full_finetune.sh TUEV         # Only TUEV
#   ./run_full_finetune.sh TUAB         # Only TUAB
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_tuev_lmdb.py"
LOG_DIR="${PROJECT_DIR}/logs_full_finetune"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_full_finetune"

# WandB
WANDB_PROJECT="codebrain-full-finetune"

# GPU
CUDA_DEVICE=0

# Shared parameters
EPOCHS=50
CLIP_VALUE=1.0
LABEL_SMOOTHING=0.1
SEED=3407
N_LAYER=8
CLASSIFIER="all_patch_reps"
CODEBRAIN_WEIGHTS="${PROJECT_DIR}/CodeBrain/Checkpoints/CodeBrain.pth"

# t-SNE
TSNE_INTERVAL=10
TSNE_SAMPLES=2000

# ============================================================================
# Per-dataset hyperparameters
# ============================================================================

# TUEV: 6-class, 5s segments, ~71k train samples (Table 7 of paper)
TUEV_LR=2e-5
TUEV_WEIGHT_DECAY=5e-4
TUEV_DROPOUT=0.3
TUEV_BATCH_SIZE=64

# TUAB: binary, 10s segments, ~297k train samples (Table 7 of paper)
TUAB_LR=1e-5
TUAB_WEIGHT_DECAY=5e-5
TUAB_DROPOUT=0.4
TUAB_BATCH_SIZE=512

# ============================================================================
# Functions
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "Directories created: ${LOG_DIR}, ${CHECKPOINT_DIR}"
}

run_experiment() {
    local dataset=$1
    local lr=$2
    local weight_decay=$3
    local dropout=$4
    local batch_size=$5
    local wandb_group=$6

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="FT_${dataset}_lr${lr}_wd${weight_decay}_do${dropout}_bs${batch_size}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: Full Fine-Tune - ${dataset}"
    echo "======================================================================"
    echo "  Dataset:      ${dataset}"
    echo "  LR:           ${lr}"
    echo "  Weight decay: ${weight_decay}"
    echo "  Dropout:      ${dropout}"
    echo "  Batch size:   ${batch_size}"
    echo ""
    echo "  Training params:"
    echo "    - epochs:          ${EPOCHS}"
    echo "    - classifier:      ${CLASSIFIER}"
    echo "    - label_smoothing: ${LABEL_SMOOTHING}"
    echo "    - multi_lr:        False (unified lr=${lr})"
    echo ""
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # Pretrained weights
    local pretrained_arg=""
    if [ -f "${CODEBRAIN_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${CODEBRAIN_WEIGHTS}"
        echo "  Pretrained weights: ${CODEBRAIN_WEIGHTS}"
    else
        echo "  WARNING: No pretrained weights found at ${CODEBRAIN_WEIGHTS}"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --model codebrain \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --clip_value ${CLIP_VALUE} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --dropout ${dropout} \
        --n_layer ${N_LAYER} \
        --classifier ${CLASSIFIER} \
        --no_multi_lr \
        --model_dir ${CHECKPOINT_DIR} \
        --tsne_interval ${TSNE_INTERVAL} \
        --tsne_samples ${TSNE_SAMPLES} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${pretrained_arg}"

    echo "Starting training..."
    cd "${PROJECT_DIR}"
    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo "Experiment completed: ${run_name}"
        return 0
    else
        echo "Experiment FAILED: ${run_name}"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "======================================================================"
    echo "Full Fine-Tuning with CodeBrain"
    echo "======================================================================"
    echo ""
    echo "Model: CodeBrain (EEGSSM: SGConv+SWA+Gate) - ALL parameters trainable"
    echo "  backbone layers: ${N_LAYER}"
    echo "  classifier:      ${CLASSIFIER}"
    echo "  multi_lr:        False (unified lr)"
    echo ""
    echo "Per-dataset hyperparameters:"
    echo "  TUEV: lr=${TUEV_LR}, wd=${TUEV_WEIGHT_DECAY}, dropout=${TUEV_DROPOUT}, bs=${TUEV_BATCH_SIZE}"
    echo "  TUAB: lr=${TUAB_LR}, wd=${TUAB_WEIGHT_DECAY}, dropout=${TUAB_DROPOUT}, bs=${TUAB_BATCH_SIZE}"
    echo ""
    echo "Training: epochs=${EPOCHS}, seed=${SEED}"
    echo "======================================================================"
    echo ""

    setup_directories

    local SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="FT_codebrain_${SWEEP_TIMESTAMP}"

    local total=0
    local success=0
    local fail=0

    # Filter datasets if specified
    local run_tuev=true
    local run_tuab=true
    if [ $# -gt 0 ]; then
        run_tuev=false
        run_tuab=false
        case "$1" in
            TUEV) run_tuev=true ;;
            TUAB) run_tuab=true ;;
            *)
                echo "Unknown dataset: $1 (expected TUEV or TUAB)"
                exit 1
                ;;
        esac
        echo "Running only dataset: $1"
        echo ""
    fi

    # TUEV
    if [ "${run_tuev}" = true ]; then
        total=$((total + 1))
        if run_experiment "TUEV" "${TUEV_LR}" "${TUEV_WEIGHT_DECAY}" "${TUEV_DROPOUT}" "${TUEV_BATCH_SIZE}" "${wandb_group}"; then
            success=$((success + 1))
        else
            fail=$((fail + 1))
        fi
        echo ""
        echo "Progress: ${success}/${total} passed, ${fail} failed"
        echo ""
    fi

    # TUAB
    if [ "${run_tuab}" = true ]; then
        total=$((total + 1))
        if run_experiment "TUAB" "${TUAB_LR}" "${TUAB_WEIGHT_DECAY}" "${TUAB_DROPOUT}" "${TUAB_BATCH_SIZE}" "${wandb_group}"; then
            success=$((success + 1))
        else
            fail=$((fail + 1))
        fi
        echo ""
        echo "Progress: ${success}/${total} passed, ${fail} failed"
        echo ""
    fi

    # Summary
    echo ""
    echo "======================================================================"
    echo "Full Fine-Tuning Complete"
    echo "======================================================================"
    echo "  Total:      ${total}"
    echo "  Successful: ${success}"
    echo "  Failed:     ${fail}"
    echo ""
    echo "Results:"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  WandB:       ${WANDB_PROJECT}"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [DATASET]"
    echo ""
    echo "Full fine-tuning with CodeBrain EEGSSM backbone (all params trainable, single LR)"
    echo ""
    echo "Arguments:"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo ""
    echo "Per-dataset hyperparameters:"
    echo "  TUEV: lr=2e-5, wd=5e-4, dropout=0.3, bs=64"
    echo "  TUAB: lr=1e-5, wd=5e-5, dropout=0.4, bs=512"
    echo ""
    echo "Examples:"
    echo "  $0              # Run both TUEV and TUAB"
    echo "  $0 TUEV         # Run TUEV only"
    echo "  $0 TUAB         # Run TUAB only"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
