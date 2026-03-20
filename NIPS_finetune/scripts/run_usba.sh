#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# USBA Fine-Tuning: CBraMod / CodeBrain / LUNA
#
# Usage:
#   ./run_usba.sh MODEL DATASET LAYERS [RUN_NAME]
#   WANDB_GROUP=tusz_place ./run_usba.sh cbramod TUSZ 10,11 tusz_last2
#   WANDB_GROUP=tusz_place ./run_usba.sh cbramod TUSZ 9,10,11 tusz_last3
#   WANDB_GROUP=tusz_place ./run_usba.sh cbramod TUSZ all tusz_alllayers
################################################################################

set -e

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg2025

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_usba.py"
LOG_DIR="${PROJECT_DIR}/logs_usba"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_usba"

# WandB
WANDB_PROJECT="eeg_usba_v2"

# GPU
CUDA_DEVICE=0

# Shared training parameters
EPOCHS=100
CLIP_VALUE=5.0
SEED=3407
PATIENCE=15

# USBA parameters
USBA_LATENT_DIM=64
USBA_BETA=1e-4
USBA_LAMBDA_CC=0.01
USBA_ETA_BUDGET=1e-3

# USBA layer selection (passed from CLI)
WANDB_GROUP="${WANDB_GROUP:-}"

# ============================================================================
# Models
# ============================================================================
ALL_MODELS=("cbramod" "codebrain" "luna")

# CodeBrain-specific
CODEBRAIN_N_LAYER=8

# LUNA-specific
LUNA_SIZE="base"

# ============================================================================
# Per-dataset hyperparameters
# ============================================================================

# TUEV: 6-class, 5s segments
TUEV_LR=1e-3
TUEV_WEIGHT_DECAY=1e-3
TUEV_DROPOUT=0.1
TUEV_BATCH_SIZE=64

# TUAB: binary, 10s segments
TUAB_LR=1e-3
TUAB_WEIGHT_DECAY=1e-3
TUAB_DROPOUT=0.1
TUAB_BATCH_SIZE=64

# TUSZ: binary seizure detection, 22ch, 5s segments
TUSZ_LR=1e-3
TUSZ_WEIGHT_DECAY=1e-3
TUSZ_DROPOUT=0.1
TUSZ_BATCH_SIZE=64

# DIAGNOSIS: 4-class disease classification, 58ch, 5s segments
DIAGNOSIS_LR=1e-3
DIAGNOSIS_WEIGHT_DECAY=1e-3
DIAGNOSIS_DROPOUT=0.1
DIAGNOSIS_BATCH_SIZE=64

ALL_DATASETS=("TUEV" "TUAB" "TUSZ" "DIAGNOSIS")

# ============================================================================
# Functions
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
}

get_dataset_param() {
    local dataset=$1
    local param=$2
    eval echo "\${${dataset}_${param}}"
}

run_experiment() {
    local model=$1
    local dataset=$2
    local selected_layers=${3:-"output"}
    local custom_run_name=$4
    shift 4 2>/dev/null || true
    local extra_args="$*"

    local lr=$(get_dataset_param "${dataset}" "LR")
    local weight_decay=$(get_dataset_param "${dataset}" "WEIGHT_DECAY")
    local dropout=$(get_dataset_param "${dataset}" "DROPOUT")
    local batch_size=$(get_dataset_param "${dataset}" "BATCH_SIZE")

    local run_name="${custom_run_name:-USBA_${model}_${dataset}_$(date +%Y%m%d_%H%M%S)}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "USBA: ${model} / ${dataset} / layers=${selected_layers}"
    echo "======================================================================"
    echo "  LR: ${lr}  WD: ${weight_decay}  Dropout: ${dropout}  BS: ${batch_size}"
    echo "  USBA: latent=${USBA_LATENT_DIM} beta=${USBA_BETA} lambda_cc=${USBA_LAMBDA_CC} eta=${USBA_ETA_BUDGET}"
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # Model-specific args
    local model_args=""
    case "${model}" in
        codebrain)
            model_args="--n_layer ${CODEBRAIN_N_LAYER}"
            ;;
        luna)
            model_args="--luna_size ${LUNA_SIZE}"
            ;;
        cbramod)
            model_args=""
            ;;
    esac

    local cmd="python ${PYTHON_SCRIPT} \
        --model ${model} \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --clip_value ${CLIP_VALUE} \
        --dropout ${dropout} \
        --patience ${PATIENCE} \
        --save_dir ${CHECKPOINT_DIR} \
        --eval_test_every_epoch \
        --usba \
        --usba_latent_dim ${USBA_LATENT_DIM} \
        --usba_beta ${USBA_BETA} \
        --usba_lambda_cc ${USBA_LAMBDA_CC} \
        --usba_eta_budget ${USBA_ETA_BUDGET} \
        --usba_selected_layers ${selected_layers} \
        --wandb_project ${WANDB_PROJECT} \
        ${WANDB_GROUP:+--wandb_group ${WANDB_GROUP}} \
        --wandb_run_name ${run_name} \
        ${model_args} ${extra_args}"

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
    local arg_model="${1:-cbramod}"
    local arg_dataset="${2:-TUSZ}"
    local arg_layers="${3:-output}"
    local arg_run_name="${4:-}"
    shift 4 2>/dev/null || true
    local extra_args="$*"

    echo "======================================================================"
    echo "USBA Fine-Tuning"
    echo "======================================================================"
    echo "  Model:   ${arg_model}"
    echo "  Dataset: ${arg_dataset}"
    echo "  Layers:  ${arg_layers}"
    [ -n "${extra_args}" ] && echo "  Extra:   ${extra_args}"
    echo "======================================================================"

    setup_directories

    run_experiment "${arg_model}" "${arg_dataset}" "${arg_layers}" "${arg_run_name}" ${extra_args}
}

show_usage() {
    echo "Usage: $0 MODEL DATASET LAYERS [RUN_NAME] [EXTRA_ARGS...]"
    echo ""
    echo "  MODEL:      cbramod | codebrain | luna"
    echo "  DATASET:    TUEV | TUAB | TUSZ | DIAGNOSIS"
    echo "  LAYERS:     e.g. 10,11 | 9,10,11 | all | output"
    echo "  RUN_NAME:   custom wandb run name (optional)"
    echo "  EXTRA_ARGS: additional flags passed to train_usba.py"
    echo ""
    echo "  Set WANDB_GROUP env var for wandb grouping."
    echo ""
    echo "Examples:"
    echo "  WANDB_GROUP=tusz_place $0 cbramod TUSZ 10,11 tusz_last2"
    echo "  WANDB_GROUP=tuev_diag $0 cbramod TUEV output tuev_adapter_ce --no_subjects --usba_no_cc_inv --usba_no_budget --usba_beta 0"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
