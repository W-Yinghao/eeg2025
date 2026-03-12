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
#   ./run_usba.sh                      # All models x all datasets
#   ./run_usba.sh codebrain            # CodeBrain on all datasets
#   ./run_usba.sh cbramod TUEV         # CBraMod on TUEV only
#   ./run_usba.sh all DIAGNOSIS        # All models on DIAGNOSIS only
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
WANDB_PROJECT="eeg_usba"

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

    local lr=$(get_dataset_param "${dataset}" "LR")
    local weight_decay=$(get_dataset_param "${dataset}" "WEIGHT_DECAY")
    local dropout=$(get_dataset_param "${dataset}" "DROPOUT")
    local batch_size=$(get_dataset_param "${dataset}" "BATCH_SIZE")

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="USBA_${model}_${dataset}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "USBA: ${model} / ${dataset}"
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
        --usba \
        --usba_latent_dim ${USBA_LATENT_DIM} \
        --usba_beta ${USBA_BETA} \
        --usba_lambda_cc ${USBA_LAMBDA_CC} \
        --usba_eta_budget ${USBA_ETA_BUDGET} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${model_args}"

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
    local arg_model="${1:-all}"
    local arg_dataset="${2:-all}"

    # Resolve model list
    local models=()
    if [ "${arg_model}" = "all" ]; then
        models=("${ALL_MODELS[@]}")
    else
        models=("${arg_model}")
    fi

    # Resolve dataset list
    local datasets=()
    if [ "${arg_dataset}" = "all" ]; then
        datasets=("${ALL_DATASETS[@]}")
    else
        datasets=("${arg_dataset}")
    fi

    local total=$(( ${#models[@]} * ${#datasets[@]} ))
    echo "======================================================================"
    echo "USBA Fine-Tuning"
    echo "======================================================================"
    echo "  Models:   ${models[*]}"
    echo "  Datasets: ${datasets[*]}"
    echo "  Total:    ${total} experiments"
    echo "======================================================================"

    setup_directories

    local success=0
    local fail=0
    local count=0

    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            count=$((count + 1))
            if run_experiment "${model}" "${dataset}"; then
                success=$((success + 1))
            else
                fail=$((fail + 1))
            fi
            echo "Progress: ${count}/${total} (${success} ok, ${fail} failed)"
        done
    done

    echo ""
    echo "======================================================================"
    echo "Done: ${success}/${total} succeeded, ${fail} failed"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [MODEL] [DATASET]"
    echo ""
    echo "  MODEL:   cbramod | codebrain | luna | all (default: all)"
    echo "  DATASET: TUEV | TUAB | TUSZ | DIAGNOSIS | all (default: all)"
    echo ""
    echo "Examples:"
    echo "  $0                       # all models x all datasets"
    echo "  $0 codebrain             # CodeBrain on all datasets"
    echo "  $0 cbramod DIAGNOSIS     # CBraMod on DIAGNOSIS only"
    echo "  $0 all TUEV              # All models on TUEV"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
