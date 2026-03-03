#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# SCOPE: Structured Prototype-Guided Adaptation for EEG Foundation Models
#
# Ablation Study Runner
#
# Reproduces all ablation experiments from the SCOPE paper (Table 2):
#   1. Supervision construction ablations (ETF, Prototype, full Stage 1)
#   2. ProAdapter design ablations (adapter, confidence, prototype cond.)
#   3. Training strategy ablations (warmup, sequential, two-stage)
#
# Sensitivity analyses (Appendix F):
#   4. ProAdapter depth L (1-8)
#   5. Confidence threshold ρ (0-0.95)
#   6. Prototypes per class M (1-5)
#   7. ETF loss weight λ_ETF (0.01-0.5)
#   8. Pseudo-labeled data ratio (0.5-3)
#
# Usage:
#   ./run_scope.sh                           # All datasets x all ablations
#   ./run_scope.sh TUEV                      # Only TUEV
#   ./run_scope.sh TUEV main                 # Only main ablation (Table 2)
#   ./run_scope.sh TUEV sensitivity_depth    # Only adapter depth sensitivity
#
# Backbone can be set via MODEL env variable:
#   MODEL=luna ./run_scope.sh TUEV main      # LUNA backbone
#   MODEL=femba ./run_scope.sh TUAB          # FEMBA backbone
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_scope.py"
LOG_DIR="${PROJECT_DIR}/logs_scope"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_scope"

# WandB
WANDB_PROJECT="scope-eeg"

# GPU
CUDA_DEVICE=0

# Fixed parameters
SEED=42
LABEL_RATIO=0.3
BATCH_SIZE=64

# Stage 1 defaults
TPN_EPOCHS=50
TPN_LR=5e-4
LAMBDA_ETF=0.1
NUM_PROTOTYPES=3
PROTO_EPOCHS=50
CONFIDENCE_THRESHOLD=0.5

# Stage 2 defaults
EPOCHS=60
LR=1e-4
MIN_LR=1e-6
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=10
ADAPTER_LAYERS=3
PSEUDO_RATIO=2.0
PATIENCE=15
DROPOUT=0.1

# Backbone (can be overridden by MODEL env variable)
MODEL="${MODEL:-codebrain}"
case "${MODEL}" in
    codebrain) N_LAYER=8 ;;
    cbramod)   N_LAYER=12 ;;
    luna)      N_LAYER=8 ;;
    femba)     N_LAYER=2 ;;
    *)         echo "Unknown model: ${MODEL}"; exit 1 ;;
esac

# Datasets
DATASETS=("TUEV" "TUAB")

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
    local run_name=$2
    local wandb_group=$3
    shift 3
    local extra_args="$@"

    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: ${run_name}"
    echo "======================================================================"
    echo "  Dataset:    ${dataset}"
    echo "  Extra args: ${extra_args}"
    echo "  Log:        ${log_file}"
    echo "----------------------------------------------------------------------"

    # Model-specific extra args
    local model_args=""
    if [ "${MODEL}" = "luna" ]; then
        model_args="--luna_size base"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --dataset ${dataset} \
        --model ${MODEL} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --label_ratio ${LABEL_RATIO} \
        --batch_size ${BATCH_SIZE} \
        --n_layer ${N_LAYER} \
        --tpn_epochs ${TPN_EPOCHS} \
        --tpn_lr ${TPN_LR} \
        --lambda_etf ${LAMBDA_ETF} \
        --num_prototypes ${NUM_PROTOTYPES} \
        --proto_epochs ${PROTO_EPOCHS} \
        --confidence_threshold ${CONFIDENCE_THRESHOLD} \
        --adapter_layers ${ADAPTER_LAYERS} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --min_lr ${MIN_LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --pseudo_ratio ${PSEUDO_RATIO} \
        --dropout ${DROPOUT} \
        --patience ${PATIENCE} \
        --save_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        --wandb_group ${wandb_group} \
        ${model_args} \
        ${extra_args}"

    echo "Starting..."
    cd "${PROJECT_DIR}"
    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo "Completed: ${run_name}"
        return 0
    else
        echo "FAILED: ${run_name}"
        return 1
    fi
}

# ============================================================================
# Ablation: Table 2 - Main ablation study
# ============================================================================

run_main_ablation() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="scope_table2_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Table 2: Main Ablation Study - ${dataset}"
    echo "############################################################"

    # 1. Full SCOPE model
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_full_${timestamp}" \
        "${wandb_group}" ""

    # --- Supervision Construction Ablations ---

    # 2. w/o ETF guidance
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_etf_${timestamp}" \
        "${wandb_group}" "--no_etf"

    # 3. w/o Prototype Clustering
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_proto_${timestamp}" \
        "${wandb_group}" "--no_prototype"

    # 4. w/o Supervision construction (entire Stage 1)
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_supcon_${timestamp}" \
        "${wandb_group}" "--no_supervision_construction"

    # --- ProAdapter Design Ablations ---

    # 5. w/o ProAdapter (frozen backbone + classifier only)
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_adapter_${timestamp}" \
        "${wandb_group}" "--no_proadapter"

    # 6. w/o Confidence Weights
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_conf_${timestamp}" \
        "${wandb_group}" "--no_confidence_weights"

    # 7. w/o Prototype Conditioning
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_pcond_${timestamp}" \
        "${wandb_group}" "--no_prototype_conditioning"

    # --- Training Strategy Ablations ---

    # 8. w/o Warm-up
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_no_warmup_${timestamp}" \
        "${wandb_group}" "--no_warmup"

    # 9. Sequential training
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_sequential_${timestamp}" \
        "${wandb_group}" "--sequential_training"

    # 10. Two-stage training
    run_experiment "${dataset}" \
        "SCOPE_${dataset}_twostage_${timestamp}" \
        "${wandb_group}" "--two_stage_training"
}

# ============================================================================
# Sensitivity: ProAdapter Depth L
# ============================================================================

run_sensitivity_depth() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="scope_depth_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Sensitivity: ProAdapter Depth L - ${dataset}"
    echo "############################################################"

    for depth in 0 1 2 3 4 5 6 7 8; do
        local extra=""
        if [ "${depth}" -eq 0 ]; then
            extra="--no_proadapter"
        else
            extra="--adapter_layers ${depth}"
        fi
        run_experiment "${dataset}" \
            "SCOPE_${dataset}_depth${depth}_${timestamp}" \
            "${wandb_group}" "${extra}"
    done
}

# ============================================================================
# Sensitivity: Confidence Threshold ρ
# ============================================================================

run_sensitivity_conf() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="scope_conf_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Sensitivity: Confidence Threshold ρ - ${dataset}"
    echo "############################################################"

    for rho in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"; do
        run_experiment "${dataset}" \
            "SCOPE_${dataset}_rho${rho}_${timestamp}" \
            "${wandb_group}" "--confidence_threshold ${rho}"
    done
}

# ============================================================================
# Sensitivity: Prototypes per class M
# ============================================================================

run_sensitivity_proto() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="scope_proto_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Sensitivity: Prototypes per Class M - ${dataset}"
    echo "############################################################"

    for M in 1 2 3 4 5; do
        run_experiment "${dataset}" \
            "SCOPE_${dataset}_M${M}_${timestamp}" \
            "${wandb_group}" "--num_prototypes ${M}"
    done
}

# ============================================================================
# Sensitivity: ETF Loss Weight λ_ETF
# ============================================================================

run_sensitivity_etf() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="scope_etf_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Sensitivity: ETF Loss Weight λ_ETF - ${dataset}"
    echo "############################################################"

    for etf_w in "0.01" "0.05" "0.1" "0.2" "0.5"; do
        run_experiment "${dataset}" \
            "SCOPE_${dataset}_etf${etf_w}_${timestamp}" \
            "${wandb_group}" "--lambda_etf ${etf_w}"
    done
}

# ============================================================================
# Sensitivity: Pseudo-labeled Data Ratio
# ============================================================================

run_sensitivity_ratio() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="scope_ratio_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Sensitivity: Pseudo-labeled Data Ratio - ${dataset}"
    echo "############################################################"

    for ratio in "0.5" "1.0" "1.5" "2.0" "2.5" "3.0"; do
        run_experiment "${dataset}" \
            "SCOPE_${dataset}_ratio${ratio}_${timestamp}" \
            "${wandb_group}" "--pseudo_ratio ${ratio}"
    done
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "======================================================================"
    echo "SCOPE: Structured Prototype-Guided Adaptation"
    echo "======================================================================"
    echo ""
    echo "Backbone:  ${MODEL} (${N_LAYER} layers)"
    echo "Label:     ${LABEL_RATIO} ratio"
    echo "Stage 1:   TPN(${TPN_EPOCHS}ep) + Proto(M=${NUM_PROTOTYPES}) + Fusion(ρ=${CONFIDENCE_THRESHOLD})"
    echo "Stage 2:   ProAdapter(L=${ADAPTER_LAYERS}) + epochs=${EPOCHS} + warmup=${WARMUP_EPOCHS}"
    echo "Training:  lr=${LR}, wd=${WEIGHT_DECAY}, pseudo_ratio=${PSEUDO_RATIO}"
    echo "======================================================================"
    echo ""

    setup_directories

    local target_dataset=""
    local target_ablation=""

    if [ $# -ge 1 ]; then
        target_dataset="$1"
    fi
    if [ $# -ge 2 ]; then
        target_ablation="$2"
    fi

    local run_datasets=("${DATASETS[@]}")
    if [ -n "${target_dataset}" ]; then
        run_datasets=("${target_dataset}")
        echo "Running only dataset: ${target_dataset}"
    fi

    for dataset in "${run_datasets[@]}"; do
        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "main" ]; then
            run_main_ablation "${dataset}"
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "sensitivity_depth" ]; then
            run_sensitivity_depth "${dataset}"
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "sensitivity_conf" ]; then
            run_sensitivity_conf "${dataset}"
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "sensitivity_proto" ]; then
            run_sensitivity_proto "${dataset}"
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "sensitivity_etf" ]; then
            run_sensitivity_etf "${dataset}"
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "sensitivity_ratio" ]; then
            run_sensitivity_ratio "${dataset}"
        fi
    done

    echo ""
    echo "======================================================================"
    echo "SCOPE Ablation Study Complete"
    echo "======================================================================"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  WandB:       ${WANDB_PROJECT}"
    echo ""
    echo "Ablation groups:"
    echo "  Table 2:     Main ablations (10 configs)"
    echo "  Depth:       ProAdapter depth L (0-8)"
    echo "  Confidence:  Threshold ρ (0-0.9)"
    echo "  Prototypes:  Per-class M (1-5)"
    echo "  ETF weight:  λ_ETF (0.01-0.5)"
    echo "  Pseudo ratio: data ratio (0.5-3.0)"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [DATASET] [ABLATION]"
    echo ""
    echo "SCOPE adaptation with ablation studies"
    echo ""
    echo "Arguments:"
    echo "  DATASET    (optional) TUEV, TUAB, etc."
    echo "  ABLATION   (optional) Specific ablation:"
    echo "               main              Table 2: all 10 ablations"
    echo "               sensitivity_depth  ProAdapter depth L"
    echo "               sensitivity_conf   Confidence threshold ρ"
    echo "               sensitivity_proto  Prototypes per class M"
    echo "               sensitivity_etf    ETF loss weight λ_ETF"
    echo "               sensitivity_ratio  Pseudo-labeled data ratio"
    echo ""
    echo "Examples:"
    echo "  $0                              # All datasets x all ablations"
    echo "  $0 TUEV                         # All ablations on TUEV"
    echo "  $0 TUEV main                    # Only Table 2 ablations on TUEV"
    echo "  $0 TUEV sensitivity_depth       # Only depth sweep on TUEV"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
