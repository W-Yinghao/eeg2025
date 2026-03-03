#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# MSFT Fine-Tuning: Multi-Scale Feature Transformation with Multiple Backbones
#
# Supports: CodeBrain (SSSM), CBraMod (Transformer), LUNA, FEMBA
# Note: For LUNA/FEMBA, MSFT falls back to backbone_factory + simple head
#       since multi-scale cross-aggregation is designed for patch-based backbones.
#
# Usage:
#   ./run_msft.sh                         # All experiments
#   ./run_msft.sh codebrain               # CodeBrain only
#   ./run_msft.sh cbramod                 # CBraMod only
#   ./run_msft.sh luna                    # LUNA only
#   ./run_msft.sh femba                   # FEMBA only
#   ./run_msft.sh codebrain TUEV          # CodeBrain + TUEV only
#   ./run_msft.sh luna TUAB               # LUNA + TUAB only
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_msft.py"
LOG_DIR="${PROJECT_DIR}/logs_msft"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_msft"

# WandB
WANDB_PROJECT="eeg-msft"

# GPU
CUDA_DEVICE=0
SEED=3407

# Fixed training params
EPOCHS=30
CLIP_VALUE=1.0
LABEL_SMOOTHING=0.1

# ============================================================================
# Per-model per-dataset hyperparameters
# ============================================================================

# --- CodeBrain ---
codebrain_TUEV_LR=2e-5
codebrain_TUEV_WEIGHT_DECAY=5e-4
codebrain_TUEV_DROPOUT=0.4
codebrain_TUEV_BATCH_SIZE=64
codebrain_TUEV_N_LAYER=8

codebrain_TUAB_LR=1e-5
codebrain_TUAB_WEIGHT_DECAY=5e-5
codebrain_TUAB_DROPOUT=0.4
codebrain_TUAB_BATCH_SIZE=512
codebrain_TUAB_N_LAYER=8

# --- CBraMod ---
cbramod_TUEV_LR=2e-5
cbramod_TUEV_WEIGHT_DECAY=5e-4
cbramod_TUEV_DROPOUT=0.4
cbramod_TUEV_BATCH_SIZE=64
cbramod_TUEV_N_LAYER=12

cbramod_TUAB_LR=1e-5
cbramod_TUAB_WEIGHT_DECAY=5e-5
cbramod_TUAB_DROPOUT=0.4
cbramod_TUAB_BATCH_SIZE=512
cbramod_TUAB_N_LAYER=12

# --- LUNA ---
luna_TUEV_LR=2e-5
luna_TUEV_WEIGHT_DECAY=5e-4
luna_TUEV_DROPOUT=0.4
luna_TUEV_BATCH_SIZE=64
luna_TUEV_N_LAYER=8

luna_TUAB_LR=1e-5
luna_TUAB_WEIGHT_DECAY=5e-5
luna_TUAB_DROPOUT=0.4
luna_TUAB_BATCH_SIZE=512
luna_TUAB_N_LAYER=8

# --- FEMBA ---
femba_TUEV_LR=2e-5
femba_TUEV_WEIGHT_DECAY=5e-4
femba_TUEV_DROPOUT=0.4
femba_TUEV_BATCH_SIZE=64
femba_TUEV_N_LAYER=2

femba_TUAB_LR=1e-5
femba_TUAB_WEIGHT_DECAY=5e-5
femba_TUAB_DROPOUT=0.4
femba_TUAB_BATCH_SIZE=512
femba_TUAB_N_LAYER=2

# ============================================================================
# Backbone weights
# ============================================================================

CODEBRAIN_WEIGHTS="${PROJECT_DIR}/CodeBrain/Checkpoints/CodeBrain.pth"
CBRAMOD_WEIGHTS="${PROJECT_DIR}/Cbramod_pretrained_weights.pth"
LUNA_WEIGHTS="${PROJECT_DIR}/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors"
FEMBA_WEIGHTS=""  # No pure pretrained backbone

# ============================================================================
# Models, Datasets, Scale configs
# ============================================================================

MODELS=("codebrain" "cbramod" "luna" "femba")
DATASETS=("TUEV" "TUAB")

# Scale ablation (only applies to codebrain/cbramod; luna/femba use num_scales=1)
declare -a SCALE_CONFIGS=(
    "1|single_scale"
    "2|two_scales"
    "3|three_scales"
    "4|four_scales"
)

# ============================================================================
# Functions
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "Directories created: ${LOG_DIR}, ${CHECKPOINT_DIR}"
}

get_param() {
    local model=$1
    local dataset=$2
    local param=$3
    eval echo "\${${model}_${dataset}_${param}}"
}

get_weights() {
    local model=$1
    case "${model}" in
        codebrain) echo "${CODEBRAIN_WEIGHTS}" ;;
        cbramod)   echo "${CBRAMOD_WEIGHTS}" ;;
        luna)      echo "${LUNA_WEIGHTS}" ;;
        femba)     echo "${FEMBA_WEIGHTS}" ;;
    esac
}

run_experiment() {
    local model=$1
    local dataset=$2
    local num_scales=$3
    local scale_desc=$4

    # Per-model per-dataset hyperparameters
    local lr=$(get_param "${model}" "${dataset}" "LR")
    local weight_decay=$(get_param "${model}" "${dataset}" "WEIGHT_DECAY")
    local dropout=$(get_param "${model}" "${dataset}" "DROPOUT")
    local batch_size=$(get_param "${model}" "${dataset}" "BATCH_SIZE")
    local n_layer=$(get_param "${model}" "${dataset}" "N_LAYER")
    local weights_path=$(get_weights "${model}")

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="MSFT_${model}_${dataset}_s${num_scales}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: MSFT - ${model} ${scale_desc}"
    echo "======================================================================"
    echo "  Model:        ${model}"
    echo "  Dataset:      ${dataset}"
    echo "  Num Scales:   ${num_scales}"
    echo ""
    echo "  Params:"
    echo "    - epochs:       ${EPOCHS}"
    echo "    - batch_size:   ${batch_size}"
    echo "    - lr:           ${lr}"
    echo "    - weight_decay: ${weight_decay}"
    echo "    - dropout:      ${dropout}"
    echo "    - n_layer:      ${n_layer}"
    echo ""
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # Pretrained weights
    local pretrained_arg=""
    if [ -n "${weights_path}" ] && [ -f "${weights_path}" ]; then
        pretrained_arg="--pretrained_weights ${weights_path}"
        echo "  Pretrained weights: ${weights_path}"
    else
        echo "  WARNING: No pretrained weights, using random init"
    fi

    # Model-specific extra args
    local extra_args=""
    if [ "${model}" = "cbramod" ]; then
        extra_args="--dim_feedforward 800 --nhead 8"
    elif [ "${model}" = "luna" ]; then
        extra_args="--luna_size base"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --model ${model} \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --dropout ${dropout} \
        --clip_value ${CLIP_VALUE} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --n_layer ${n_layer} \
        --num_scales ${num_scales} \
        --model_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${pretrained_arg} \
        ${extra_args}"

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
    # Parse arguments: [MODEL] [DATASET]
    local target_models=("${MODELS[@]}")
    local target_datasets=("${DATASETS[@]}")

    if [ $# -ge 1 ]; then
        case "$1" in
            codebrain|cbramod|femba|luna)
                target_models=("$1")
                if [ $# -ge 2 ]; then
                    target_datasets=("$2")
                fi
                ;;
            TUEV|TUAB)
                target_datasets=("$1")
                ;;
            *)
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    fi

    echo "======================================================================"
    echo "MSFT Fine-Tuning"
    echo "======================================================================"
    echo ""
    echo "  Models:   ${target_models[*]}"
    echo "  Datasets: ${target_datasets[*]}"
    echo ""
    echo "  Note: LUNA/FEMBA use backbone_factory + simple head (no multi-scale)"
    echo "======================================================================"
    echo ""

    setup_directories

    local total=0
    local success=0
    local fail=0

    for model in "${target_models[@]}"; do
        # For LUNA/FEMBA, only run single_scale (multi-scale doesn't apply)
        local configs=("${SCALE_CONFIGS[@]}")
        if [ "${model}" = "luna" ] || [ "${model}" = "femba" ]; then
            configs=("1|single_scale")
        fi

        for dataset in "${target_datasets[@]}"; do
            for scale_config in "${configs[@]}"; do
                IFS='|' read -r num_scales scale_desc <<< "${scale_config}"

                total=$((total + 1))

                if run_experiment "${model}" "${dataset}" "${num_scales}" "${scale_desc}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi

                echo ""
                echo "Progress: ${success}/${total} passed, ${fail} failed"
                echo ""
            done
        done
    done

    # Summary
    echo ""
    echo "======================================================================"
    echo "MSFT Fine-Tuning Complete"
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
    echo "Usage: $0 [MODEL] [DATASET]"
    echo ""
    echo "MSFT fine-tuning with multiple backbones"
    echo ""
    echo "Arguments:"
    echo "  MODEL     (optional) codebrain, cbramod, luna, or femba"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo ""
    echo "Examples:"
    echo "  $0                          # All experiments"
    echo "  $0 codebrain                # CodeBrain on TUEV + TUAB (4 scales each)"
    echo "  $0 luna TUEV                # LUNA on TUEV (single scale)"
    echo "  $0 femba                    # FEMBA on TUEV + TUAB (single scale)"
    echo ""
    echo "Note: LUNA/FEMBA only run single_scale since multi-scale"
    echo "      cross-aggregation is designed for patch-based backbones."
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
