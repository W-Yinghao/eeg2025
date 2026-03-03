#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# SageStream (SA-MoE + IIB) Fine-Tuning with CodeBrain / CBraMod Backbone
#
# Uses frozen backbone + trainable SA-MoE + IIB
# Supports both CodeBrain (SSSM) and CBraMod (Transformer) backbones
# Includes: Subject-Aware MoE routing, style alignment, IIB, GRL adversarial
#
# Usage:
#   sbatch run_sagestream.sh                        # All backbones x datasets x configs
#   sbatch run_sagestream.sh codebrain              # CodeBrain only
#   sbatch run_sagestream.sh cbramod                # CBraMod only
#   sbatch run_sagestream.sh codebrain TUEV         # CodeBrain + TUEV only
#   sbatch run_sagestream.sh cbramod TUAB           # CBraMod + TUAB only
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_sagestream.py"
LOG_DIR="${PROJECT_DIR}/logs_sagestream"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_sagestream"

# WandB
WANDB_PROJECT="codebrain-sagestream"

# GPU
CUDA_DEVICE=0

# Fixed training parameters
EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-3
DROPOUT=0.1
CLIP_VALUE=5.0
PATIENCE=15

# SA-MoE defaults
NUM_EXPERTS=4
TOP_K=2
N_MOE_LAYERS=2
AUX_WEIGHT=0.01

# IIB defaults
LATENT_DIM=128
ALPHA_KL=1e-3
BETA_ADV=0.5
GRL_GAMMA=10.0

# Backbone settings
SEED=3407
CODEBRAIN_WEIGHTS="${PROJECT_DIR}/CodeBrain/Checkpoints/CodeBrain.pth"
CODEBRAIN_N_LAYER=8
CBRAMOD_WEIGHTS="${PROJECT_DIR}/../NIPS/Cbramod_pretrained_weights.pth"
CBRAMOD_N_LAYER=12
LUNA_WEIGHTS="${PROJECT_DIR}/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors"
LUNA_N_LAYER=8
FEMBA_N_LAYER=2

# Backbones and Datasets
BACKBONES=("codebrain" "cbramod" "femba" "luna")
DATASETS=("TUEV" "TUAB")

# ============================================================================
# Ablation Configurations
# "alpha_kl|beta_adv|aux_weight|num_experts|top_k|n_moe_layers|use_style|description"
# ============================================================================
declare -a SAGE_CONFIGS=(
    "1e-3|0.5|0.01|4|2|2|yes|full_sagestream"
    "0.0|0.0|0.01|4|2|2|yes|moe_only"
    "1e-3|0.5|0.0|4|2|0|no|iib_only"
    "0.0|0.0|0.0|4|2|0|no|baseline_ce"
    "1e-3|0.5|0.01|8|2|2|yes|high_experts"
    "1e-3|0.5|0.01|2|1|2|yes|low_experts"
    "1e-3|0.5|0.01|4|2|3|yes|deep_moe"
    "1e-3|0.5|0.01|4|2|2|no|no_style"
)

# ============================================================================
# Functions
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "Directories created: ${LOG_DIR}, ${CHECKPOINT_DIR}"
}

run_experiment() {
    local backbone=$1
    local dataset=$2
    local alpha_kl=$3
    local beta_adv=$4
    local aux_weight=$5
    local num_experts=$6
    local top_k=$7
    local n_moe_layers=$8
    local use_style=$9
    local config_desc=${10}
    local wandb_group=${11}

    # Backbone-specific settings
    local n_layer weights_path
    case "${backbone}" in
        codebrain)
            n_layer=${CODEBRAIN_N_LAYER}
            weights_path="${CODEBRAIN_WEIGHTS}"
            ;;
        cbramod)
            n_layer=${CBRAMOD_N_LAYER}
            weights_path="${CBRAMOD_WEIGHTS}"
            ;;
        luna)
            n_layer=${LUNA_N_LAYER}
            weights_path="${LUNA_WEIGHTS}"
            ;;
        femba)
            n_layer=${FEMBA_N_LAYER}
            weights_path=""  # FEMBA has no pure pretrained backbone
            ;;
    esac

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="sage_${backbone}_${dataset}_${config_desc}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: SageStream - ${config_desc}"
    echo "======================================================================"
    echo "  Backbone:      ${backbone} (n_layer=${n_layer})"
    echo "  Dataset:       ${dataset}"
    echo "  alpha_kl:      ${alpha_kl}"
    echo "  beta_adv:      ${beta_adv}"
    echo "  aux_weight:    ${aux_weight}"
    echo "  num_experts:   ${num_experts}"
    echo "  top_k:         ${top_k}"
    echo "  n_moe_layers:  ${n_moe_layers}"
    echo "  use_style:     ${use_style}"
    echo ""
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # Pretrained weights
    local pretrained_arg=""
    if [ -f "${weights_path}" ]; then
        pretrained_arg="--pretrained_weights ${weights_path}"
    else
        echo "  WARNING: No pretrained weights found at ${weights_path}"
    fi

    # Subject args
    local subject_arg=""
    if [ "${beta_adv}" = "0.0" ] || [ "${beta_adv}" = "0" ]; then
        subject_arg="--no_subjects"
    fi

    # Style args
    local style_arg=""
    if [ "${use_style}" = "no" ]; then
        style_arg="--no_style"
    fi

    # Build command
    local cmd="python ${PYTHON_SCRIPT} \
        --model ${backbone} \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --dropout ${DROPOUT} \
        --clip_value ${CLIP_VALUE} \
        --patience ${PATIENCE} \
        --num_experts ${num_experts} \
        --top_k ${top_k} \
        --n_moe_layers ${n_moe_layers} \
        --aux_weight ${aux_weight} \
        --latent_dim ${LATENT_DIM} \
        --alpha_kl ${alpha_kl} \
        --beta_adv ${beta_adv} \
        --grl_gamma ${GRL_GAMMA} \
        --n_layer ${n_layer} \
        --save_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        --wandb_group ${wandb_group} \
        --run_name ${run_name} \
        ${pretrained_arg} \
        ${subject_arg} \
        ${style_arg}"

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
    # Parse arguments: [BACKBONE] [DATASET]
    local target_backbones=("${BACKBONES[@]}")
    local target_datasets=("${DATASETS[@]}")

    if [ $# -ge 1 ]; then
        case "$1" in
            codebrain|cbramod|femba|luna)
                target_backbones=("$1")
                if [ $# -ge 2 ]; then
                    case "$2" in
                        TUEV|TUAB) target_datasets=("$2") ;;
                        *) echo "Unknown dataset: $2 (expected TUEV or TUAB)"; exit 1 ;;
                    esac
                fi
                ;;
            TUEV|TUAB)
                target_datasets=("$1")
                ;;
            *)
                echo "Unknown argument: $1 (expected codebrain, cbramod, femba, luna, TUEV, or TUAB)"
                exit 1
                ;;
        esac
    fi

    local total_runs=$(( ${#target_backbones[@]} * ${#target_datasets[@]} * ${#SAGE_CONFIGS[@]} ))

    echo "======================================================================"
    echo "SageStream (SA-MoE + IIB) Fine-Tuning"
    echo "======================================================================"
    echo ""
    echo "Experiment configuration:"
    echo "  Backbones:    ${target_backbones[*]}"
    echo "  Datasets:     ${target_datasets[*]}"
    echo "  Configs:      ${#SAGE_CONFIGS[@]}"
    echo "  Total runs:   ${total_runs}"
    echo ""
    echo "Backbone settings:"
    echo "  codebrain: EEGSSM (n_layer=${CODEBRAIN_N_LAYER})"
    echo "  cbramod:   Transformer (n_layer=${CBRAMOD_N_LAYER})"
    echo ""
    echo "SA-MoE + IIB:"
    echo "  latent_dim:      ${LATENT_DIM}"
    echo "  default experts: ${NUM_EXPERTS}, top_k: ${TOP_K}"
    echo ""
    echo "Training: epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"
    echo "======================================================================"
    echo ""

    setup_directories

    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0

    local SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    for backbone in "${target_backbones[@]}"; do
        for dataset in "${target_datasets[@]}"; do
            local wandb_group="sage_${backbone}_${dataset}_${SWEEP_TIMESTAMP}"
            echo "WandB group: ${wandb_group}"

            for sage_config in "${SAGE_CONFIGS[@]}"; do
                IFS='|' read -r alpha_kl beta_adv aux_weight num_experts top_k n_moe_layers use_style config_desc <<< "${sage_config}"

                total_experiments=$((total_experiments + 1))

                if run_experiment "${backbone}" "${dataset}" "${alpha_kl}" "${beta_adv}" "${aux_weight}" "${num_experts}" "${top_k}" "${n_moe_layers}" "${use_style}" "${config_desc}" "${wandb_group}"; then
                    successful_experiments=$((successful_experiments + 1))
                else
                    failed_experiments=$((failed_experiments + 1))
                fi

                echo ""
                echo "Progress: ${successful_experiments}/${total_experiments} passed, ${failed_experiments} failed"
                echo ""
            done
        done
    done

    # Summary
    echo ""
    echo "======================================================================"
    echo "SageStream Ablation Study Complete"
    echo "======================================================================"
    echo "  Total:      ${total_experiments}"
    echo "  Successful: ${successful_experiments}"
    echo "  Failed:     ${failed_experiments}"
    echo ""
    echo "  Backbones:   ${target_backbones[*]}"
    echo "  Datasets:    ${target_datasets[*]}"
    echo ""
    echo "Results:"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  WandB:       ${WANDB_PROJECT}"
    echo ""
    echo "Ablation configs:"
    echo "  1. full_sagestream: SA-MoE + IIB (full model)"
    echo "  2. moe_only:        SA-MoE only, no IIB"
    echo "  3. iib_only:        IIB only, no SA-MoE layers"
    echo "  4. baseline_ce:     CE loss only (no MoE, no IIB)"
    echo "  5. high_experts:    8 experts, top_k=2"
    echo "  6. low_experts:     2 experts, top_k=1"
    echo "  7. deep_moe:        3 SA-MoE layers"
    echo "  8. no_style:        SA-MoE without subject style alignment"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [BACKBONE] [DATASET]"
    echo ""
    echo "SageStream (SA-MoE + IIB) fine-tuning with CodeBrain / CBraMod backbone"
    echo ""
    echo "Arguments:"
    echo "  BACKBONE  (optional) codebrain or cbramod"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo ""
    echo "Examples:"
    echo "  $0                        # All backbones x datasets x configs (32 experiments)"
    echo "  $0 codebrain              # CodeBrain only, all datasets (16 experiments)"
    echo "  $0 cbramod                # CBraMod only, all datasets (16 experiments)"
    echo "  $0 codebrain TUEV         # CodeBrain + TUEV only (8 experiments)"
    echo "  $0 cbramod TUAB           # CBraMod + TUAB only (8 experiments)"
    echo "  $0 TUEV                   # All backbones, TUEV only (16 experiments)"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
