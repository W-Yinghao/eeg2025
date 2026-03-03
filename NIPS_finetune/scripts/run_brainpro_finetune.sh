#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# BrainPro Downstream Fine-Tuning & Ablation Study
#
# Reproduces all ablation experiments from the BrainPro paper:
#   Table 3:  Pre-training ablation (masking, reconstruction, decoupling, retrieval)
#   Table 11: Position embedding reset ablation
#   Table 12: Token merge mode ablation (mean, aggr, all)
#   Table 13: Encoder configuration ablation (shared+affect, shared+motor, etc.)
#   Figure 6: Learning rate sensitivity
#
# Usage:
#   ./run_brainpro_finetune.sh                       # All datasets x all ablations
#   ./run_brainpro_finetune.sh TUEV                  # Only TUEV dataset
#   ./run_brainpro_finetune.sh TUEV main             # Only main ablation (Table 3)
#   ./run_brainpro_finetune.sh TUEV token_merge      # Only token merge ablation
#   ./run_brainpro_finetune.sh TUEV encoder_config   # Only encoder config ablation
#   ./run_brainpro_finetune.sh TUEV pos_emb          # Only position embedding ablation
#   ./run_brainpro_finetune.sh TUEV lr_sensitivity   # Only learning rate sensitivity
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_brainpro_finetune.py"
LOG_DIR="${PROJECT_DIR}/logs_brainpro"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_brainpro"

# WandB
WANDB_PROJECT="brainpro-finetune"

# GPU
CUDA_DEVICE=0

# Fixed training parameters (Table 7 of the paper)
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-4
MIN_LR=1e-6
WEIGHT_DECAY=0.05
DROPOUT=0.1
CLIP_GRAD_NORM=1.0
LABEL_SMOOTHING=0.1
SEED=42

# BrainPro architecture (Table 4)
K_T=32
K_C=32
K_R=32
D_MODEL=32
NHEAD=32
D_FF=64
N_ENCODER_LAYERS=4
PATCH_LEN=20
PATCH_STRIDE=20
HIDDEN_FACTOR=8

# Pretrained weights (set to empty string if training from scratch)
PRETRAINED_WEIGHTS=""
# Example: PRETRAINED_WEIGHTS="${PROJECT_DIR}/checkpoints_brainpro_pretrain/brainpro_epoch30.pth"

# Pre-training ablation weights (one per ablation, set paths when available)
WEIGHTS_NO_MASKING=""
WEIGHTS_NO_RECONSTRUCTION=""
WEIGHTS_NO_DECOUPLING=""
WEIGHTS_RANDOM_RETRIEVAL=""

# Datasets to evaluate
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

    local cmd="python ${PYTHON_SCRIPT} \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --min_lr ${MIN_LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --dropout ${DROPOUT} \
        --clip_grad_norm ${CLIP_GRAD_NORM} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --K_T ${K_T} --K_C ${K_C} --K_R ${K_R} \
        --d_model ${D_MODEL} --nhead ${NHEAD} --d_ff ${D_FF} \
        --n_encoder_layers ${N_ENCODER_LAYERS} \
        --patch_len ${PATCH_LEN} --patch_stride ${PATCH_STRIDE} \
        --hidden_factor ${HIDDEN_FACTOR} \
        --save_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        --wandb_group ${wandb_group} \
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
# Ablation: Table 3 - Pre-training ablations
# ============================================================================

run_main_ablation() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="brainpro_table3_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Table 3: Pre-Training Ablation Study - ${dataset}"
    echo "############################################################"

    # 1. Full BrainPro (with pretrained weights)
    if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
        run_experiment "${dataset}" \
            "BP_${dataset}_full_pretrained_${timestamp}" \
            "${wandb_group}" \
            "--pretrained_weights ${PRETRAINED_WEIGHTS}"
    fi

    # 2. w/o pre-training (from scratch)
    run_experiment "${dataset}" \
        "BP_${dataset}_no_pretrain_${timestamp}" \
        "${wandb_group}" \
        ""

    # 3. w/o masking (pretrained without masking)
    if [ -n "${WEIGHTS_NO_MASKING}" ] && [ -f "${WEIGHTS_NO_MASKING}" ]; then
        run_experiment "${dataset}" \
            "BP_${dataset}_wo_masking_${timestamp}" \
            "${wandb_group}" \
            "--pretrained_weights ${WEIGHTS_NO_MASKING}"
    fi

    # 4. w/o reconstruction (pretrained without reconstruction)
    if [ -n "${WEIGHTS_NO_RECONSTRUCTION}" ] && [ -f "${WEIGHTS_NO_RECONSTRUCTION}" ]; then
        run_experiment "${dataset}" \
            "BP_${dataset}_wo_reconstruction_${timestamp}" \
            "${wandb_group}" \
            "--pretrained_weights ${WEIGHTS_NO_RECONSTRUCTION}"
    fi

    # 5. w/o decoupling (pretrained without decoupling)
    if [ -n "${WEIGHTS_NO_DECOUPLING}" ] && [ -f "${WEIGHTS_NO_DECOUPLING}" ]; then
        run_experiment "${dataset}" \
            "BP_${dataset}_wo_decoupling_${timestamp}" \
            "${wandb_group}" \
            "--pretrained_weights ${WEIGHTS_NO_DECOUPLING}"
    fi

    # 6. w random retrieval (spatial learner uses random filters)
    local random_retr_weights="${PRETRAINED_WEIGHTS}"
    if [ -n "${WEIGHTS_RANDOM_RETRIEVAL}" ] && [ -f "${WEIGHTS_RANDOM_RETRIEVAL}" ]; then
        random_retr_weights="${WEIGHTS_RANDOM_RETRIEVAL}"
    fi
    local retr_extra="--random_retrieval"
    if [ -n "${random_retr_weights}" ] && [ -f "${random_retr_weights}" ]; then
        retr_extra="${retr_extra} --pretrained_weights ${random_retr_weights}"
    fi
    run_experiment "${dataset}" \
        "BP_${dataset}_random_retrieval_${timestamp}" \
        "${wandb_group}" \
        "${retr_extra}"
}

# ============================================================================
# Ablation: Table 12 - Token merge modes
# ============================================================================

run_token_merge_ablation() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="brainpro_table12_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Table 12: Token Merge Mode Ablation - ${dataset}"
    echo "############################################################"

    local pretrained_arg=""
    if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${PRETRAINED_WEIGHTS}"
    fi

    for merge_mode in "mean" "aggr" "all"; do
        run_experiment "${dataset}" \
            "BP_${dataset}_merge_${merge_mode}_${timestamp}" \
            "${wandb_group}" \
            "--token_merge ${merge_mode} ${pretrained_arg}"
    done
}

# ============================================================================
# Ablation: Table 13 - Encoder configuration
# ============================================================================

run_encoder_config_ablation() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="brainpro_table13_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Table 13: Encoder Configuration Ablation - ${dataset}"
    echo "############################################################"

    local pretrained_arg=""
    if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${PRETRAINED_WEIGHTS}"
    fi

    # Single state encoders (shared is always active)
    for state in "affect" "motor" "others"; do
        run_experiment "${dataset}" \
            "BP_${dataset}_enc_shared_${state}_${timestamp}" \
            "${wandb_group}" \
            "--active_states ${state} ${pretrained_arg}"
    done

    # Two state encoders
    for states in "affect motor" "affect others" "motor others"; do
        local states_tag=$(echo "${states}" | tr ' ' '_')
        run_experiment "${dataset}" \
            "BP_${dataset}_enc_shared_${states_tag}_${timestamp}" \
            "${wandb_group}" \
            "--active_states ${states} ${pretrained_arg}"
    done

    # All encoders
    run_experiment "${dataset}" \
        "BP_${dataset}_enc_all_${timestamp}" \
        "${wandb_group}" \
        "--active_states affect motor others ${pretrained_arg}"
}

# ============================================================================
# Ablation: Table 11 - Position embedding reset
# ============================================================================

run_pos_emb_ablation() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="brainpro_table11_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Table 11: Position Embedding Reset Ablation - ${dataset}"
    echo "############################################################"

    local pretrained_arg=""
    if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${PRETRAINED_WEIGHTS}"
    fi

    # With reset (default)
    run_experiment "${dataset}" \
        "BP_${dataset}_pos_reset_${timestamp}" \
        "${wandb_group}" \
        "${pretrained_arg}"

    # Without reset
    run_experiment "${dataset}" \
        "BP_${dataset}_pos_no_reset_${timestamp}" \
        "${wandb_group}" \
        "--no_reset_pos_emb ${pretrained_arg}"
}

# ============================================================================
# Ablation: Figure 6 - Learning rate sensitivity
# ============================================================================

run_lr_sensitivity() {
    local dataset=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local wandb_group="brainpro_fig6_${dataset}_${timestamp}"

    echo ""
    echo "############################################################"
    echo "# Figure 6: Learning Rate Sensitivity - ${dataset}"
    echo "############################################################"

    local pretrained_arg=""
    if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${PRETRAINED_WEIGHTS}"
    fi

    for lr in "1e-5" "5e-5" "1e-4" "5e-4" "1e-3"; do
        run_experiment "${dataset}" \
            "BP_${dataset}_lr_${lr}_${timestamp}" \
            "${wandb_group}" \
            "--lr ${lr} ${pretrained_arg}"
    done
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "======================================================================"
    echo "BrainPro Downstream Fine-Tuning & Ablation Study"
    echo "======================================================================"
    echo ""
    echo "Architecture: K_T=${K_T}, K_C=${K_C}, K_R=${K_R}, d=${D_MODEL}"
    echo "              nhead=${NHEAD}, d_ff=${D_FF}, layers=${N_ENCODER_LAYERS}"
    echo "              patch=${PATCH_LEN}x${PATCH_STRIDE}"
    echo ""
    echo "Training:     epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"
    echo "              weight_decay=${WEIGHT_DECAY}, clip_norm=${CLIP_GRAD_NORM}"
    echo "              label_smoothing=${LABEL_SMOOTHING}"
    echo ""
    if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
        echo "Pretrained:   ${PRETRAINED_WEIGHTS}"
    else
        echo "Pretrained:   NONE (training from scratch)"
    fi
    echo "======================================================================"
    echo ""

    setup_directories

    # Parse arguments
    local target_dataset=""
    local target_ablation=""

    if [ $# -ge 1 ]; then
        target_dataset="$1"
    fi
    if [ $# -ge 2 ]; then
        target_ablation="$2"
    fi

    # Select datasets
    local run_datasets=("${DATASETS[@]}")
    if [ -n "${target_dataset}" ]; then
        run_datasets=("${target_dataset}")
        echo "Running only dataset: ${target_dataset}"
    fi

    local total=0
    local success=0
    local fail=0

    for dataset in "${run_datasets[@]}"; do
        # Determine which ablations to run
        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "main" ]; then
            run_main_ablation "${dataset}" && ((success++)) || ((fail++))
            ((total++))
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "token_merge" ]; then
            run_token_merge_ablation "${dataset}" && ((success++)) || ((fail++))
            ((total++))
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "encoder_config" ]; then
            run_encoder_config_ablation "${dataset}" && ((success++)) || ((fail++))
            ((total++))
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "pos_emb" ]; then
            run_pos_emb_ablation "${dataset}" && ((success++)) || ((fail++))
            ((total++))
        fi

        if [ -z "${target_ablation}" ] || [ "${target_ablation}" = "lr_sensitivity" ]; then
            run_lr_sensitivity "${dataset}" && ((success++)) || ((fail++))
            ((total++))
        fi
    done

    echo ""
    echo "======================================================================"
    echo "BrainPro Fine-Tuning Ablation Study Complete"
    echo "======================================================================"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  WandB:       ${WANDB_PROJECT}"
    echo ""
    echo "Ablation groups:"
    echo "  Table 3:  Pre-training ablations (full, no_pretrain, w/o masking, etc.)"
    echo "  Table 11: Position embedding reset"
    echo "  Table 12: Token merge modes (mean, aggr, all)"
    echo "  Table 13: Encoder configurations (shared+affect, shared+motor, etc.)"
    echo "  Figure 6: Learning rate sensitivity (1e-5 to 1e-3)"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [DATASET] [ABLATION]"
    echo ""
    echo "BrainPro downstream fine-tuning with ablation studies"
    echo ""
    echo "Arguments:"
    echo "  DATASET    (optional) Specific dataset: TUEV, TUAB, TUSZ, etc."
    echo "  ABLATION   (optional) Specific ablation group:"
    echo "               main            Table 3: pre-training ablations"
    echo "               token_merge     Table 12: merge modes"
    echo "               encoder_config  Table 13: encoder selection"
    echo "               pos_emb         Table 11: position embedding reset"
    echo "               lr_sensitivity  Figure 6: learning rate sweep"
    echo ""
    echo "Examples:"
    echo "  $0                           # All datasets x all ablations"
    echo "  $0 TUEV                      # All ablations on TUEV"
    echo "  $0 TUEV main                 # Only Table 3 ablations on TUEV"
    echo "  $0 TUEV token_merge          # Only merge mode ablation on TUEV"
    echo "  $0 TUAB encoder_config       # Only encoder config ablation on TUAB"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
