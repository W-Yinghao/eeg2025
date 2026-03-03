#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# Vanilla Fine-Tuning for BioFoundation Models (FEMBA + LUNA)
#
# Supports:
#   - FEMBA (Bidirectional Mamba): Full finetune + Linear probe
#   - LUNA  (Cross-Attention + RoPE Transformer): Full finetune + Linear probe
#   - Datasets: TUEV (6-class) and TUAB (binary)
#   - LUNA sizes: base, large, huge
#
# FEMBA: Uses built-in MambaClassifier. Layerwise LR decay recommended.
#        Paper recommends: lr=5e-4, wd=0.5, warmup=10 epochs, cosine schedule.
#        Note: FEMBA HuggingFace only has task-specific finetuned weights,
#              not pure pretrained backbone. We train from scratch or from task weights.
#
# LUNA:  Uses ClassificationHead with learned aggregation query.
#        Paper recommends: lr=5e-4, wd=0.05, layerwise_lr_decay=0.75.
#        Has pure pretrained backbone weights from HuggingFace.
#
# Usage:
#   ./run_biofoundation_finetune.sh                      # All (2 models x 2 datasets x 2 modes = 8)
#   ./run_biofoundation_finetune.sh femba                # FEMBA only (TUEV + TUAB, full + LP)
#   ./run_biofoundation_finetune.sh luna                 # LUNA only  (TUEV + TUAB, full + LP)
#   ./run_biofoundation_finetune.sh luna TUEV            # LUNA on TUEV only
#   ./run_biofoundation_finetune.sh femba TUAB full      # FEMBA full finetune on TUAB
#   ./run_biofoundation_finetune.sh luna TUEV linear     # LUNA linear probe on TUEV
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_tuev_lmdb.py"
LOG_DIR="${PROJECT_DIR}/logs_biofoundation"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_biofoundation"

# WandB
WANDB_PROJECT="biofoundation-finetune"

# GPU
CUDA_DEVICE=0

# Shared
EPOCHS=50
CLIP_VALUE=1.0
LABEL_SMOOTHING=0.1
SEED=3407
CLASSIFIER="all_patch_reps"

# t-SNE
TSNE_INTERVAL=10
TSNE_SAMPLES=2000

# ============================================================================
# FEMBA hyperparameters (from paper + config)
# ============================================================================

# Full finetune: all params trainable, single LR
FEMBA_TUEV_LR=5e-4
FEMBA_TUEV_WEIGHT_DECAY=0.5
FEMBA_TUEV_DROPOUT=0.1
FEMBA_TUEV_BATCH_SIZE=64

FEMBA_TUAB_LR=5e-4
FEMBA_TUAB_WEIGHT_DECAY=0.5
FEMBA_TUAB_DROPOUT=0.1
FEMBA_TUAB_BATCH_SIZE=256

# Linear probe: higher LR for classifier only
FEMBA_LP_LR=0.01
FEMBA_LP_WEIGHT_DECAY=5e-2
FEMBA_LP_BATCH_SIZE=256

# FEMBA architecture
FEMBA_EMBED_DIM=79
FEMBA_NUM_BLOCKS=2
FEMBA_EXP=4
FEMBA_PATCH_SIZE="2 16"
FEMBA_STRIDE="2 16"

# FEMBA doesn't have pure pretrained backbone — train from scratch or use task weights
# To use task-specific weights as initialization:
# FEMBA_WEIGHTS="${PROJECT_DIR}/checkpoints/FEMBA/TUAB/FEMBA_base.safetensors"
FEMBA_WEIGHTS=""

# ============================================================================
# LUNA hyperparameters (from paper + config)
# ============================================================================

# Full finetune
LUNA_TUEV_LR=5e-4
LUNA_TUEV_WEIGHT_DECAY=0.05
LUNA_TUEV_DROPOUT=0.1
LUNA_TUEV_BATCH_SIZE=64

LUNA_TUAB_LR=5e-4
LUNA_TUAB_WEIGHT_DECAY=0.05
LUNA_TUAB_DROPOUT=0.1
LUNA_TUAB_BATCH_SIZE=256

# Linear probe
LUNA_LP_LR=0.01
LUNA_LP_WEIGHT_DECAY=5e-2
LUNA_LP_BATCH_SIZE=256

# LUNA architecture
LUNA_SIZE="base"  # base, large, or huge

# LUNA pretrained backbone weights (pure pretrained, not task-specific)
LUNA_WEIGHTS="${PROJECT_DIR}/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors"

# ============================================================================
# Functions
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
}

run_femba_experiment() {
    local dataset=$1
    local mode=$2  # "full" or "linear"
    local lr=$3
    local weight_decay=$4
    local dropout=$5
    local batch_size=$6

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local mode_label=$([ "$mode" = "linear" ] && echo "LP" || echo "FT")
    local run_name="${mode_label}_FEMBA_${dataset}_lr${lr}_wd${weight_decay}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "FEMBA ${mode_label} - ${dataset}"
    echo "======================================================================"
    echo "  LR: ${lr}, WD: ${weight_decay}, Dropout: ${dropout}, BS: ${batch_size}"
    echo "  Log: ${log_file}"

    local mode_args=""
    if [ "$mode" = "linear" ]; then
        mode_args="--linear_probe"
    else
        mode_args="--no_multi_lr"
    fi

    local pretrained_arg=""
    if [ -n "${FEMBA_WEIGHTS}" ] && [ -f "${FEMBA_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${FEMBA_WEIGHTS}"
    else
        pretrained_arg="--no_pretrained"
        echo "  Training FEMBA from scratch (no pure pretrained backbone available)"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --model femba \
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
        --classifier ${CLASSIFIER} \
        --femba_embed_dim ${FEMBA_EMBED_DIM} \
        --femba_num_blocks ${FEMBA_NUM_BLOCKS} \
        --femba_exp ${FEMBA_EXP} \
        --femba_patch_size ${FEMBA_PATCH_SIZE} \
        --femba_stride ${FEMBA_STRIDE} \
        --femba_use_builtin_classifier \
        --model_dir ${CHECKPOINT_DIR} \
        --tsne_interval ${TSNE_INTERVAL} \
        --tsne_samples ${TSNE_SAMPLES} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${mode_args} \
        ${pretrained_arg}"

    cd "${PROJECT_DIR}"
    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo "Completed: ${run_name}"
        return 0
    else
        echo "FAILED: ${run_name}"
        return 1
    fi
}

run_luna_experiment() {
    local dataset=$1
    local mode=$2  # "full" or "linear"
    local lr=$3
    local weight_decay=$4
    local dropout=$5
    local batch_size=$6

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local mode_label=$([ "$mode" = "linear" ] && echo "LP" || echo "FT")
    local run_name="${mode_label}_LUNA_${dataset}_${LUNA_SIZE}_lr${lr}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "LUNA (${LUNA_SIZE}) ${mode_label} - ${dataset}"
    echo "======================================================================"
    echo "  LR: ${lr}, WD: ${weight_decay}, Dropout: ${dropout}, BS: ${batch_size}"
    echo "  Log: ${log_file}"

    local mode_args=""
    if [ "$mode" = "linear" ]; then
        mode_args="--linear_probe"
    else
        mode_args="--no_multi_lr"
    fi

    local pretrained_arg=""
    if [ -f "${LUNA_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${LUNA_WEIGHTS}"
        echo "  Pretrained: ${LUNA_WEIGHTS}"
    else
        pretrained_arg="--no_pretrained"
        echo "  WARNING: No pretrained weights found"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --model luna \
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
        --classifier ${CLASSIFIER} \
        --luna_size ${LUNA_SIZE} \
        --luna_drop_path 0.1 \
        --model_dir ${CHECKPOINT_DIR} \
        --tsne_interval ${TSNE_INTERVAL} \
        --tsne_samples ${TSNE_SAMPLES} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${mode_args} \
        ${pretrained_arg}"

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
# Main
# ============================================================================

main() {
    local target_model="${1:-all}"
    local target_dataset="${2:-all}"
    local target_mode="${3:-all}"

    echo "======================================================================"
    echo "BioFoundation Vanilla Finetune"
    echo "======================================================================"
    echo "  Model:   ${target_model}"
    echo "  Dataset: ${target_dataset}"
    echo "  Mode:    ${target_mode}"
    echo "======================================================================"

    setup_directories

    local total=0
    local success=0
    local fail=0

    # --- FEMBA ---
    if [ "${target_model}" = "all" ] || [ "${target_model}" = "femba" ]; then
        # TUEV
        if [ "${target_dataset}" = "all" ] || [ "${target_dataset}" = "TUEV" ]; then
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "full" ]; then
                total=$((total + 1))
                if run_femba_experiment "TUEV" "full" "${FEMBA_TUEV_LR}" "${FEMBA_TUEV_WEIGHT_DECAY}" "${FEMBA_TUEV_DROPOUT}" "${FEMBA_TUEV_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "linear" ]; then
                total=$((total + 1))
                if run_femba_experiment "TUEV" "linear" "${FEMBA_LP_LR}" "${FEMBA_LP_WEIGHT_DECAY}" "0.1" "${FEMBA_LP_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
        fi
        # TUAB
        if [ "${target_dataset}" = "all" ] || [ "${target_dataset}" = "TUAB" ]; then
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "full" ]; then
                total=$((total + 1))
                if run_femba_experiment "TUAB" "full" "${FEMBA_TUAB_LR}" "${FEMBA_TUAB_WEIGHT_DECAY}" "${FEMBA_TUAB_DROPOUT}" "${FEMBA_TUAB_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "linear" ]; then
                total=$((total + 1))
                if run_femba_experiment "TUAB" "linear" "${FEMBA_LP_LR}" "${FEMBA_LP_WEIGHT_DECAY}" "0.1" "${FEMBA_LP_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
        fi
    fi

    # --- LUNA ---
    if [ "${target_model}" = "all" ] || [ "${target_model}" = "luna" ]; then
        # TUEV
        if [ "${target_dataset}" = "all" ] || [ "${target_dataset}" = "TUEV" ]; then
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "full" ]; then
                total=$((total + 1))
                if run_luna_experiment "TUEV" "full" "${LUNA_TUEV_LR}" "${LUNA_TUEV_WEIGHT_DECAY}" "${LUNA_TUEV_DROPOUT}" "${LUNA_TUEV_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "linear" ]; then
                total=$((total + 1))
                if run_luna_experiment "TUEV" "linear" "${LUNA_LP_LR}" "${LUNA_LP_WEIGHT_DECAY}" "0.1" "${LUNA_LP_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
        fi
        # TUAB
        if [ "${target_dataset}" = "all" ] || [ "${target_dataset}" = "TUAB" ]; then
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "full" ]; then
                total=$((total + 1))
                if run_luna_experiment "TUAB" "full" "${LUNA_TUAB_LR}" "${LUNA_TUAB_WEIGHT_DECAY}" "${LUNA_TUAB_DROPOUT}" "${LUNA_TUAB_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
            if [ "${target_mode}" = "all" ] || [ "${target_mode}" = "linear" ]; then
                total=$((total + 1))
                if run_luna_experiment "TUAB" "linear" "${LUNA_LP_LR}" "${LUNA_LP_WEIGHT_DECAY}" "0.1" "${LUNA_LP_BATCH_SIZE}"; then
                    success=$((success + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
        fi
    fi

    # Summary
    echo ""
    echo "======================================================================"
    echo "BioFoundation Finetune Complete"
    echo "======================================================================"
    echo "  Total: ${total}, Success: ${success}, Failed: ${fail}"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "======================================================================"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [MODEL] [DATASET] [MODE]"
    echo ""
    echo "  MODEL:   femba | luna | all (default: all)"
    echo "  DATASET: TUEV | TUAB | all (default: all)"
    echo "  MODE:    full | linear | all (default: all)"
    echo ""
    echo "Examples:"
    echo "  $0                          # All 8 experiments"
    echo "  $0 luna                     # LUNA on both datasets, full+LP"
    echo "  $0 femba TUEV full          # FEMBA full finetune on TUEV"
    echo "  $0 luna TUAB linear         # LUNA linear probe on TUAB"
    exit 0
fi

main "$@"
