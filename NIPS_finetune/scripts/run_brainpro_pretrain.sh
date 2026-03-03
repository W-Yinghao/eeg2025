#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# BrainPro Pre-Training & Pre-Training Ablation Study
#
# Pre-trains BrainPro on multiple EEG datasets grouped by brain state.
# Runs 5 configurations for ablation (Table 3):
#   1. Full pre-training (masking + reconstruction + decoupling + learned retrieval)
#   2. w/o masking
#   3. w/o reconstruction
#   4. w/o decoupling
#   5. w random retrieval
#
# Usage:
#   ./run_brainpro_pretrain.sh full             # Full pre-training only
#   ./run_brainpro_pretrain.sh ablation         # All 5 ablation configs
#   ./run_brainpro_pretrain.sh no_masking       # Single ablation config
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_brainpro_pretrain.py"
LOG_DIR="${PROJECT_DIR}/logs_brainpro_pretrain"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_brainpro_pretrain"

# Data config: JSON file specifying datasets per brain state
# Format: {"affect": [{"path": "...", "n_channels": 62, ...}], "motor": [...], "others": [...]}
DATA_CONFIG="${PROJECT_DIR}/configs/brainpro_pretrain_config.json"

# WandB
WANDB_PROJECT="brainpro-pretrain"

# GPU
CUDA_DEVICE=0

# Pre-training hyperparameters (Table 4)
EPOCHS=30
BATCH_SIZE=32
LR=1e-4
MIN_LR=1e-5
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=2
CLIP_GRAD_NORM=3.0
MASK_RATIO=0.5
DECOUPLING_MARGIN=0.1
SEED=42

# Architecture (Table 4)
K_T=32
K_C=32
K_R=32
D_MODEL=32
NHEAD=32
D_FF=64
N_ENCODER_LAYERS=4
N_DECODER_LAYERS=2
PATCH_LEN=20
PATCH_STRIDE=20
DROPOUT=0.1

# ============================================================================
# Functions
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "Directories created: ${LOG_DIR}, ${CHECKPOINT_DIR}"
}

run_pretrain() {
    local config_name=$1
    shift
    local extra_args="$@"

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="BP_pretrain_${config_name}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"
    local save_dir="${CHECKPOINT_DIR}/${config_name}"

    mkdir -p "${save_dir}"

    echo ""
    echo "======================================================================"
    echo "Pre-Training: ${config_name}"
    echo "======================================================================"
    echo "  Config:     ${config_name}"
    echo "  Extra args: ${extra_args}"
    echo "  Save dir:   ${save_dir}"
    echo "  Log:        ${log_file}"
    echo "----------------------------------------------------------------------"

    local cmd="python ${PYTHON_SCRIPT} \
        --data_config ${DATA_CONFIG} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --min_lr ${MIN_LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --clip_grad_norm ${CLIP_GRAD_NORM} \
        --mask_ratio ${MASK_RATIO} \
        --decoupling_margin ${DECOUPLING_MARGIN} \
        --K_T ${K_T} --K_C ${K_C} --K_R ${K_R} \
        --d_model ${D_MODEL} --nhead ${NHEAD} --d_ff ${D_FF} \
        --n_encoder_layers ${N_ENCODER_LAYERS} \
        --n_decoder_layers ${N_DECODER_LAYERS} \
        --patch_len ${PATCH_LEN} --patch_stride ${PATCH_STRIDE} \
        --dropout ${DROPOUT} \
        --save_dir ${save_dir} \
        --save_every 10 \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${extra_args}"

    echo "Starting pre-training..."
    cd "${PROJECT_DIR}"
    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo "Pre-training completed: ${run_name}"
        echo "Checkpoints saved to: ${save_dir}"
        return 0
    else
        echo "Pre-training FAILED: ${run_name}"
        return 1
    fi
}

# ============================================================================
# Pre-training configurations
# ============================================================================

run_full() {
    echo "Running: Full pre-training"
    run_pretrain "full" ""
}

run_no_masking() {
    echo "Running: w/o masking ablation"
    run_pretrain "no_masking" "--no_masking"
}

run_no_reconstruction() {
    echo "Running: w/o reconstruction ablation"
    run_pretrain "no_reconstruction" "--no_reconstruction"
}

run_no_decoupling() {
    echo "Running: w/o decoupling ablation"
    run_pretrain "no_decoupling" "--no_decoupling"
}

run_random_retrieval() {
    echo "Running: w random retrieval ablation"
    run_pretrain "random_retrieval" "--random_retrieval"
}

run_all_ablations() {
    echo "Running all 5 pre-training configurations..."
    echo ""

    local total=0
    local success=0
    local fail=0

    for config in "full" "no_masking" "no_reconstruction" "no_decoupling" "random_retrieval"; do
        ((total++))
        case "${config}" in
            full)               run_full && ((success++)) || ((fail++)) ;;
            no_masking)         run_no_masking && ((success++)) || ((fail++)) ;;
            no_reconstruction)  run_no_reconstruction && ((success++)) || ((fail++)) ;;
            no_decoupling)      run_no_decoupling && ((success++)) || ((fail++)) ;;
            random_retrieval)   run_random_retrieval && ((success++)) || ((fail++)) ;;
        esac
        echo "Progress: ${success}/${total} passed, ${fail} failed"
    done

    echo ""
    echo "======================================================================"
    echo "Pre-Training Ablation Complete"
    echo "  Total: ${total}, Success: ${success}, Failed: ${fail}"
    echo "======================================================================"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "======================================================================"
    echo "BrainPro Pre-Training"
    echo "======================================================================"
    echo ""
    echo "Architecture: K_T=${K_T}, K_C=${K_C}, K_R=${K_R}, d=${D_MODEL}"
    echo "              nhead=${NHEAD}, d_ff=${D_FF}, enc_layers=${N_ENCODER_LAYERS}"
    echo "              dec_layers=${N_DECODER_LAYERS}, patch=${PATCH_LEN}x${PATCH_STRIDE}"
    echo ""
    echo "Training:     epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"
    echo "              min_lr=${MIN_LR}, warmup=${WARMUP_EPOCHS}"
    echo "              weight_decay=${WEIGHT_DECAY}, clip_norm=${CLIP_GRAD_NORM}"
    echo "              mask_ratio=${MASK_RATIO}, margin=${DECOUPLING_MARGIN}"
    echo ""
    echo "Data config:  ${DATA_CONFIG}"
    echo "======================================================================"
    echo ""

    setup_directories

    # Check data config exists
    if [ ! -f "${DATA_CONFIG}" ]; then
        echo "WARNING: Data config not found: ${DATA_CONFIG}"
        echo "Please create a JSON config file with the following format:"
        echo '  {'
        echo '    "affect": [{"path": "/path/to/data.lmdb", "n_channels": 62, "sampling_rate": 200}],'
        echo '    "motor": [...],'
        echo '    "others": [...]'
        echo '  }'
        echo ""
        echo "Proceeding anyway (will fail if data config is required)..."
        echo ""
    fi

    local command="${1:-full}"

    case "${command}" in
        full)               run_full ;;
        no_masking)         run_no_masking ;;
        no_reconstruction)  run_no_reconstruction ;;
        no_decoupling)      run_no_decoupling ;;
        random_retrieval)   run_random_retrieval ;;
        ablation|all)       run_all_ablations ;;
        *)
            echo "Unknown config: ${command}"
            show_usage
            exit 1
            ;;
    esac
}

show_usage() {
    echo "Usage: $0 [CONFIG]"
    echo ""
    echo "BrainPro pre-training with ablation study"
    echo ""
    echo "Configs:"
    echo "  full               Full pre-training (default)"
    echo "  no_masking         w/o masking ablation"
    echo "  no_reconstruction  w/o reconstruction ablation"
    echo "  no_decoupling      w/o decoupling ablation"
    echo "  random_retrieval   w random retrieval ablation"
    echo "  ablation           Run all 5 configurations sequentially"
    echo ""
    echo "Examples:"
    echo "  $0                 # Full pre-training"
    echo "  $0 ablation        # All 5 ablation configs"
    echo "  $0 no_masking      # Only the no-masking ablation"
    echo ""
    echo "IMPORTANT: Set DATA_CONFIG to point to your pretrain_config.json"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
