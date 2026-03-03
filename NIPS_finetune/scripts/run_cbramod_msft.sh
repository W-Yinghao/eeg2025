#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# CBraMod MSFT - Frozen CBraMod backbone + MSFT adapter
#
# 使用CBraMod作为frozen backbone进行MSFT微调
# 测试不同scale配置的效果
#
# Usage:
#   ./run_cbramod_msft.sh              # 运行所有实验 (2 datasets x 4 scales)
#   ./run_cbramod_msft.sh TUEV         # 只运行TUEV
#   ./run_cbramod_msft.sh TUAB         # 只运行TUAB
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_msft.py"
LOG_DIR="${PROJECT_DIR}/logs_cbramod_msft"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_cbramod_msft"

# WandB
WANDB_PROJECT="cbramod-msft-ablation"

# GPU
CUDA_DEVICE=0

# Fixed training params
EPOCHS=30
CLIP_VALUE=1.0
LABEL_SMOOTHING=0.1
SEED=3407

# Per-dataset hyperparameters
# TUEV: 6-class, 5s segments
TUEV_LR=2e-5
TUEV_WEIGHT_DECAY=5e-4
TUEV_DROPOUT=0.4
TUEV_BATCH_SIZE=64

# TUAB: binary, 10s segments
TUAB_LR=1e-5
TUAB_WEIGHT_DECAY=5e-5
TUAB_DROPOUT=0.4
TUAB_BATCH_SIZE=512

# CBraMod backbone params
MODEL="cbramod"
N_LAYER=12
DIM_FEEDFORWARD=800
NHEAD=8

# Pretrained weights (uses default in finetune_msft.py if not specified)
CBRAMOD_WEIGHTS="${PROJECT_DIR}/Cbramod_pretrained_weights.pth"

# ============================================================================
# Experiment configs
# ============================================================================

DATASETS=("TUEV" "TUAB")

# Scale ablation: "scale_num|description"
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

get_dataset_param() {
    local dataset=$1
    local param=$2
    eval echo "\${${dataset}_${param}}"
}

run_experiment() {
    local dataset=$1
    local num_scales=$2
    local scale_desc=$3

    # Per-dataset hyperparameters
    local lr=$(get_dataset_param "${dataset}" "LR")
    local weight_decay=$(get_dataset_param "${dataset}" "WEIGHT_DECAY")
    local dropout=$(get_dataset_param "${dataset}" "DROPOUT")
    local batch_size=$(get_dataset_param "${dataset}" "BATCH_SIZE")

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="CBraMod_MSFT_${dataset}_s${num_scales}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: CBraMod MSFT - ${scale_desc}"
    echo "======================================================================"
    echo "  Model:        CBraMod"
    echo "  Dataset:      ${dataset}"
    echo "  Num Scales:   ${num_scales}"
    echo ""
    echo "  Params:"
    echo "    - epochs:       ${EPOCHS}"
    echo "    - batch_size:   ${batch_size}"
    echo "    - lr:           ${lr}"
    echo "    - weight_decay: ${weight_decay}"
    echo "    - dropout:      ${dropout}"
    echo "    - n_layer:      ${N_LAYER}"
    echo ""
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # Pretrained weights
    local pretrained_arg=""
    if [ -f "${CBRAMOD_WEIGHTS}" ]; then
        pretrained_arg="--pretrained_weights ${CBRAMOD_WEIGHTS}"
        echo "  Pretrained weights: ${CBRAMOD_WEIGHTS}"
    else
        echo "  WARNING: No pretrained weights found at ${CBRAMOD_WEIGHTS}"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --model ${MODEL} \
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
        --n_layer ${N_LAYER} \
        --dim_feedforward ${DIM_FEEDFORWARD} \
        --nhead ${NHEAD} \
        --num_scales ${num_scales} \
        --model_dir ${CHECKPOINT_DIR} \
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
    echo "CBraMod MSFT Ablation Study"
    echo "======================================================================"
    echo ""
    echo "Model: Frozen CBraMod + MSFT adapter"
    echo "  n_layer:         ${N_LAYER}"
    echo "  dim_feedforward:  ${DIM_FEEDFORWARD}"
    echo "  nhead:            ${NHEAD}"
    echo ""
    echo "Experiment config:"
    echo "  Datasets:     ${#DATASETS[@]}"
    echo "  Scale configs: ${#SCALE_CONFIGS[@]}"
    echo "  Total runs:    $((${#DATASETS[@]} * ${#SCALE_CONFIGS[@]}))"
    echo ""
    echo "Per-dataset hyperparameters:"
    echo "  TUEV: lr=${TUEV_LR}, wd=${TUEV_WEIGHT_DECAY}, dropout=${TUEV_DROPOUT}, bs=${TUEV_BATCH_SIZE}"
    echo "  TUAB: lr=${TUAB_LR}, wd=${TUAB_WEIGHT_DECAY}, dropout=${TUAB_DROPOUT}, bs=${TUAB_BATCH_SIZE}"
    echo ""
    echo "Training: epochs=${EPOCHS}, seed=${SEED}"
    echo "======================================================================"
    echo ""

    setup_directories

    # Filter datasets if specified
    target_datasets=("${DATASETS[@]}")
    if [ $# -gt 0 ]; then
        target_datasets=("$1")
        echo "Running only dataset: $1"
        echo ""
    fi

    total_experiments=0
    successful_experiments=0
    failed_experiments=0

    for dataset in "${target_datasets[@]}"; do
        for scale_config in "${SCALE_CONFIGS[@]}"; do
            IFS='|' read -r num_scales scale_desc <<< "${scale_config}"

            total_experiments=$((total_experiments + 1))

            if run_experiment "${dataset}" "${num_scales}" "${scale_desc}"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
            fi

            echo ""
            echo "Progress: ${successful_experiments}/${total_experiments} passed, ${failed_experiments} failed"
            echo ""
        done
    done

    # Summary
    echo ""
    echo "======================================================================"
    echo "CBraMod MSFT Ablation Study Complete"
    echo "======================================================================"
    echo "  Total:      ${total_experiments}"
    echo "  Successful: ${successful_experiments}"
    echo "  Failed:     ${failed_experiments}"
    echo ""
    echo "Results:"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  WandB:       ${WANDB_PROJECT}"
    echo ""
    echo "Scale configs:"
    echo "  1. single_scale  - no multi-scale (baseline)"
    echo "  2. two_scales    - [1x, 2x]"
    echo "  3. three_scales  - [1x, 2x, 4x]"
    echo "  4. four_scales   - [1x, 2x, 4x, 8x]"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [DATASET]"
    echo ""
    echo "CBraMod MSFT fine-tuning (frozen backbone + MSFT adapter)"
    echo ""
    echo "Arguments:"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo ""
    echo "Per-dataset hyperparameters:"
    echo "  TUEV: lr=2e-5, wd=5e-4, dropout=0.4, bs=64"
    echo "  TUAB: lr=1e-5, wd=5e-5, dropout=0.4, bs=512"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all (2 datasets x 4 scales = 8 experiments)"
    echo "  $0 TUEV         # Run TUEV only (4 experiments)"
    echo "  $0 TUAB         # Run TUAB only (4 experiments)"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
