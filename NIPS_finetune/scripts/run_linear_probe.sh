#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# Linear Probe Fine-Tuning with CodeBrain Backbone
#
# Frozen CodeBrain (SSSM) backbone + trainable linear/MLP classifier head
# Ablation study over classifier architectures and learning rates
#
# Usage:
#   ./run_linear_probe.sh                  # Run all experiments (2 datasets x 4 classifiers)
#   ./run_linear_probe.sh TUEV             # Only TUEV
#   ./run_linear_probe.sh TUAB             # Only TUAB
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_tuev_lmdb.py"
LOG_DIR="${PROJECT_DIR}/logs_linear_probe"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_linear_probe"

# WandB
WANDB_PROJECT="codebrain-linear-probe"

# GPU
CUDA_DEVICE=0

# Training parameters
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=0.01
WEIGHT_DECAY=5e-2
DROPOUT=0.1
CLIP_VALUE=1.0
LABEL_SMOOTHING=0.1
SEED=3407

# CodeBrain backbone
N_LAYER=8
CODEBRAIN_WEIGHTS="${PROJECT_DIR}/CodeBrain/Checkpoints/CodeBrain.pth"

# t-SNE
TSNE_INTERVAL=10
TSNE_SAMPLES=2000

# ============================================================================
# Experiment configs
# ============================================================================

DATASETS=("TUEV" "TUAB")

# Classifier ablation: "classifier_type|lr|description"
declare -a CLASSIFIER_CONFIGS=(
    "all_patch_reps_onelayer|0.01|linear"
    "avgpooling_patch_reps|0.01|avgpool_linear"
    "all_patch_reps_twolayer|0.01|mlp_2layer"
    "all_patch_reps|0.01|mlp_3layer"
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
    local dataset=$1
    local classifier=$2
    local lr=$3
    local config_desc=$4
    local wandb_group=$5

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="LP_${dataset}_${config_desc}_lr${lr}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: Linear Probe - ${config_desc}"
    echo "======================================================================"
    echo "  Dataset:    ${dataset}"
    echo "  Classifier: ${classifier}"
    echo "  LR:         ${lr}"
    echo ""
    echo "  Training params:"
    echo "    - epochs:          ${EPOCHS}"
    echo "    - batch_size:      ${BATCH_SIZE}"
    echo "    - weight_decay:    ${WEIGHT_DECAY}"
    echo "    - label_smoothing: ${LABEL_SMOOTHING}"
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
        --batch_size ${BATCH_SIZE} \
        --lr ${lr} \
        --weight_decay ${WEIGHT_DECAY} \
        --clip_value ${CLIP_VALUE} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --dropout ${DROPOUT} \
        --n_layer ${N_LAYER} \
        --classifier ${classifier} \
        --linear_probe \
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
    echo "Linear Probe Fine-Tuning with CodeBrain"
    echo "======================================================================"
    echo ""
    echo "Experiment configuration:"
    echo "  Datasets:          ${#DATASETS[@]}"
    echo "  Classifier configs: ${#CLASSIFIER_CONFIGS[@]}"
    echo "  Total runs:         $((${#DATASETS[@]} * ${#CLASSIFIER_CONFIGS[@]}))"
    echo ""
    echo "Model: Frozen CodeBrain (SSSM) + Trainable Classifier Head"
    echo "  backbone layers: ${N_LAYER}"
    echo ""
    echo "Training: epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"
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

    SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    for dataset in "${target_datasets[@]}"; do
        local wandb_group="LP_${dataset}_${SWEEP_TIMESTAMP}"
        echo "WandB group: ${wandb_group}"

        for config in "${CLASSIFIER_CONFIGS[@]}"; do
            IFS='|' read -r classifier lr config_desc <<< "${config}"

            total_experiments=$((total_experiments + 1))

            if run_experiment "${dataset}" "${classifier}" "${lr}" "${config_desc}" "${wandb_group}"; then
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
    echo "Linear Probe Study Complete"
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
    echo "Classifier configs:"
    echo "  1. linear:        all_patch_reps_onelayer  (single linear layer)"
    echo "  2. avgpool_linear: avgpooling_patch_reps   (avgpool + linear)"
    echo "  3. mlp_2layer:    all_patch_reps_twolayer  (2-layer MLP)"
    echo "  4. mlp_3layer:    all_patch_reps           (3-layer MLP)"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [DATASET]"
    echo ""
    echo "Linear probe fine-tuning with frozen CodeBrain backbone"
    echo ""
    echo "Arguments:"
    echo "  DATASET   (optional) Specific dataset: TUEV or TUAB"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all (2 datasets x 4 classifiers = 8 experiments)"
    echo "  $0 TUEV         # Run TUEV only (4 experiments)"
    echo "  $0 TUAB         # Run TUAB only (4 experiments)"
    echo ""
    echo "Classifier ablation:"
    echo "  linear:         single linear layer on flattened features"
    echo "  avgpool_linear: global avg pool + linear"
    echo "  mlp_2layer:     2-layer MLP head"
    echo "  mlp_3layer:     3-layer MLP head"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
