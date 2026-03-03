#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# MI Fine-Tuning: VIB + InfoNCE with Multiple Backbones
#
# Frozen backbone + trainable MI head (VIB noise suppression + InfoNCE alignment)
# Supports: CodeBrain (SSSM), CBraMod (Transformer), LUNA, FEMBA
# Ablation over alpha (InfoNCE weight) and beta (VIB weight)
#
# Usage:
#   ./run_mi_finetune.sh                           # All experiments
#   ./run_mi_finetune.sh codebrain                 # CodeBrain only
#   ./run_mi_finetune.sh cbramod                   # CBraMod only
#   ./run_mi_finetune.sh luna                      # LUNA only
#   ./run_mi_finetune.sh femba                     # FEMBA only
#   ./run_mi_finetune.sh codebrain TUEV            # CodeBrain + TUEV only
#   ./run_mi_finetune.sh luna TUAB                 # LUNA + TUAB only
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_mi_finetuning.py"
LOG_DIR="${PROJECT_DIR}/logs_mi_finetune"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_mi_finetune"

# WandB
WANDB_PROJECT="eeg-mi-finetuning"

# GPU
CUDA_DEVICE=0
SEED=3407

# MI-specific defaults
VIB_DIM=128
HIDDEN_DIM=256
TEMPERATURE=0.07
EXPERT_FEATURE="psd"
CLIP_VALUE=1.0
DROPOUT=0.3
LABEL_SMOOTHING=0.1
PATIENCE=10

# ============================================================================
# Per-model per-dataset hyperparameters
# ============================================================================

# --- CodeBrain ---
# TUEV (multiclass)
codebrain_TUEV_EPOCHS=50
codebrain_TUEV_BATCH_SIZE=64
codebrain_TUEV_LR=2e-5
codebrain_TUEV_WEIGHT_DECAY=5e-4
codebrain_TUEV_N_LAYER=8

# TUAB (binary) — increased regularization to prevent early overfitting (best_epoch=3)
codebrain_TUAB_EPOCHS=50
codebrain_TUAB_BATCH_SIZE=64
codebrain_TUAB_LR=1e-5
codebrain_TUAB_WEIGHT_DECAY=5e-3
codebrain_TUAB_N_LAYER=8

# --- CBraMod ---
# TUEV (multiclass)
cbramod_TUEV_EPOCHS=50
cbramod_TUEV_BATCH_SIZE=64
cbramod_TUEV_LR=2e-5
cbramod_TUEV_WEIGHT_DECAY=5e-4
cbramod_TUEV_N_LAYER=12

# TUAB (binary) — increased regularization to prevent early overfitting
cbramod_TUAB_EPOCHS=50
cbramod_TUAB_BATCH_SIZE=64
cbramod_TUAB_LR=1e-5
cbramod_TUAB_WEIGHT_DECAY=5e-3
cbramod_TUAB_N_LAYER=12

# --- LUNA ---
luna_TUEV_EPOCHS=50
luna_TUEV_BATCH_SIZE=64
luna_TUEV_LR=2e-5
luna_TUEV_WEIGHT_DECAY=5e-4
luna_TUEV_N_LAYER=8

luna_TUAB_EPOCHS=50
luna_TUAB_BATCH_SIZE=64
luna_TUAB_LR=1e-5
luna_TUAB_WEIGHT_DECAY=5e-3
luna_TUAB_N_LAYER=8

# --- FEMBA ---
femba_TUEV_EPOCHS=50
femba_TUEV_BATCH_SIZE=64
femba_TUEV_LR=2e-5
femba_TUEV_WEIGHT_DECAY=5e-4
femba_TUEV_N_LAYER=2

femba_TUAB_EPOCHS=50
femba_TUAB_BATCH_SIZE=64
femba_TUAB_LR=1e-5
femba_TUAB_WEIGHT_DECAY=5e-3
femba_TUAB_N_LAYER=2

# ============================================================================
# Ablation configs: "alpha|beta|description"
# alpha: InfoNCE weight, beta: VIB weight
# ============================================================================

declare -a MI_CONFIGS=(
    "1.0|1e-3|full_vib_nce"
    "1.0|0.0|nce_only"
    "0.0|1e-3|vib_only"
    "0.0|0.0|baseline_ce"
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

run_experiment() {
    local model=$1
    local dataset=$2
    local alpha=$3
    local beta=$4
    local config_desc=$5

    # Per-model per-dataset hyperparameters
    local epochs=$(get_param "${model}" "${dataset}" "EPOCHS")
    local batch_size=$(get_param "${model}" "${dataset}" "BATCH_SIZE")
    local lr=$(get_param "${model}" "${dataset}" "LR")
    local weight_decay=$(get_param "${model}" "${dataset}" "WEIGHT_DECAY")
    local n_layer=$(get_param "${model}" "${dataset}" "N_LAYER")

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="MI_${model}_${dataset}_a${alpha}_b${beta}_${config_desc}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: MI ${config_desc} - ${model} on ${dataset}"
    echo "======================================================================"
    echo "  Model:        ${model}"
    echo "  Dataset:      ${dataset}"
    echo "  alpha (NCE):  ${alpha}"
    echo "  beta (VIB):   ${beta}"
    echo "  expert:       ${EXPERT_FEATURE}"
    echo ""
    echo "  Params:"
    echo "    - epochs:       ${epochs}"
    echo "    - batch_size:   ${batch_size}"
    echo "    - lr:           ${lr}"
    echo "    - weight_decay: ${weight_decay}"
    echo "    - vib_dim:      ${VIB_DIM}"
    echo "    - hidden_dim:   ${HIDDEN_DIM}"
    echo "    - temperature:  ${TEMPERATURE}"
    echo ""
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # Model-specific extra args
    local extra_args=""
    if [ "${model}" = "cbramod" ]; then
        extra_args="--n_layer_cbramod ${n_layer} --dim_feedforward 800 --nhead 8"
    elif [ "${model}" = "luna" ]; then
        extra_args="--luna_size base"
    fi

    local cmd="python ${PYTHON_SCRIPT} \
        --model ${model} \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --n_layer ${n_layer} \
        --alpha ${alpha} \
        --beta ${beta} \
        --temperature ${TEMPERATURE} \
        --vib_dim ${VIB_DIM} \
        --hidden_dim ${HIDDEN_DIM} \
        --expert_feature ${EXPERT_FEATURE} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --clip_value ${CLIP_VALUE} \
        --dropout ${DROPOUT} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --patience ${PATIENCE} \
        --model_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
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
# Main
# ============================================================================

main() {
    echo "======================================================================"
    echo "MI Fine-Tuning: VIB + InfoNCE"
    echo "======================================================================"
    echo ""
    echo "Models: CodeBrain (SSSM) + CBraMod + LUNA + FEMBA"
    echo "Datasets: TUEV (multiclass) + TUAB (binary)"
    echo ""
    echo "MI configs (${#MI_CONFIGS[@]} per model-dataset pair):"
    echo "  full_vib_nce: alpha=1.0, beta=1e-3 (full model)"
    echo "  nce_only:     alpha=1.0, beta=0    (InfoNCE only)"
    echo "  vib_only:     alpha=0,   beta=1e-3 (VIB only)"
    echo "  baseline_ce:  alpha=0,   beta=0    (CE baseline)"
    echo ""
    echo "Per-model per-dataset hyperparameters:"
    echo "  CodeBrain + TUEV: lr=2e-5, wd=5e-4, bs=64,  epochs=50"
    echo "  CodeBrain + TUAB: lr=1e-5, wd=5e-5, bs=64,  epochs=50"
    echo "  CBraMod   + TUEV: lr=2e-5, wd=5e-4, bs=64,  epochs=50"
    echo "  CBraMod   + TUAB: lr=1e-5, wd=5e-5, bs=64,  epochs=50"
    echo "======================================================================"
    echo ""

    setup_directories

    # Parse arguments
    local target_model=""
    local target_dataset=""
    if [ $# -ge 1 ]; then
        target_model="$1"
    fi
    if [ $# -ge 2 ]; then
        target_dataset="$2"
    fi

    local models=("codebrain" "cbramod" "luna" "femba")
    local datasets=("TUEV" "TUAB")

    if [ -n "${target_model}" ]; then
        models=("${target_model}")
        echo "Running only model: ${target_model}"
    fi
    if [ -n "${target_dataset}" ]; then
        datasets=("${target_dataset}")
        echo "Running only dataset: ${target_dataset}"
    fi
    echo ""

    local total=0
    local success=0
    local fail=0

    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for mi_config in "${MI_CONFIGS[@]}"; do
                IFS='|' read -r alpha beta config_desc <<< "${mi_config}"

                total=$((total + 1))

                if run_experiment "${model}" "${dataset}" "${alpha}" "${beta}" "${config_desc}"; then
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
    echo "MI Fine-Tuning Complete"
    echo "======================================================================"
    echo "  Total:      ${total}"
    echo "  Successful: ${success}"
    echo "  Failed:     ${fail}"
    echo ""
    echo "Results:"
    echo "  Logs:        ${LOG_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  WandB:       ${WANDB_PROJECT}"
    echo ""
    echo "Ablation configs:"
    echo "  1. full_vib_nce: alpha=1.0, beta=1e-3 (full model)"
    echo "  2. nce_only:     alpha=1.0, beta=0    (InfoNCE only)"
    echo "  3. vib_only:     alpha=0,   beta=1e-3 (VIB only)"
    echo "  4. baseline_ce:  alpha=0,   beta=0    (CE baseline)"
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [MODEL] [DATASET]"
    echo ""
    echo "MI fine-tuning with VIB + InfoNCE (frozen backbone)"
    echo ""
    echo "Arguments:"
    echo "  MODEL     (optional) codebrain, cbramod, luna, or femba"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo ""
    echo "Examples:"
    echo "  $0                          # All (4 models x 2 datasets x 4 configs = 32 experiments)"
    echo "  $0 codebrain                # CodeBrain on TUEV + TUAB (8 experiments)"
    echo "  $0 luna TUEV                # LUNA on TUEV only (4 experiments)"
    echo "  $0 femba TUAB               # FEMBA on TUAB only (4 experiments)"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
