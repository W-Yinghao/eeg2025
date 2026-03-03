#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# SCOPE Fine-Tuning: Multiple Backbones on TUAB & TUEV
#
# Structured Prototype-Guided Adaptation with frozen backbone + ProAdapter
# Supports: CodeBrain, CBraMod, LUNA, FEMBA
#
# Usage:
#   ./run_scope_finetune.sh                         # All experiments
#   ./run_scope_finetune.sh codebrain               # CodeBrain on TUEV + TUAB
#   ./run_scope_finetune.sh cbramod                 # CBraMod on TUEV + TUAB
#   ./run_scope_finetune.sh luna                    # LUNA on TUEV + TUAB
#   ./run_scope_finetune.sh femba                   # FEMBA on TUEV + TUAB
#   ./run_scope_finetune.sh codebrain TUEV          # CodeBrain on TUEV only
#   ./run_scope_finetune.sh luna TUAB               # LUNA on TUAB only
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_scope.py"
LOG_DIR="${PROJECT_DIR}/logs_scope_finetune"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_scope_finetune"

# WandB
WANDB_PROJECT="scope-finetune"

# GPU
CUDA_DEVICE=0
SEED=42

# Stage 1: Supervision Construction (shared across all configs, from Table 15)
LABEL_RATIO=0.3
TPN_EPOCHS=50
TPN_LR=5e-4
LAMBDA_ETF=0.1
NUM_PROTOTYPES=3
PROTO_EPOCHS=50
CONFIDENCE_THRESHOLD=0.5

# Stage 2: ProAdapter (shared)
ADAPTER_LAYERS=3
WARMUP_EPOCHS=10
MIN_LR=1e-6
PSEUDO_RATIO=2.0
PATIENCE=15

# ============================================================================
# Per-model per-dataset hyperparameters (from paper Tables 16 & 17)
# Naming: <MODEL>_<DATASET>_<PARAM>
# ============================================================================

# --- CodeBrain ---
# TUEV (multiclass, Table 17)
codebrain_TUEV_EPOCHS=60
codebrain_TUEV_BATCH_SIZE=64
codebrain_TUEV_LR=5e-4
codebrain_TUEV_WEIGHT_DECAY=0.01
codebrain_TUEV_DROPOUT=0.1
codebrain_TUEV_N_LAYER=8

# TUAB (binary, Table 16)
codebrain_TUAB_EPOCHS=30
codebrain_TUAB_BATCH_SIZE=16
codebrain_TUAB_LR=5e-5
codebrain_TUAB_WEIGHT_DECAY=0.001
codebrain_TUAB_DROPOUT=0.1
codebrain_TUAB_N_LAYER=8

# --- CBraMod ---
# TUEV (multiclass, Table 17)
cbramod_TUEV_EPOCHS=60
cbramod_TUEV_BATCH_SIZE=64
cbramod_TUEV_LR=1e-4
cbramod_TUEV_WEIGHT_DECAY=0.01
cbramod_TUEV_DROPOUT=0.1
cbramod_TUEV_N_LAYER=12

# TUAB (binary, Table 16)
cbramod_TUAB_EPOCHS=50
cbramod_TUAB_BATCH_SIZE=64
cbramod_TUAB_LR=1e-4
cbramod_TUAB_WEIGHT_DECAY=0.001
cbramod_TUAB_DROPOUT=0.1
cbramod_TUAB_N_LAYER=12

# --- LUNA ---
luna_TUEV_EPOCHS=60
luna_TUEV_BATCH_SIZE=64
luna_TUEV_LR=5e-4
luna_TUEV_WEIGHT_DECAY=0.01
luna_TUEV_DROPOUT=0.1
luna_TUEV_N_LAYER=8

luna_TUAB_EPOCHS=30
luna_TUAB_BATCH_SIZE=16
luna_TUAB_LR=5e-5
luna_TUAB_WEIGHT_DECAY=0.001
luna_TUAB_DROPOUT=0.1
luna_TUAB_N_LAYER=8

# --- FEMBA ---
femba_TUEV_EPOCHS=60
femba_TUEV_BATCH_SIZE=64
femba_TUEV_LR=5e-4
femba_TUEV_WEIGHT_DECAY=0.01
femba_TUEV_DROPOUT=0.1
femba_TUEV_N_LAYER=2

femba_TUAB_EPOCHS=30
femba_TUAB_BATCH_SIZE=16
femba_TUAB_LR=5e-5
femba_TUAB_WEIGHT_DECAY=0.001
femba_TUAB_DROPOUT=0.1
femba_TUAB_N_LAYER=2

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

    # Per-model per-dataset hyperparameters
    local epochs=$(get_param "${model}" "${dataset}" "EPOCHS")
    local batch_size=$(get_param "${model}" "${dataset}" "BATCH_SIZE")
    local lr=$(get_param "${model}" "${dataset}" "LR")
    local weight_decay=$(get_param "${model}" "${dataset}" "WEIGHT_DECAY")
    local dropout=$(get_param "${model}" "${dataset}" "DROPOUT")
    local n_layer=$(get_param "${model}" "${dataset}" "N_LAYER")

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="SCOPE_${model}_${dataset}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: SCOPE - ${model} on ${dataset}"
    echo "======================================================================"
    echo "  Model:        ${model}"
    echo "  Dataset:      ${dataset}"
    echo ""
    echo "  Stage 1 (Supervision Construction):"
    echo "    - TPN epochs:     ${TPN_EPOCHS}"
    echo "    - TPN lr:         ${TPN_LR}"
    echo "    - lambda_etf:     ${LAMBDA_ETF}"
    echo "    - num_prototypes: ${NUM_PROTOTYPES}"
    echo "    - conf_threshold: ${CONFIDENCE_THRESHOLD}"
    echo ""
    echo "  Stage 2 (ProAdapter):"
    echo "    - epochs:       ${epochs}"
    echo "    - batch_size:   ${batch_size}"
    echo "    - lr:           ${lr}"
    echo "    - weight_decay: ${weight_decay}"
    echo "    - dropout:      ${dropout}"
    echo "    - adapter_L:    ${ADAPTER_LAYERS}"
    echo "    - warmup:       ${WARMUP_EPOCHS}"
    echo "    - pseudo_ratio: ${PSEUDO_RATIO}"
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
        --dataset ${dataset} \
        --model ${model} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --label_ratio ${LABEL_RATIO} \
        --n_layer ${n_layer} \
        --tpn_epochs ${TPN_EPOCHS} \
        --tpn_lr ${TPN_LR} \
        --lambda_etf ${LAMBDA_ETF} \
        --num_prototypes ${NUM_PROTOTYPES} \
        --proto_epochs ${PROTO_EPOCHS} \
        --confidence_threshold ${CONFIDENCE_THRESHOLD} \
        --adapter_layers ${ADAPTER_LAYERS} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --min_lr ${MIN_LR} \
        --weight_decay ${weight_decay} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --pseudo_ratio ${PSEUDO_RATIO} \
        --dropout ${dropout} \
        --patience ${PATIENCE} \
        --save_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        --wandb_group ${wandb_group} \
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
    echo "SCOPE Fine-Tuning: Multiple Backbones on TUAB & TUEV"
    echo "======================================================================"
    echo ""
    echo "Per-model per-dataset hyperparameters:"
    echo ""
    echo "  CodeBrain + TUEV: epochs=60, bs=64,  lr=5e-4, wd=0.01"
    echo "  CodeBrain + TUAB: epochs=30, bs=16,  lr=5e-5, wd=0.001"
    echo "  CBraMod   + TUEV: epochs=60, bs=64,  lr=1e-4, wd=0.01"
    echo "  CBraMod   + TUAB: epochs=50, bs=64,  lr=1e-4, wd=0.001"
    echo "  LUNA      + TUEV: epochs=60, bs=64,  lr=5e-4, wd=0.01"
    echo "  LUNA      + TUAB: epochs=30, bs=16,  lr=5e-5, wd=0.001"
    echo "  FEMBA     + TUEV: epochs=60, bs=64,  lr=5e-4, wd=0.01"
    echo "  FEMBA     + TUAB: epochs=30, bs=16,  lr=5e-5, wd=0.001"
    echo ""
    echo "Stage 1: TPN(${TPN_EPOCHS}ep) + Proto(M=${NUM_PROTOTYPES}) + Fusion(rho=${CONFIDENCE_THRESHOLD})"
    echo "Stage 2: ProAdapter(L=${ADAPTER_LAYERS}), warmup=${WARMUP_EPOCHS}, pseudo_ratio=${PSEUDO_RATIO}"
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

    local SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    for model in "${models[@]}"; do
        wandb_group="SCOPE_${model}_${SWEEP_TIMESTAMP}"

        for dataset in "${datasets[@]}"; do
            total=$((total + 1))

            if run_experiment "${model}" "${dataset}"; then
                success=$((success + 1))
            else
                fail=$((fail + 1))
            fi

            echo ""
            echo "Progress: ${success}/${total} passed, ${fail} failed"
            echo ""
        done
    done

    # Summary
    echo ""
    echo "======================================================================"
    echo "SCOPE Fine-Tuning Complete"
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
    echo "SCOPE fine-tuning with multiple backbones"
    echo ""
    echo "Arguments:"
    echo "  MODEL     (optional) codebrain, cbramod, luna, or femba"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo ""
    echo "Examples:"
    echo "  $0                          # All experiments (4 models x 2 datasets)"
    echo "  $0 codebrain                # CodeBrain on TUEV + TUAB"
    echo "  $0 luna                     # LUNA on TUEV + TUAB"
    echo "  $0 femba TUEV               # FEMBA on TUEV only"
    echo "  $0 cbramod TUAB             # CBraMod on TUAB only"
    echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

main "$@"
