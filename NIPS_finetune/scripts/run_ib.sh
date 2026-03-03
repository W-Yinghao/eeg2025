#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

################################################################################
# IB + Disentanglement Fine-Tuning with Multiple Backbones
#
# 使用frozen backbone进行IB + Disentanglement微调
# 支持: CodeBrain (SSSM), CBraMod, LUNA, FEMBA
# 包含: token-level IB, GRL adversarial subject removal, interpretability heatmap
#
# 使用方法:
#   ./run_ib.sh                              # 运行所有实验 (所有backbone x 所有数据集 x 所有配置)
#   ./run_ib.sh codebrain                    # 只运行CodeBrain backbone
#   ./run_ib.sh luna                         # 只运行LUNA backbone
#   ./run_ib.sh femba                        # 只运行FEMBA backbone
#   ./run_ib.sh codebrain TUEV              # CodeBrain + TUEV
#   ./run_ib.sh codebrain TUAB batch1       # CodeBrain + TUAB, configs 1-4
#   ./run_ib.sh codebrain TUAB batch2       # CodeBrain + TUAB, configs 5-8
#   ./run_ib.sh test                         # 只运行测试脚本
################################################################################

set -e

# ============================================================================
# 配置
# ============================================================================

# 路径 (使用绝对路径，避免SLURM环境下BASH_SOURCE解析到/var/spool/slurmd)
PROJECT_DIR="/home/infres/yinwang/eeg2025/NIPS_finetune"
PYTHON_SCRIPT="${PROJECT_DIR}/train_ib_disentangle.py"
TEST_SCRIPT="${PROJECT_DIR}/test_ib_disentangle.py"
LOG_DIR="${PROJECT_DIR}/logs_ib"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_ib"

# WandB配置
WANDB_PROJECT="0303-ib-disentangle"

# GPU设置
CUDA_DEVICE=0

# 固定的训练参数
EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-3
DROPOUT=0.1
CLIP_VALUE=5.0
PATIENCE=15

# IB + Disentanglement参数
LATENT_DIM=128
GRL_GAMMA=10.0
HEATMAP_INTERVAL=10

# Backbone参数
SEED=3407

# Per-backbone settings
CODEBRAIN_N_LAYER=8
CODEBRAIN_WEIGHTS="${PROJECT_DIR}/CodeBrain/Checkpoints/CodeBrain.pth"
CBRAMOD_N_LAYER=12
CBRAMOD_WEIGHTS="${PROJECT_DIR}/../NIPS/Cbramod_pretrained_weights.pth"
LUNA_N_LAYER=8
LUNA_WEIGHTS="${PROJECT_DIR}/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors"
FEMBA_N_LAYER=2
FEMBA_WEIGHTS=""  # FEMBA has no pure pretrained backbone

# Backbones to run
BACKBONES=("codebrain" "cbramod" "luna" "femba")

# ============================================================================
# 实验配置
# ============================================================================

# 数据集
DATASETS=("TUEV" "TUAB")

# Ablation配置: "beta|lambda_adv|description"
# beta: IB (KL) loss weight
# lambda_adv: adversarial subject loss weight
declare -a IB_CONFIGS=(
    "1e-3|0.5|full_ib_adv"
    "1e-3|0.0|ib_only"
    "0.0|0.5|adv_only"
    "0.0|0.0|baseline_ce"
    "1e-2|0.5|high_beta"
    "1e-4|0.5|low_beta"
    "1e-3|1.0|high_lambda"
    "1e-3|0.1|low_lambda"
)

# ============================================================================
# 函数
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "Directories created: ${LOG_DIR}, ${CHECKPOINT_DIR}"
}

run_test() {
    echo "======================================================================"
    echo "Running IB + Disentanglement test script"
    echo "======================================================================"

    local cmd="python ${TEST_SCRIPT} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run ib_test_$(date +%Y%m%d_%H%M%S)"

    echo "Command: ${cmd}"
    echo ""

    cd "${PROJECT_DIR}"
    if eval "${cmd}" 2>&1; then
        echo "Test: PASSED"
        return 0
    else
        echo "Test: FAILED"
        return 1
    fi
}

run_experiment() {
    local backbone=$1
    local dataset=$2
    local beta=$3
    local lambda_adv=$4
    local config_desc=$5
    local wandb_group=$6

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
            weights_path="${FEMBA_WEIGHTS}"
            ;;
    esac

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="IB_${backbone}_${dataset}_b${beta}_l${lambda_adv}_${config_desc}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "Experiment: IB + Disentanglement - ${config_desc}"
    echo "======================================================================"
    echo "  Backbone:   ${backbone} (n_layer=${n_layer})"
    echo "  Dataset:    ${dataset}"
    echo "  beta (IB):  ${beta}"
    echo "  lambda_adv: ${lambda_adv}"
    echo ""
    echo "  Training params:"
    echo "    - epochs:     ${EPOCHS}"
    echo "    - batch_size: ${BATCH_SIZE}"
    echo "    - lr:         ${LEARNING_RATE}"
    echo "    - latent_dim: ${LATENT_DIM}"
    echo "    - patience:   ${PATIENCE}"
    echo ""
    echo "  Log: ${log_file}"
    echo "----------------------------------------------------------------------"

    # 检查预训练权重
    local pretrained_arg=""
    if [ -n "${weights_path}" ] && [ -f "${weights_path}" ]; then
        pretrained_arg="--pretrained_weights ${weights_path}"
        echo "  Pretrained weights: ${weights_path}"
    else
        echo "  WARNING: No pretrained weights found, using random init"
    fi

    # 构建使用subject adversarial的参数
    local subject_arg=""
    if [ "${lambda_adv}" = "0.0" ] || [ "${lambda_adv}" = "0" ]; then
        subject_arg="--no_subjects"
    fi

    # LUNA-specific args
    local extra_args=""
    if [ "${backbone}" = "luna" ]; then
        extra_args="--luna_size base"
    fi

    # 构建命令
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
        --latent_dim ${LATENT_DIM} \
        --beta ${beta} \
        --lambda_adv ${lambda_adv} \
        --grl_gamma ${GRL_GAMMA} \
        --n_layer ${n_layer} \
        --save_dir ${CHECKPOINT_DIR} \
        --heatmap_interval ${HEATMAP_INTERVAL} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        --wandb_group ${wandb_group} \
        ${pretrained_arg} \
        ${subject_arg} \
        ${extra_args}"

    # 运行实验
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
# 主程序
# ============================================================================

main() {
    # Parse arguments: [BACKBONE] [DATASET] [batch1|batch2]
    local target_backbones=("${BACKBONES[@]}")
    local target_datasets=("${DATASETS[@]}")
    local batch_filter=""

    if [ $# -ge 1 ]; then
        if [ "$1" = "test" ]; then
            run_test
            exit $?
        fi
        case "$1" in
            codebrain|cbramod|femba|luna)
                target_backbones=("$1")
                if [ $# -ge 2 ]; then
                    case "$2" in
                        TUEV|TUAB)
                            target_datasets=("$2")
                            ;;
                        batch1|batch2)
                            batch_filter="$2"
                            ;;
                        *)
                            echo "Unknown argument: $2 (expected TUEV, TUAB, batch1, or batch2)"
                            exit 1
                            ;;
                    esac
                fi
                if [ $# -ge 3 ]; then
                    case "$3" in
                        batch1|batch2)
                            batch_filter="$3"
                            ;;
                        *)
                            echo "Unknown argument: $3 (expected batch1 or batch2)"
                            exit 1
                            ;;
                    esac
                fi
                ;;
            TUEV|TUAB)
                target_datasets=("$1")
                if [ $# -ge 2 ]; then
                    case "$2" in
                        batch1|batch2)
                            batch_filter="$2"
                            ;;
                        *)
                            echo "Unknown argument: $2 (expected batch1 or batch2)"
                            exit 1
                            ;;
                    esac
                fi
                ;;
            batch1|batch2)
                batch_filter="$1"
                ;;
            *)
                echo "Unknown argument: $1 (expected codebrain, cbramod, femba, luna, TUEV, TUAB, batch1, or batch2)"
                exit 1
                ;;
        esac
    fi

    # Apply batch filter to IB_CONFIGS
    local selected_configs=()
    if [ "${batch_filter}" = "batch1" ]; then
        # Configs 1-4: full_ib_adv, ib_only, adv_only, baseline_ce
        selected_configs=("${IB_CONFIGS[@]:0:4}")
        echo "Batch filter: batch1 (configs 1-4)"
    elif [ "${batch_filter}" = "batch2" ]; then
        # Configs 5-8: high_beta, low_beta, high_lambda, low_lambda
        selected_configs=("${IB_CONFIGS[@]:4:4}")
        echo "Batch filter: batch2 (configs 5-8)"
    else
        selected_configs=("${IB_CONFIGS[@]}")
    fi

    local total_runs=$(( ${#target_backbones[@]} * ${#target_datasets[@]} * ${#selected_configs[@]} ))

    echo "======================================================================"
    echo "IB + Disentanglement Fine-Tuning"
    echo "======================================================================"
    echo ""
    echo "Experiment configuration:"
    echo "  Backbones:    ${target_backbones[*]}"
    echo "  Datasets:     ${target_datasets[*]}"
    echo "  IB configs:   ${#selected_configs[@]}"
    echo "  Total runs:   ${total_runs}"
    echo ""
    echo "Backbone settings:"
    echo "  codebrain: EEGSSM (n_layer=${CODEBRAIN_N_LAYER})"
    echo "  cbramod:   Transformer (n_layer=${CBRAMOD_N_LAYER})"
    echo "  luna:      LUNA (depth=${LUNA_N_LAYER})"
    echo "  femba:     FEMBA (num_blocks=${FEMBA_N_LAYER})"
    echo ""
    echo "Training: epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"
    echo "======================================================================"
    echo ""

    setup_directories

    total_experiments=0
    successful_experiments=0
    failed_experiments=0

    # 生成本次sweep的group名
    SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # 遍历backbone、数据集和IB配置
    for backbone in "${target_backbones[@]}"; do
        for dataset in "${target_datasets[@]}"; do
            local wandb_group="IB_${backbone}_${dataset}_${SWEEP_TIMESTAMP}"
            echo "WandB group: ${wandb_group}"

            for ib_config in "${selected_configs[@]}"; do
                IFS='|' read -r beta lambda_adv config_desc <<< "${ib_config}"

                total_experiments=$((total_experiments + 1))

                if run_experiment "${backbone}" "${dataset}" "${beta}" "${lambda_adv}" "${config_desc}" "${wandb_group}"; then
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

    # 最终总结
    echo ""
    echo "======================================================================"
    echo "IB + Disentanglement Ablation Study Complete"
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
    echo "======================================================================"
}

show_usage() {
    echo "Usage: $0 [BACKBONE] [DATASET] [batch1|batch2]"
    echo ""
    echo "IB + Disentanglement fine-tuning with multiple backbones"
    echo ""
    echo "Arguments:"
    echo "  BACKBONE  (optional) codebrain, cbramod, femba, or luna"
    echo "  DATASET   (optional) TUEV or TUAB"
    echo "  batch1    (optional) Run configs 1-4: full_ib_adv, ib_only, adv_only, baseline_ce"
    echo "  batch2    (optional) Run configs 5-8: high_beta, low_beta, high_lambda, low_lambda"
    echo "  test      Run the test script only"
    echo ""
    echo "Examples:"
    echo "  $0                              # All backbones x datasets x configs"
    echo "  $0 codebrain                    # CodeBrain only"
    echo "  $0 luna TUEV                    # LUNA + TUEV only"
    echo "  $0 codebrain TUAB batch1       # CodeBrain + TUAB, configs 1-4"
    echo "  $0 codebrain TUAB batch2       # CodeBrain + TUAB, configs 5-8"
    echo "  $0 TUEV batch1                 # All backbones, TUEV, configs 1-4"
    echo "  $0 test                         # Run test script"
    echo ""
}

# 检查参数
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# 运行主程序
main "$@"
