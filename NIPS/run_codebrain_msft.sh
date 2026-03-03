#!/bin/bash

################################################################################
# CodeBrain MSFT - 固定参数Ablation Study
#
# 使用CodeBrain (SSSM) 作为backbone进行MSFT微调
# 测试不同scale配置的效果
#
# 使用方法:
#   ./run_codebrain_msft.sh              # 运行所有实验
#   ./run_codebrain_msft.sh TUEV         # 只运行TUEV数据集
################################################################################

set -e

# ============================================================================
# 配置
# ============================================================================

# 路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/finetune_msft.py"
LOG_DIR="${SCRIPT_DIR}/logs_codebrain"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_codebrain"

# WandB配置
WANDB_PROJECT="codebrain-msft-ablation"
WANDB_ENTITY=""

# GPU设置
CUDA_DEVICE=0

# 固定的训练参数（CodeBrain最优配置）
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-2
DROPOUT=0.1
CLIP_VALUE=1.0

# CodeBrain特定参数
MODEL="codebrain"
N_LAYER=8  # CodeBrain默认8层
CODEBOOK_SIZE_T=4096
CODEBOOK_SIZE_F=4096
SEED=3407

# CodeBrain预训练权重
CODEBRAIN_WEIGHTS="/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"

# ============================================================================
# 实验配置
# ============================================================================

# 数据集
DATASETS=("TUEV" "TUAB")

# 测试不同的scale配置
# 格式: "scale_num|description"
declare -a SCALE_CONFIGS=(
    "1|single_scale"
    "2|two_scales"
    "3|three_scales"
    "4|four_scales"
)

# ============================================================================
# 函数
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "✓ 目录已创建"
}

run_experiment() {
    local dataset=$1
    local num_scales=$2
    local scale_desc=$3

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="CodeBrain_MSFT_${dataset}_s${num_scales}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "运行实验: CodeBrain MSFT - ${scale_desc}"
    echo "======================================================================"
    echo "  模型: CodeBrain (SSSM)"
    echo "  数据集: ${dataset}"
    echo "  Num Scales: ${num_scales}"
    echo "  Scale Description: ${scale_desc}"
    echo ""
    echo "  固定参数:"
    echo "    - epochs: ${EPOCHS}"
    echo "    - batch_size: ${BATCH_SIZE}"
    echo "    - lr: ${LEARNING_RATE}"
    echo "    - weight_decay: ${WEIGHT_DECAY}"
    echo "    - dropout: ${DROPOUT}"
    echo "    - n_layer: ${N_LAYER}"
    echo ""
    echo "  日志: ${log_file}"
    echo "----------------------------------------------------------------------"

    # 检查预训练权重
    if [ ! -f "${CODEBRAIN_WEIGHTS}" ]; then
        echo "⚠️  警告: 预训练权重未找到: ${CODEBRAIN_WEIGHTS}"
        echo "    将使用随机初始化的模型"
        local pretrained_arg="--no_pretrained"
    else
        echo "  预训练权重: ${CODEBRAIN_WEIGHTS}"
        local pretrained_arg="--pretrained_weights ${CODEBRAIN_WEIGHTS}"
    fi

    # 构建命令
    local cmd="python ${PYTHON_SCRIPT} \
        --model ${MODEL} \
        --dataset ${dataset} \
        --cuda ${CUDA_DEVICE} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --dropout ${DROPOUT} \
        --clip_value ${CLIP_VALUE} \
        --n_layer ${N_LAYER} \
        --num_scales ${num_scales} \
        --codebook_size_t ${CODEBOOK_SIZE_T} \
        --codebook_size_f ${CODEBOOK_SIZE_F} \
        --model_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name} \
        ${pretrained_arg}"

    if [ -n "${WANDB_ENTITY}" ]; then
        cmd="${cmd} --wandb_entity ${WANDB_ENTITY}"
    fi

    # 运行实验
    echo "开始训练..."
    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo "✓ 实验完成: ${run_name}"
        return 0
    else
        echo "✗ 实验失败: ${run_name}"
        return 1
    fi
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    echo "======================================================================"
    echo "CodeBrain MSFT Ablation Study"
    echo "======================================================================"
    echo ""
    echo "实验配置:"
    echo "  数据集: ${#DATASETS[@]}"
    echo "  Scale配置: ${#SCALE_CONFIGS[@]}"
    echo "  总实验数: $((${#DATASETS[@]} * ${#SCALE_CONFIGS[@]}))"
    echo ""
    echo "模型配置:"
    echo "  Backbone: CodeBrain (SSSM)"
    echo "  Layers: ${N_LAYER}"
    echo "  Codebook: temporal=${CODEBOOK_SIZE_T}, frequency=${CODEBOOK_SIZE_F}"
    echo ""
    echo "训练参数:"
    echo "  epochs: ${EPOCHS}"
    echo "  batch_size: ${BATCH_SIZE}"
    echo "  learning_rate: ${LEARNING_RATE}"
    echo "  weight_decay: ${WEIGHT_DECAY}"
    echo "  dropout: ${DROPOUT}"
    echo ""
    echo "======================================================================"
    echo ""

    setup_directories

    # 检查是否指定了特定数据集
    target_datasets=("${DATASETS[@]}")
    if [ $# -gt 0 ]; then
        target_datasets=("$1")
        echo "只运行数据集: $1"
        echo ""
    fi

    total_experiments=0
    successful_experiments=0
    failed_experiments=0

    # 遍历数据集和scale配置
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
            echo "进度: ${successful_experiments}/${total_experiments} 成功, ${failed_experiments} 失败"
            echo ""
        done
    done

    # 最终总结
    echo ""
    echo "======================================================================"
    echo "CodeBrain MSFT Ablation Study 完成"
    echo "======================================================================"
    echo "  总实验数: ${total_experiments}"
    echo "  成功: ${successful_experiments}"
    echo "  失败: ${failed_experiments}"
    echo ""
    echo "结果保存在:"
    echo "  日志: ${LOG_DIR}/"
    echo "  模型: ${CHECKPOINT_DIR}/"
    echo ""
    echo "WandB项目: ${WANDB_PROJECT}"
    echo ""
    echo "分析建议:"
    echo "  1. 对比不同scale数量的效果"
    echo "  2. 观察scale权重分布"
    echo "  3. 与CBraMod的MSFT结果对比"
    echo "======================================================================"
}

show_usage() {
    echo "用法: $0 [DATASET]"
    echo ""
    echo "使用CodeBrain (SSSM)进行MSFT微调的ablation study"
    echo ""
    echo "参数:"
    echo "  DATASET    (可选) 指定数据集 (TUEV 或 TUAB)"
    echo "             如果不指定，将运行所有数据集"
    echo ""
    echo "示例:"
    echo "  $0              # 运行所有数据集，测试4种scale配置，共8个实验"
    echo "  $0 TUEV         # 只运行TUEV数据集的4种scale配置"
    echo "  $0 TUAB         # 只运行TUAB数据集的4种scale配置"
    echo ""
    echo "Scale配置:"
    echo "  1. single_scale  - 无多尺度（baseline）"
    echo "  2. two_scales    - 2个尺度 [1x, 2x]"
    echo "  3. three_scales  - 3个尺度 [1x, 2x, 4x]"
    echo "  4. four_scales   - 4个尺度 [1x, 2x, 4x, 8x]"
    echo ""
    echo "注意:"
    echo "  - 确保CodeBrain预训练权重存在: ${CODEBRAIN_WEIGHTS}"
    echo "  - 如果权重不存在，将使用随机初始化"
    echo ""
}

# 检查参数
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# 运行主程序
main "$@"
