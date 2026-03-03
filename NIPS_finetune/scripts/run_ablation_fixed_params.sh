#!/bin/bash

################################################################################
# MSFT CBraMod Ablation Study - Fixed Parameters
#
# 这个脚本只测试ablation变体，所有其他参数固定为默认值
#
# Ablation维度:
#   - 2个数据集: TUEV, TUAB
#   - 4个变体: baseline, pos_refiner, criss_cross_agg, full
#   - 总计: 2 × 4 = 8 个实验
#
# 使用方法:
#   ./run_ablation_fixed_params.sh           # 本地运行所有实验
#   ./run_ablation_fixed_params.sh TUEV      # 只运行TUEV数据集
################################################################################

set -e

# ============================================================================
# 配置 - 所有参数固定为最优默认值
# ============================================================================

# 路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_msft_improved.py"
LOG_DIR="${PROJECT_DIR}/logs_ablation"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_ablation"

# WandB配置
WANDB_PROJECT="msft-ablation-study"
WANDB_ENTITY=""  # 填入你的entity（如果有）

# GPU设置
CUDA_DEVICE=0

# 固定的训练参数（最优默认值）
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-2
DROPOUT=0.1
LABEL_SMOOTHING=0.1
CLIP_VALUE=1.0
NUM_SCALES=3

# 固定的模型架构
MODEL="cbramod"
N_LAYER=12
DIM_FF=800
NHEAD=8
SEED=3407

# ============================================================================
# 数据集和变体配置
# ============================================================================

# 数据集列表
DATASETS=("TUEV" "TUAB")

# MSFT改进变体
# 格式: "变体名称|use_pos_refiner|use_criss_cross_agg"
declare -a VARIANTS=(
    "baseline|false|false"
    "pos_refiner|true|false"
    "criss_cross_agg|false|true"
    "full|true|true"
)

# ============================================================================
# 函数
# ============================================================================

# 创建目录
setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "✓ 目录已创建"
}

# 运行单个实验
run_experiment() {
    local dataset=$1
    local variant_name=$2
    local use_pos_refiner=$3
    local use_criss_cross_agg=$4

    # 生成run name
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_name="MSFT_${variant_name}_${dataset}_s${NUM_SCALES}_${timestamp}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "运行实验: ${variant_name} on ${dataset}"
    echo "======================================================================"
    echo "  数据集: ${dataset}"
    echo "  变体: ${variant_name}"
    echo "    - use_pos_refiner: ${use_pos_refiner}"
    echo "    - use_criss_cross_agg: ${use_criss_cross_agg}"
    echo "  固定参数:"
    echo "    - epochs: ${EPOCHS}"
    echo "    - batch_size: ${BATCH_SIZE}"
    echo "    - lr: ${LEARNING_RATE}"
    echo "    - weight_decay: ${WEIGHT_DECAY}"
    echo "    - dropout: ${DROPOUT}"
    echo "    - num_scales: ${NUM_SCALES}"
    echo "  日志: ${log_file}"
    echo "----------------------------------------------------------------------"

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
        --label_smoothing ${LABEL_SMOOTHING} \
        --clip_value ${CLIP_VALUE} \
        --num_scales ${NUM_SCALES} \
        --n_layer ${N_LAYER} \
        --dim_feedforward ${DIM_FF} \
        --nhead ${NHEAD} \
        --model_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name}"

    # 添加WandB entity
    if [ -n "${WANDB_ENTITY}" ]; then
        cmd="${cmd} --wandb_entity ${WANDB_ENTITY}"
    fi

    # 添加ablation参数
    if [ "${use_pos_refiner}" = "true" ]; then
        cmd="${cmd} --use_pos_refiner"
    else
        cmd="${cmd} --no_pos_refiner"
    fi

    if [ "${use_criss_cross_agg}" = "true" ]; then
        cmd="${cmd} --use_criss_cross_agg"
    else
        cmd="${cmd} --no_criss_cross_agg"
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
    echo "MSFT CBraMod Ablation Study - 固定参数版本"
    echo "======================================================================"
    echo ""
    echo "实验配置:"
    echo "  数据集数量: ${#DATASETS[@]}"
    echo "  变体数量: ${#VARIANTS[@]}"
    echo "  总实验数: $((${#DATASETS[@]} * ${#VARIANTS[@]}))"
    echo ""
    echo "固定参数:"
    echo "  epochs: ${EPOCHS}"
    echo "  batch_size: ${BATCH_SIZE}"
    echo "  learning_rate: ${LEARNING_RATE}"
    echo "  weight_decay: ${WEIGHT_DECAY}"
    echo "  dropout: ${DROPOUT}"
    echo "  num_scales: ${NUM_SCALES}"
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

    # 遍历数据集和变体
    for dataset in "${target_datasets[@]}"; do
        for variant_config in "${VARIANTS[@]}"; do
            # 解析变体配置
            IFS='|' read -r variant_name use_pos_refiner use_criss_cross_agg <<< "${variant_config}"

            total_experiments=$((total_experiments + 1))

            # 运行实验
            if run_experiment "${dataset}" "${variant_name}" "${use_pos_refiner}" "${use_criss_cross_agg}"; then
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
    echo "Ablation Study 完成"
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
    echo "======================================================================"
}

# 显示帮助
show_usage() {
    echo "用法: $0 [DATASET]"
    echo ""
    echo "运行MSFT CBraMod的ablation study，所有参数固定为默认值"
    echo ""
    echo "参数:"
    echo "  DATASET    (可选) 指定数据集 (TUEV 或 TUAB)"
    echo "             如果不指定，将运行所有数据集"
    echo ""
    echo "示例:"
    echo "  $0              # 运行所有数据集 (TUEV + TUAB) 的4个变体，共8个实验"
    echo "  $0 TUEV         # 只运行TUEV数据集的4个变体"
    echo "  $0 TUAB         # 只运行TUAB数据集的4个变体"
    echo ""
    echo "变体列表:"
    echo "  1. baseline         - 无改进（对照组）"
    echo "  2. pos_refiner      - 只有位置编码refinement"
    echo "  3. criss_cross_agg  - 只有Criss-Cross aggregator"
    echo "  4. full             - 所有改进"
    echo ""
}

# 检查参数
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# 运行主程序
main "$@"
