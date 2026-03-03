#!/bin/bash

################################################################################
# MSFT CBraMod Ablation Study - 调试版本
#
# 添加了详细的调试输出和dry-run模式
################################################################################

set -e

# 启用调试输出
set -x

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/finetune_msft_improved.py"
LOG_DIR="${SCRIPT_DIR}/logs_ablation"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_ablation"

# 其他配置...
WANDB_PROJECT="msft-ablation-study"
CUDA_DEVICE=0
EPOCHS=2  # 减少epoch用于测试
BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-2
DROPOUT=0.1
NUM_SCALES=3
MODEL="cbramod"
SEED=3407

# 数据集和变体
DATASETS=("TUEV" "TUAB")
declare -a VARIANTS=(
    "baseline|false|false"
    "pos_refiner|true|false"
    "criss_cross_agg|false|true"
    "full|true|true"
)

# ============================================================================
# 主程序
# ============================================================================

echo "======================================================================"
echo "调试模式：检查配置"
echo "======================================================================"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "PYTHON_SCRIPT: ${PYTHON_SCRIPT}"
echo "Python脚本存在: $([ -f "${PYTHON_SCRIPT}" ] && echo "是" || echo "否")"
echo ""
echo "数据集数量: ${#DATASETS[@]}"
echo "数据集列表: ${DATASETS[@]}"
echo ""
echo "变体数量: ${#VARIANTS[@]}"
for i in "${!VARIANTS[@]}"; do
    echo "  变体[$i]: ${VARIANTS[$i]}"
done
echo "======================================================================"
echo ""

# 创建目录
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
echo "✓ 目录已创建"
echo ""

# 测试循环
echo "======================================================================"
echo "测试循环逻辑"
echo "======================================================================"

local_counter=0
for dataset in "${DATASETS[@]}"; do
    echo "数据集: ${dataset}"

    for variant_config in "${VARIANTS[@]}"; do
        IFS='|' read -r variant_name use_pos_refiner use_criss_cross_agg <<< "${variant_config}"

        ((local_counter++))

        echo "  [$local_counter] 变体: ${variant_name}"
        echo "      use_pos_refiner: ${use_pos_refiner}"
        echo "      use_criss_cross_agg: ${use_criss_cross_agg}"

        # Dry run: 只打印命令，不执行
        echo "      命令预览:"
        echo "      python ${PYTHON_SCRIPT} \\"
        echo "        --model ${MODEL} \\"
        echo "        --dataset ${dataset} \\"
        echo "        --cuda ${CUDA_DEVICE} \\"
        echo "        --epochs ${EPOCHS} \\"
        echo "        --batch_size ${BATCH_SIZE} \\"
        echo "        --num_scales ${NUM_SCALES} \\"
        if [ "${use_pos_refiner}" = "true" ]; then
            echo "        --use_pos_refiner \\"
        else
            echo "        --no_pos_refiner \\"
        fi
        if [ "${use_criss_cross_agg}" = "true" ]; then
            echo "        --use_criss_cross_agg"
        else
            echo "        --no_criss_cross_agg"
        fi
        echo ""
    done
done

echo "======================================================================"
echo "循环测试完成"
echo "======================================================================"
echo "预期实验总数: $local_counter"
echo ""
echo "如果看到这条消息，说明脚本逻辑是正确的。"
echo "问题可能出在Python脚本执行上。"
echo ""
echo "建议："
echo "1. 手动运行上面的Python命令测试"
echo "2. 检查Python环境和依赖"
echo "3. 查看详细的错误信息"
echo "======================================================================"
