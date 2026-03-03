#!/bin/bash

################################################################################
# MSFT CBraMod Ablation Study Runner
#
# 这个脚本提供多种方式运行MSFT CBraMod的ablation study:
#   1. 单个实验运行
#   2. 本地批量运行（顺序或并行）
#   3. WandB Sweep（Grid或Bayesian搜索）
#
# 使用方法:
#   ./run_ablation_study.sh single          # 运行单个测试实验
#   ./run_ablation_study.sh local           # 本地顺序运行所有核心配置
#   ./run_ablation_study.sh parallel        # 本地并行运行（需要多GPU）
#   ./run_ablation_study.sh sweep_grid      # 启动WandB Grid Sweep
#   ./run_ablation_study.sh sweep_bayes     # 启动WandB Bayesian Sweep
#   ./run_ablation_study.sh agent           # 启动WandB Sweep Agent
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置变量
# ============================================================================

# GPU设置
CUDA_DEVICE=0
NUM_GPUS=1  # 如果要并行运行，设置为可用GPU数量

# 基本路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/finetune_msft_improved.py"
LOG_DIR="${SCRIPT_DIR}/logs"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"

# WandB配置
WANDB_PROJECT="eeg-msft-improved-ablation"
WANDB_ENTITY=""  # 留空或填入你的entity/team名称

# 数据集
DATASETS=("TUEV" "TUAB")

# 模型（当前只支持cbramod）
MODEL="cbramod"

# MSFT变体 (4种组合)
# 格式: "pos_refiner,criss_cross_agg,variant_name"
VARIANTS=(
    "false,false,baseline"
    "true,false,pos_refiner"
    "false,true,criss_cross_agg"
    "true,true,full"
)

# Scale配置
NUM_SCALES_LIST=(2 3 4)

# 超参数
EPOCHS=50
BATCH_SIZES=(32 64)
LEARNING_RATES=(5e-4 1e-3 2e-3)
WEIGHT_DECAYS=(5e-2 1e-1)
DROPOUTS=(0.1 0.2)

# 其他固定参数
SEED=3407
N_LAYER=12
DIM_FF=800
NHEAD=8
LABEL_SMOOTHING=0.1
CLIP_VALUE=1.0

# ============================================================================
# 函数定义
# ============================================================================

# 创建必要的目录
setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    echo "✓ 目录已创建: ${LOG_DIR}, ${CHECKPOINT_DIR}"
}

# 生成run name
generate_run_name() {
    local dataset=$1
    local variant=$2
    local num_scales=$3
    local batch_size=$4
    local lr=$5
    local wd=$6
    local dropout=$7
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    echo "MSFT_${variant}_${MODEL}_${dataset}_s${num_scales}_bs${batch_size}_lr${lr}_wd${wd}_do${dropout}_${timestamp}"
}

# 运行单个实验
run_single_experiment() {
    local dataset=$1
    local use_pos_refiner=$2
    local use_criss_cross_agg=$3
    local variant_name=$4
    local num_scales=$5
    local batch_size=$6
    local lr=$7
    local weight_decay=$8
    local dropout=$9
    local cuda_device=${10:-0}

    local run_name=$(generate_run_name "${dataset}" "${variant_name}" "${num_scales}" \
                                        "${batch_size}" "${lr}" "${weight_decay}" "${dropout}")

    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "======================================================================"
    echo "开始实验: ${run_name}"
    echo "======================================================================"
    echo "  数据集: ${dataset}"
    echo "  变体: ${variant_name} (pos_refiner=${use_pos_refiner}, criss_cross_agg=${use_criss_cross_agg})"
    echo "  Scales: ${num_scales}"
    echo "  超参数: bs=${batch_size}, lr=${lr}, wd=${weight_decay}, dropout=${dropout}"
    echo "  GPU: cuda:${cuda_device}"
    echo "  日志: ${log_file}"
    echo "----------------------------------------------------------------------"

    # 构建命令
    local cmd="python ${PYTHON_SCRIPT} \
        --model ${MODEL} \
        --dataset ${dataset} \
        --cuda ${cuda_device} \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --dropout ${dropout} \
        --label_smoothing ${LABEL_SMOOTHING} \
        --clip_value ${CLIP_VALUE} \
        --num_scales ${num_scales} \
        --n_layer ${N_LAYER} \
        --dim_feedforward ${DIM_FF} \
        --nhead ${NHEAD} \
        --model_dir ${CHECKPOINT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${run_name}"

    # 添加entity（如果设置了）
    if [ -n "${WANDB_ENTITY}" ]; then
        cmd="${cmd} --wandb_entity ${WANDB_ENTITY}"
    fi

    # 添加MSFT变体参数
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
    echo "执行命令: ${cmd}"
    echo ""

    if eval "${cmd}" 2>&1 | tee "${log_file}"; then
        echo "✓ 实验完成: ${run_name}"
    else
        echo "✗ 实验失败: ${run_name}"
        return 1
    fi
}

# ============================================================================
# 主要功能
# ============================================================================

# 1. 运行单个测试实验
run_single_test() {
    echo "======================================================================"
    echo "运行单个测试实验"
    echo "======================================================================"

    setup_directories

    # 使用full variant, TUEV数据集, 3 scales作为测试
    run_single_experiment "TUEV" "true" "true" "full" 3 64 1e-3 5e-2 0.1 ${CUDA_DEVICE}

    echo ""
    echo "======================================================================"
    echo "测试实验完成！"
    echo "======================================================================"
}

# 2. 本地顺序运行核心配置
run_local_sequential() {
    echo "======================================================================"
    echo "本地顺序运行核心ablation study配置"
    echo "======================================================================"

    setup_directories

    local total_runs=0
    local successful_runs=0
    local failed_runs=0

    # 遍历数据集
    for dataset in "${DATASETS[@]}"; do
        # 遍历变体
        for variant_config in "${VARIANTS[@]}"; do
            IFS=',' read -r use_pos_refiner use_criss_cross_agg variant_name <<< "${variant_config}"

            # 遍历scales
            for num_scales in "${NUM_SCALES_LIST[@]}"; do
                # 使用默认超参数组合
                local batch_size=64
                local lr=1e-3
                local weight_decay=5e-2
                local dropout=0.1

                ((total_runs++))

                if run_single_experiment "${dataset}" "${use_pos_refiner}" "${use_criss_cross_agg}" \
                                        "${variant_name}" "${num_scales}" \
                                        "${batch_size}" "${lr}" "${weight_decay}" "${dropout}" \
                                        ${CUDA_DEVICE}; then
                    ((successful_runs++))
                else
                    ((failed_runs++))
                fi

                echo ""
                echo "进度: ${successful_runs}/${total_runs} 成功, ${failed_runs} 失败"
                echo ""
            done
        done
    done

    echo ""
    echo "======================================================================"
    echo "本地顺序运行完成！"
    echo "  总实验数: ${total_runs}"
    echo "  成功: ${successful_runs}"
    echo "  失败: ${failed_runs}"
    echo "======================================================================"
}

# 3. 本地并行运行（需要多GPU）
run_local_parallel() {
    echo "======================================================================"
    echo "本地并行运行（需要${NUM_GPUS}个GPU）"
    echo "======================================================================"

    if [ ${NUM_GPUS} -lt 2 ]; then
        echo "错误: NUM_GPUS < 2，无法并行运行。请修改脚本中的NUM_GPUS变量。"
        exit 1
    fi

    setup_directories

    echo "警告: 并行运行功能需要手动管理GPU分配。"
    echo "这个示例会生成后台任务，但你需要确保有足够的GPU内存。"
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    local gpu_idx=0

    # 遍历数据集和变体
    for dataset in "${DATASETS[@]}"; do
        for variant_config in "${VARIANTS[@]}"; do
            IFS=',' read -r use_pos_refiner use_criss_cross_agg variant_name <<< "${variant_config}"

            for num_scales in "${NUM_SCALES_LIST[@]}"; do
                local batch_size=64
                local lr=1e-3
                local weight_decay=5e-2
                local dropout=0.1

                # 循环分配GPU
                local cuda_dev=$((gpu_idx % NUM_GPUS))

                echo "在GPU ${cuda_dev}上启动: ${variant_name}, ${dataset}, scales=${num_scales}"

                # 在后台运行
                run_single_experiment "${dataset}" "${use_pos_refiner}" "${use_criss_cross_agg}" \
                                    "${variant_name}" "${num_scales}" \
                                    "${batch_size}" "${lr}" "${weight_decay}" "${dropout}" \
                                    ${cuda_dev} &

                ((gpu_idx++))

                # 控制并发数量
                if [ $((gpu_idx % NUM_GPUS)) -eq 0 ]; then
                    echo "等待当前批次完成..."
                    wait
                fi
            done
        done
    done

    # 等待所有后台任务完成
    wait

    echo ""
    echo "======================================================================"
    echo "并行运行完成！"
    echo "======================================================================"
}

# 4. 启动WandB Grid Sweep
run_wandb_sweep_grid() {
    echo "======================================================================"
    echo "启动WandB Grid Sweep"
    echo "======================================================================"

    local sweep_config="${SCRIPT_DIR}/sweep_msft_cbramod.yaml"

    if [ ! -f "${sweep_config}" ]; then
        echo "错误: Sweep配置文件不存在: ${sweep_config}"
        exit 1
    fi

    echo "Sweep配置: ${sweep_config}"
    echo ""

    # 创建sweep并获取sweep ID
    local sweep_id=$(wandb sweep --project ${WANDB_PROJECT} "${sweep_config}" 2>&1 | grep -oP 'wandb agent \K[^ ]+')

    if [ -z "${sweep_id}" ]; then
        echo "错误: 无法创建sweep"
        exit 1
    fi

    echo ""
    echo "✓ Sweep创建成功！"
    echo "  Sweep ID: ${sweep_id}"
    echo ""
    echo "启动agent的命令:"
    echo "  wandb agent ${sweep_id}"
    echo ""
    echo "或者使用本脚本:"
    echo "  ./run_ablation_study.sh agent ${sweep_id}"
    echo ""
}

# 5. 启动WandB Bayesian Sweep
run_wandb_sweep_bayes() {
    echo "======================================================================"
    echo "启动WandB Bayesian Sweep"
    echo "======================================================================"

    local sweep_config="${SCRIPT_DIR}/sweep_msft_cbramod_bayesian.yaml"

    if [ ! -f "${sweep_config}" ]; then
        echo "错误: Sweep配置文件不存在: ${sweep_config}"
        exit 1
    fi

    echo "Sweep配置: ${sweep_config}"
    echo ""

    # 创建sweep并获取sweep ID
    local sweep_id=$(wandb sweep --project ${WANDB_PROJECT} "${sweep_config}" 2>&1 | grep -oP 'wandb agent \K[^ ]+')

    if [ -z "${sweep_id}" ]; then
        echo "错误: 无法创建sweep"
        exit 1
    fi

    echo ""
    echo "✓ Sweep创建成功！"
    echo "  Sweep ID: ${sweep_id}"
    echo ""
    echo "启动agent的命令:"
    echo "  wandb agent ${sweep_id}"
    echo ""
    echo "或者使用本脚本:"
    echo "  ./run_ablation_study.sh agent ${sweep_id}"
    echo ""
}

# 6. 启动WandB Agent
run_wandb_agent() {
    local sweep_id=$1

    if [ -z "${sweep_id}" ]; then
        echo "错误: 需要提供sweep ID"
        echo "用法: ./run_ablation_study.sh agent <sweep_id>"
        exit 1
    fi

    echo "======================================================================"
    echo "启动WandB Agent"
    echo "======================================================================"
    echo "  Sweep ID: ${sweep_id}"
    echo "  GPU: cuda:${CUDA_DEVICE}"
    echo ""

    # 设置CUDA设备
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

    # 启动agent
    wandb agent ${sweep_id}
}

# ============================================================================
# 主程序入口
# ============================================================================

# 显示使用帮助
show_usage() {
    echo "用法: $0 <command> [options]"
    echo ""
    echo "命令:"
    echo "  single        运行单个测试实验"
    echo "  local         本地顺序运行核心配置（约24个实验）"
    echo "  parallel      本地并行运行（需要多GPU）"
    echo "  sweep_grid    启动WandB Grid Sweep"
    echo "  sweep_bayes   启动WandB Bayesian Sweep"
    echo "  agent <id>    启动WandB Sweep Agent"
    echo ""
    echo "示例:"
    echo "  $0 single"
    echo "  $0 local"
    echo "  $0 sweep_grid"
    echo "  $0 agent user/project/sweep_id"
    echo ""
}

# 主入口
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    local command=$1
    shift

    case "${command}" in
        single)
            run_single_test
            ;;
        local)
            run_local_sequential
            ;;
        parallel)
            run_local_parallel
            ;;
        sweep_grid)
            run_wandb_sweep_grid
            ;;
        sweep_bayes)
            run_wandb_sweep_bayes
            ;;
        agent)
            run_wandb_agent "$@"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "错误: 未知命令 '${command}'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# 运行主程序
main "$@"
