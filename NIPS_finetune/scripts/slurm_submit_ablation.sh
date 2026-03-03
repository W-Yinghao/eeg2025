#!/bin/bash

################################################################################
# MSFT CBraMod Ablation Study - Slurm批量提交脚本
#
# 这个脚本可以批量提交ablation study实验到Slurm队列
# 支持最多4张A100 GPU并行运行
#
# 使用方法:
#   ./slurm_submit_ablation.sh core        # 提交核心配置（24个实验）
#   ./slurm_submit_ablation.sh full        # 提交完整grid搜索（需要大量时间）
#   ./slurm_submit_ablation.sh custom      # 提交自定义配置
#   ./slurm_submit_ablation.sh status      # 查看作业状态
#   ./slurm_submit_ablation.sh cancel      # 取消所有作业
################################################################################

set -e

# ============================================================================
# 配置变量
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_single_experiment.sh"
LOG_DIR="${PROJECT_DIR}/logs"

# Slurm配置
MAX_PARALLEL_JOBS=4  # 最多4张A100并行
PARTITION="gpu"      # GPU分区名称
TIME_LIMIT="12:00:00"  # 每个作业时间限制

# 数据集
DATASETS=("TUEV" "TUAB")

# MSFT变体
VARIANTS=("baseline" "pos_refiner" "criss_cross_agg" "full")

# Scale配置
NUM_SCALES_LIST=(2 3 4)

# 核心超参数（用于core模式）
CORE_BATCH_SIZE=64
CORE_LR=1e-3
CORE_WD=5e-2
CORE_DROPOUT=0.1

# 完整超参数网格（用于full模式）
BATCH_SIZES=(32 64)
LEARNING_RATES=(5e-4 1e-3 2e-3)
WEIGHT_DECAYS=(5e-2 1e-1)
DROPOUTS=(0.1 0.2)

# 作业跟踪
JOB_IDS_FILE="${LOG_DIR}/submitted_job_ids.txt"

# ============================================================================
# 函数定义
# ============================================================================

# 创建目录
setup_directories() {
    mkdir -p "${LOG_DIR}"
    echo "✓ 日志目录已创建: ${LOG_DIR}"
}

# 检查Slurm脚本存在
check_slurm_script() {
    if [ ! -f "${SLURM_SCRIPT}" ]; then
        echo "错误: Slurm脚本不存在: ${SLURM_SCRIPT}"
        exit 1
    fi
    chmod +x "${SLURM_SCRIPT}"
}

# 提交单个作业
submit_job() {
    local dataset=$1
    local variant=$2
    local num_scales=$3
    local batch_size=$4
    local lr=$5
    local weight_decay=$6
    local dropout=$7

    local job_name="msft_${variant}_${dataset}_s${num_scales}"

    echo "提交作业: ${job_name}"
    echo "  参数: bs=${batch_size}, lr=${lr}, wd=${weight_decay}, dropout=${dropout}"

    # 提交作业并获取Job ID
    local job_id=$(sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --output="${LOG_DIR}/%j_${job_name}.out" \
        --error="${LOG_DIR}/%j_${job_name}.err" \
        "${SLURM_SCRIPT}" \
        "${dataset}" "${variant}" "${num_scales}" "${batch_size}" "${lr}" "${weight_decay}" "${dropout}" \
        | awk '{print $4}')

    if [ -n "${job_id}" ]; then
        echo "  ✓ 作业已提交: Job ID = ${job_id}"
        echo "${job_id}" >> "${JOB_IDS_FILE}"
        return 0
    else
        echo "  ✗ 作业提交失败"
        return 1
    fi
}

# 等待作业槽位可用
wait_for_slot() {
    while true; do
        # 获取当前运行和待处理的作业数
        local running_jobs=$(squeue -u $USER -t RUNNING,PENDING -h | wc -l)

        if [ ${running_jobs} -lt ${MAX_PARALLEL_JOBS} ]; then
            return 0
        fi

        echo "  等待作业槽位... (当前: ${running_jobs}/${MAX_PARALLEL_JOBS})"
        sleep 30
    done
}

# ============================================================================
# 主要功能
# ============================================================================

# 1. 提交核心配置（24个实验）
submit_core_configs() {
    echo "======================================================================"
    echo "提交核心配置实验"
    echo "======================================================================"
    echo "配置: 2数据集 × 4变体 × 3scales = 24个实验"
    echo "超参数: bs=${CORE_BATCH_SIZE}, lr=${CORE_LR}, wd=${CORE_WD}, dropout=${CORE_DROPOUT}"
    echo "最大并行数: ${MAX_PARALLEL_JOBS}"
    echo "======================================================================"
    echo ""

    setup_directories
    check_slurm_script

    # 清空作业ID文件
    > "${JOB_IDS_FILE}"

    local total=0
    local submitted=0

    for dataset in "${DATASETS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for num_scales in "${NUM_SCALES_LIST[@]}"; do
                ((total++))

                # 等待槽位可用
                wait_for_slot

                if submit_job "${dataset}" "${variant}" "${num_scales}" \
                             "${CORE_BATCH_SIZE}" "${CORE_LR}" "${CORE_WD}" "${CORE_DROPOUT}"; then
                    ((submitted++))
                fi

                echo ""
                sleep 2  # 短暂延迟避免过快提交
            done
        done
    done

    echo ""
    echo "======================================================================"
    echo "核心配置提交完成"
    echo "  总实验数: ${total}"
    echo "  已提交: ${submitted}"
    echo "======================================================================"
    echo ""
    echo "查看作业状态: squeue -u $USER"
    echo "或使用: ./slurm_submit_ablation.sh status"
}

# 2. 提交完整grid搜索
submit_full_grid() {
    echo "======================================================================"
    echo "提交完整Grid搜索"
    echo "======================================================================"

    local total_experiments=$((${#DATASETS[@]} * ${#VARIANTS[@]} * ${#NUM_SCALES_LIST[@]} * \
                               ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#WEIGHT_DECAYS[@]} * ${#DROPOUTS[@]}))

    echo "警告: 这将提交约 ${total_experiments} 个实验！"
    echo "预计总时间: 约 $((total_experiments * 2 / MAX_PARALLEL_JOBS)) 小时（假设每个实验2小时）"
    echo ""
    read -p "是否继续? (yes/no) " -r
    if [[ ! $REPLY == "yes" ]]; then
        echo "取消提交"
        exit 0
    fi

    setup_directories
    check_slurm_script

    # 清空作业ID文件
    > "${JOB_IDS_FILE}"

    local total=0
    local submitted=0

    for dataset in "${DATASETS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for num_scales in "${NUM_SCALES_LIST[@]}"; do
                for batch_size in "${BATCH_SIZES[@]}"; do
                    for lr in "${LEARNING_RATES[@]}"; do
                        for wd in "${WEIGHT_DECAYS[@]}"; do
                            for dropout in "${DROPOUTS[@]}"; do
                                ((total++))

                                wait_for_slot

                                if submit_job "${dataset}" "${variant}" "${num_scales}" \
                                             "${batch_size}" "${lr}" "${wd}" "${dropout}"; then
                                    ((submitted++))
                                fi

                                echo ""
                                sleep 2
                            done
                        done
                    done
                done
            done
        done
    done

    echo ""
    echo "======================================================================"
    echo "完整Grid搜索提交完成"
    echo "  总实验数: ${total}"
    echo "  已提交: ${submitted}"
    echo "======================================================================"
}

# 3. 提交自定义配置（交互式）
submit_custom() {
    echo "======================================================================"
    echo "自定义配置提交"
    echo "======================================================================"

    setup_directories
    check_slurm_script

    # 交互式输入
    echo "请选择数据集:"
    select dataset in "${DATASETS[@]}"; do
        if [ -n "${dataset}" ]; then
            break
        fi
    done

    echo "请选择变体:"
    select variant in "${VARIANTS[@]}"; do
        if [ -n "${variant}" ]; then
            break
        fi
    done

    echo "请选择scale数量:"
    select num_scales in "${NUM_SCALES_LIST[@]}"; do
        if [ -n "${num_scales}" ]; then
            break
        fi
    done

    read -p "Batch size (默认: 64): " batch_size
    batch_size=${batch_size:-64}

    read -p "Learning rate (默认: 1e-3): " lr
    lr=${lr:-1e-3}

    read -p "Weight decay (默认: 5e-2): " wd
    wd=${wd:-5e-2}

    read -p "Dropout (默认: 0.1): " dropout
    dropout=${dropout:-0.1}

    echo ""
    echo "======================================================================"
    echo "配置摘要:"
    echo "  数据集: ${dataset}"
    echo "  变体: ${variant}"
    echo "  Scales: ${num_scales}"
    echo "  Batch size: ${batch_size}"
    echo "  Learning rate: ${lr}"
    echo "  Weight decay: ${wd}"
    echo "  Dropout: ${dropout}"
    echo "======================================================================"
    echo ""

    read -p "确认提交? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        submit_job "${dataset}" "${variant}" "${num_scales}" \
                   "${batch_size}" "${lr}" "${wd}" "${dropout}"
    else
        echo "取消提交"
    fi
}

# 4. 查看作业状态
show_status() {
    echo "======================================================================"
    echo "Slurm作业状态"
    echo "======================================================================"

    if [ ! -f "${JOB_IDS_FILE}" ]; then
        echo "没有找到作业记录文件"
        echo ""
        echo "当前用户的所有作业:"
        squeue -u $USER
        return
    fi

    local job_ids=$(cat "${JOB_IDS_FILE}" | tr '\n' ',')
    if [ -z "${job_ids}" ]; then
        echo "没有记录的作业"
        echo ""
        echo "当前用户的所有作业:"
        squeue -u $USER
        return
    fi

    echo "记录的作业状态:"
    squeue -j ${job_ids%,} -o "%.18i %.20j %.8T %.10M %.6D %R" 2>/dev/null || true

    echo ""
    echo "作业统计:"
    local total=$(cat "${JOB_IDS_FILE}" | wc -l)
    local running=$(squeue -j ${job_ids%,} -t RUNNING -h 2>/dev/null | wc -l)
    local pending=$(squeue -j ${job_ids%,} -t PENDING -h 2>/dev/null | wc -l)
    local completed=$((total - running - pending))

    echo "  总计: ${total}"
    echo "  运行中: ${running}"
    echo "  等待中: ${pending}"
    echo "  已完成: ${completed}"

    echo ""
    echo "======================================================================"
}

# 5. 取消所有作业
cancel_jobs() {
    echo "======================================================================"
    echo "取消作业"
    echo "======================================================================"

    if [ ! -f "${JOB_IDS_FILE}" ]; then
        echo "没有找到作业记录文件"
        return
    fi

    local job_ids=$(cat "${JOB_IDS_FILE}" | tr '\n' ' ')
    if [ -z "${job_ids}" ]; then
        echo "没有记录的作业"
        return
    fi

    echo "将要取消以下作业:"
    squeue -j $(echo ${job_ids} | tr ' ' ',') 2>/dev/null || echo "没有活动作业"

    echo ""
    read -p "确认取消所有作业? (yes/no) " -r
    if [[ $REPLY == "yes" ]]; then
        for job_id in ${job_ids}; do
            scancel ${job_id} 2>/dev/null && echo "  ✓ 取消作业: ${job_id}" || true
        done
        echo ""
        echo "所有作业已取消"
    else
        echo "取消操作"
    fi
}

# ============================================================================
# 主程序入口
# ============================================================================

show_usage() {
    echo "用法: $0 <command>"
    echo ""
    echo "命令:"
    echo "  core      提交核心配置（24个实验）"
    echo "  full      提交完整grid搜索（576个实验）"
    echo "  custom    提交自定义配置（交互式）"
    echo "  status    查看作业状态"
    echo "  cancel    取消所有记录的作业"
    echo ""
    echo "示例:"
    echo "  $0 core"
    echo "  $0 status"
    echo ""
}

main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    local command=$1

    case "${command}" in
        core)
            submit_core_configs
            ;;
        full)
            submit_full_grid
            ;;
        custom)
            submit_custom
            ;;
        status)
            show_status
            ;;
        cancel)
            cancel_jobs
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

main "$@"
