#!/bin/bash

################################################################################
# MSFT CBraMod Ablation Study - Slurm批量提交（固定参数版本）
#
# 提交所有ablation配置到Slurm队列，参数固定为默认值
#
# 实验配置:
#   - 2个数据集: TUEV, TUAB
#   - 4个变体: baseline, pos_refiner, criss_cross_agg, full
#   - 总计: 8个实验
#
# 使用方法:
#   ./slurm_ablation_fixed_params.sh submit     # 提交所有实验
#   ./slurm_ablation_fixed_params.sh submit_one TUEV baseline  # 提交单个实验
#   ./slurm_ablation_fixed_params.sh status     # 查看状态
#   ./slurm_ablation_fixed_params.sh cancel     # 取消所有作业
################################################################################

set -e

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/finetune_msft_improved.py"
LOG_DIR="${SCRIPT_DIR}/logs_ablation"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_ablation"

# Slurm配置
PARTITION="gpu"
TIME_LIMIT="12:00:00"
MEM="64G"
CPUS=8
GPUS="gpu:a100:1"

# WandB配置
WANDB_PROJECT="msft-ablation-study"
WANDB_ENTITY=""

# 固定的训练参数
EPOCHS=50
BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=5e-2
DROPOUT=0.1
LABEL_SMOOTHING=0.1
CLIP_VALUE=1.0
NUM_SCALES=3

# 模型架构
MODEL="cbramod"
N_LAYER=12
DIM_FF=800
NHEAD=8
SEED=3407

# 数据集和变体
DATASETS=("TUEV" "TUAB")
declare -a VARIANTS=(
    "baseline|false|false"
    "pos_refiner|true|false"
    "criss_cross_agg|false|true"
    "full|true|true"
)

# 作业ID跟踪
JOB_IDS_FILE="${LOG_DIR}/ablation_job_ids.txt"

# ============================================================================
# 函数
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
}

# 提交单个作业
submit_single_job() {
    local dataset=$1
    local variant_name=$2
    local use_pos_refiner=$3
    local use_criss_cross_agg=$4

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local job_name="msft_${variant_name}_${dataset}"
    local run_name="MSFT_${variant_name}_${dataset}_s${NUM_SCALES}_${timestamp}"
    local log_out="${LOG_DIR}/${job_name}_%j.out"
    local log_err="${LOG_DIR}/${job_name}_%j.err"

    echo "提交作业: ${job_name}"

    # 构建Python命令
    local python_cmd="python ${PYTHON_SCRIPT} \
        --model ${MODEL} \
        --dataset ${dataset} \
        --cuda 0 \
        --seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
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

    if [ -n "${WANDB_ENTITY}" ]; then
        python_cmd="${python_cmd} --wandb_entity ${WANDB_ENTITY}"
    fi

    if [ "${use_pos_refiner}" = "true" ]; then
        python_cmd="${python_cmd} --use_pos_refiner"
    else
        python_cmd="${python_cmd} --no_pos_refiner"
    fi

    if [ "${use_criss_cross_agg}" = "true" ]; then
        python_cmd="${python_cmd} --use_criss_cross_agg"
    else
        python_cmd="${python_cmd} --no_criss_cross_agg"
    fi

    # 创建临时Slurm脚本
    local slurm_script=$(mktemp)
    cat > "${slurm_script}" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_out}
#SBATCH --error=${log_err}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=${GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "======================================================================"
echo "MSFT Ablation Study - ${variant_name} on ${dataset}"
echo "======================================================================"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node: \${SLURMD_NODENAME}"
echo "Start Time: \$(date)"
echo ""
echo "配置:"
echo "  数据集: ${dataset}"
echo "  变体: ${variant_name}"
echo "  use_pos_refiner: ${use_pos_refiner}"
echo "  use_criss_cross_agg: ${use_criss_cross_agg}"
echo "  num_scales: ${NUM_SCALES}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  learning_rate: ${LR}"
echo "======================================================================"
echo ""

# 加载环境 - TODO: 根据你的集群配置修改
# module load cuda/11.8
# module load python/3.9
# source /path/to/venv/bin/activate

echo "GPU信息:"
nvidia-smi
echo ""

cd ${SCRIPT_DIR}

echo "开始训练..."
${python_cmd}

EXIT_CODE=\$?

echo ""
echo "======================================================================"
echo "作业完成"
echo "End Time: \$(date)"
echo "Exit Code: \${EXIT_CODE}"
echo "======================================================================"

exit \${EXIT_CODE}
EOF

    # 提交作业
    local job_id=$(sbatch "${slurm_script}" | awk '{print $4}')
    rm "${slurm_script}"

    if [ -n "${job_id}" ]; then
        echo "  ✓ 作业已提交: Job ID = ${job_id}"
        echo "${job_id}" >> "${JOB_IDS_FILE}"
        return 0
    else
        echo "  ✗ 作业提交失败"
        return 1
    fi
}

# 提交所有作业
submit_all_jobs() {
    echo "======================================================================"
    echo "提交所有Ablation Study实验"
    echo "======================================================================"
    echo "  数据集: ${#DATASETS[@]}"
    echo "  变体: ${#VARIANTS[@]}"
    echo "  总实验数: $((${#DATASETS[@]} * ${#VARIANTS[@]}))"
    echo ""
    echo "固定参数:"
    echo "  epochs: ${EPOCHS}, batch_size: ${BATCH_SIZE}"
    echo "  lr: ${LR}, weight_decay: ${WEIGHT_DECAY}"
    echo "  dropout: ${DROPOUT}, num_scales: ${NUM_SCALES}"
    echo "======================================================================"
    echo ""

    setup_directories

    # 清空作业ID文件
    > "${JOB_IDS_FILE}"

    local total=0
    local submitted=0

    for dataset in "${DATASETS[@]}"; do
        for variant_config in "${VARIANTS[@]}"; do
            IFS='|' read -r variant_name use_pos_refiner use_criss_cross_agg <<< "${variant_config}"

            ((total++))

            if submit_single_job "${dataset}" "${variant_name}" "${use_pos_refiner}" "${use_criss_cross_agg}"; then
                ((submitted++))
            fi

            echo ""
            sleep 2
        done
    done

    echo "======================================================================"
    echo "所有作业已提交"
    echo "  总数: ${total}"
    echo "  成功: ${submitted}"
    echo "======================================================================"
    echo ""
    echo "查看状态: squeue -u \$USER"
    echo "或使用: $0 status"
}

# 查看作业状态
show_status() {
    echo "======================================================================"
    echo "Ablation Study 作业状态"
    echo "======================================================================"

    if [ ! -f "${JOB_IDS_FILE}" ] || [ ! -s "${JOB_IDS_FILE}" ]; then
        echo "没有找到作业记录"
        echo ""
        echo "当前用户的所有作业:"
        squeue -u $USER
        return
    fi

    local job_ids=$(cat "${JOB_IDS_FILE}" | tr '\n' ',')
    job_ids=${job_ids%,}

    echo "记录的作业:"
    squeue -j ${job_ids} -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || true

    echo ""
    echo "统计:"
    local total=$(cat "${JOB_IDS_FILE}" | wc -l)
    local running=$(squeue -j ${job_ids} -t RUNNING -h 2>/dev/null | wc -l)
    local pending=$(squeue -j ${job_ids} -t PENDING -h 2>/dev/null | wc -l)
    local completed=$((total - running - pending))

    echo "  总计: ${total}"
    echo "  运行中: ${running}"
    echo "  等待中: ${pending}"
    echo "  已完成: ${completed}"
    echo "======================================================================"
}

# 取消所有作业
cancel_all_jobs() {
    if [ ! -f "${JOB_IDS_FILE}" ]; then
        echo "没有找到作业记录"
        return
    fi

    echo "取消所有Ablation Study作业..."
    while read -r job_id; do
        scancel ${job_id} 2>/dev/null && echo "  ✓ 取消作业: ${job_id}" || true
    done < "${JOB_IDS_FILE}"

    echo "完成"
}

# ============================================================================
# 主程序
# ============================================================================

show_usage() {
    echo "用法: $0 <command> [args]"
    echo ""
    echo "命令:"
    echo "  submit                提交所有ablation实验 (8个作业)"
    echo "  submit_one <D> <V>    提交单个实验"
    echo "                        D: 数据集 (TUEV/TUAB)"
    echo "                        V: 变体 (baseline/pos_refiner/criss_cross_agg/full)"
    echo "  status                查看作业状态"
    echo "  cancel                取消所有作业"
    echo ""
    echo "示例:"
    echo "  $0 submit                           # 提交所有8个实验"
    echo "  $0 submit_one TUEV full            # 只提交TUEV+full配置"
    echo "  $0 status                          # 查看状态"
    echo "  $0 cancel                          # 取消所有"
    echo ""
}

main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    local command=$1
    shift

    case "${command}" in
        submit)
            submit_all_jobs
            ;;
        submit_one)
            if [ $# -lt 2 ]; then
                echo "错误: submit_one 需要2个参数: <dataset> <variant>"
                show_usage
                exit 1
            fi

            local dataset=$1
            local variant_name=$2

            # 查找对应的variant配置
            local found=false
            for variant_config in "${VARIANTS[@]}"; do
                IFS='|' read -r vname use_pos use_cc <<< "${variant_config}"
                if [ "${vname}" = "${variant_name}" ]; then
                    setup_directories
                    > "${JOB_IDS_FILE}"
                    submit_single_job "${dataset}" "${vname}" "${use_pos}" "${use_cc}"
                    found=true
                    break
                fi
            done

            if [ "${found}" = false ]; then
                echo "错误: 未知的变体 '${variant_name}'"
                echo "可用变体: baseline, pos_refiner, criss_cross_agg, full"
                exit 1
            fi
            ;;
        status)
            show_status
            ;;
        cancel)
            cancel_all_jobs
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "错误: 未知命令 '${command}'"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
