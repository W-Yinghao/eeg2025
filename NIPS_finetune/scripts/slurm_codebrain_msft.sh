#!/bin/bash

################################################################################
# CodeBrain MSFT - Slurm批量提交
#
# 使用CodeBrain (SSSM)进行MSFT微调，提交到Slurm队列
#
# 使用方法:
#   ./slurm_codebrain_msft.sh submit       # 提交所有实验
#   ./slurm_codebrain_msft.sh submit_one TUEV 3  # 提交单个实验
#   ./slurm_codebrain_msft.sh status       # 查看状态
#   ./slurm_codebrain_msft.sh cancel       # 取消所有作业
################################################################################

set -e

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON_SCRIPT="${PROJECT_DIR}/finetune_msft.py"
LOG_DIR="${PROJECT_DIR}/logs_codebrain"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints_codebrain"

# Slurm配置
PARTITION="gpu"
TIME_LIMIT="12:00:00"
MEM="64G"
CPUS=8
GPUS="gpu:a100:1"

# WandB配置
WANDB_PROJECT="codebrain-msft-ablation"
WANDB_ENTITY=""

# 训练参数
EPOCHS=50
BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=5e-2
DROPOUT=0.1
CLIP_VALUE=1.0

# CodeBrain参数
MODEL="codebrain"
N_LAYER=8
CODEBOOK_SIZE_T=4096
CODEBOOK_SIZE_F=4096
SEED=3407
CODEBRAIN_WEIGHTS="${PROJECT_DIR}/CodeBrain/Checkpoints/CodeBrain.pth"

# 实验配置
DATASETS=("TUEV" "TUAB")
SCALE_CONFIGS=(1 2 3 4)

# 作业ID跟踪
JOB_IDS_FILE="${LOG_DIR}/codebrain_job_ids.txt"

# ============================================================================
# 函数
# ============================================================================

setup_directories() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
}

submit_single_job() {
    local dataset=$1
    local num_scales=$2

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local job_name="codebrain_s${num_scales}_${dataset}"
    local run_name="CodeBrain_MSFT_${dataset}_s${num_scales}_${timestamp}"
    local log_out="${LOG_DIR}/${job_name}_%j.out"
    local log_err="${LOG_DIR}/${job_name}_%j.err"

    echo "提交作业: ${job_name}"

    # 检查预训练权重
    if [ -f "${CODEBRAIN_WEIGHTS}" ]; then
        local pretrained_arg="--pretrained_weights ${CODEBRAIN_WEIGHTS}"
    else
        echo "  ⚠️  预训练权重未找到，使用随机初始化"
        local pretrained_arg="--no_pretrained"
    fi

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
        python_cmd="${python_cmd} --wandb_entity ${WANDB_ENTITY}"
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
echo "CodeBrain MSFT - ${num_scales} scales on ${dataset}"
echo "======================================================================"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node: \${SLURMD_NODENAME}"
echo "Start Time: \$(date)"
echo ""
echo "配置:"
echo "  模型: CodeBrain (SSSM)"
echo "  数据集: ${dataset}"
echo "  Num Scales: ${num_scales}"
echo "  N Layer: ${N_LAYER}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LR}"
echo "======================================================================"
echo ""

# 加载环境 - TODO: 根据你的集群配置修改
# module load cuda/11.8
# module load python/3.9
# source /path/to/venv/bin/activate

echo "GPU信息:"
nvidia-smi
echo ""

cd ${PROJECT_DIR}

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

submit_all_jobs() {
    echo "======================================================================"
    echo "提交所有CodeBrain MSFT实验"
    echo "======================================================================"
    echo "  数据集: ${#DATASETS[@]}"
    echo "  Scale配置: ${#SCALE_CONFIGS[@]}"
    echo "  总实验数: $((${#DATASETS[@]} * ${#SCALE_CONFIGS[@]}))"
    echo ""
    echo "模型: CodeBrain (SSSM) with ${N_LAYER} layers"
    echo "训练: epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LR}"
    echo "======================================================================"
    echo ""

    setup_directories
    > "${JOB_IDS_FILE}"

    local total=0
    local submitted=0

    for dataset in "${DATASETS[@]}"; do
        for num_scales in "${SCALE_CONFIGS[@]}"; do
            ((total++))

            if submit_single_job "${dataset}" "${num_scales}"; then
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

show_status() {
    echo "======================================================================"
    echo "CodeBrain MSFT 作业状态"
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
    squeue -j ${job_ids} -o "%.10i %.25j %.8T %.10M %.6D %R" 2>/dev/null || true

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

cancel_all_jobs() {
    if [ ! -f "${JOB_IDS_FILE}" ]; then
        echo "没有找到作业记录"
        return
    fi

    echo "取消所有CodeBrain MSFT作业..."
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
    echo "  submit                  提交所有实验 (8个作业)"
    echo "  submit_one <D> <S>      提交单个实验"
    echo "                          D: 数据集 (TUEV/TUAB)"
    echo "                          S: scale数量 (1/2/3/4)"
    echo "  status                  查看作业状态"
    echo "  cancel                  取消所有作业"
    echo ""
    echo "示例:"
    echo "  $0 submit                      # 提交所有8个实验"
    echo "  $0 submit_one TUEV 3           # 只提交TUEV+3scales"
    echo "  $0 status                      # 查看状态"
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
                echo "错误: submit_one 需要2个参数: <dataset> <num_scales>"
                show_usage
                exit 1
            fi

            setup_directories
            > "${JOB_IDS_FILE}"
            submit_single_job "$1" "$2"
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
