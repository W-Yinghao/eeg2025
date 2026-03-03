#!/bin/bash
#SBATCH --job-name=wandb_agent
#SBATCH --output=logs/slurm_agent_%j.out
#SBATCH --error=logs/slurm_agent_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-3

################################################################################
# WandB Sweep Agent - Slurm作业脚本
#
# 这个脚本启动多个WandB agent来并行运行sweep实验
# 使用Slurm job array支持最多4个并行agent
#
# 使用方法:
#   # 1. 先创建sweep
#   wandb sweep sweep_msft_cbramod.yaml
#
#   # 2. 复制返回的sweep ID，然后提交agent作业
#   sbatch slurm_wandb_agent.sh <sweep_id>
#
#   # 或者指定并行agent数量（1-4）
#   sbatch --array=0-1 slurm_wandb_agent.sh <sweep_id>  # 2个agent
#
# 参数:
#   $1: sweep_id (必需) - WandB sweep ID，格式: entity/project/sweep_id
################################################################################

# 检查sweep ID参数
if [ -z "$1" ]; then
    echo "错误: 需要提供sweep ID"
    echo "用法: sbatch slurm_wandb_agent.sh <sweep_id>"
    echo "示例: sbatch slurm_wandb_agent.sh user/project/abc123def"
    exit 1
fi

SWEEP_ID=$1

# 目录设置
SCRIPT_DIR="/home/infres/yinwang/eeg2025/NIPS"
LOG_DIR="${SCRIPT_DIR}/logs"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 打印作业信息
echo "======================================================================"
echo "SLURM WandB Agent Job Information"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "Sweep ID: ${SWEEP_ID}"
echo "======================================================================"
echo ""

# 加载环境（根据你的集群配置修改）
echo "Loading environment..."
# module load cuda/11.8
# module load python/3.9
# source /path/to/your/venv/bin/activate

# 或使用conda
# module load anaconda3
# source activate your_env_name

# TODO: 在这里添加你的环境加载命令
echo ""

# 打印GPU信息
echo "======================================================================"
echo "GPU Information"
echo "======================================================================"
nvidia-smi
echo "======================================================================"
echo ""

# 设置CUDA设备（使用array task ID来分配不同的GPU）
# 如果你的节点有多个GPU，这会自动分配
export CUDA_VISIBLE_DEVICES=${SLURM_ARRAY_TASK_ID}

# 切换到工作目录
cd ${SCRIPT_DIR}

# 启动WandB agent
echo "======================================================================"
echo "Starting WandB Agent"
echo "======================================================================"
echo "Agent ID: ${SLURM_ARRAY_TASK_ID}"
echo "Sweep ID: ${SWEEP_ID}"
echo "CUDA Device: ${CUDA_VISIBLE_DEVICES}"
echo "======================================================================"
echo ""

# 运行agent
# --count 参数指定这个agent运行多少个实验后自动退出
# 不指定则会一直运行直到sweep完成或作业时间限制
wandb agent ${SWEEP_ID}

EXIT_CODE=$?

# 打印结束信息
echo ""
echo "======================================================================"
echo "Agent Complete"
echo "======================================================================"
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "======================================================================"

exit ${EXIT_CODE}
