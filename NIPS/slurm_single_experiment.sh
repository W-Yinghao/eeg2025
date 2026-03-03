#!/bin/bash
#SBATCH --job-name=msft_cbramod
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

################################################################################
# MSFT CBraMod - Slurm单实验作业脚本
#
# 使用方法:
#   sbatch slurm_single_experiment.sh TUEV full 3 64 1e-3 5e-2 0.1
#
# 参数:
#   $1: dataset (TUEV/TUAB)
#   $2: variant (baseline/pos_refiner/criss_cross_agg/full)
#   $3: num_scales (2/3/4)
#   $4: batch_size
#   $5: learning_rate
#   $6: weight_decay
#   $7: dropout
################################################################################

# 默认参数
DATASET=${1:-TUEV}
VARIANT=${2:-full}
NUM_SCALES=${3:-3}
BATCH_SIZE=${4:-64}
LR=${5:-1e-3}
WEIGHT_DECAY=${6:-5e-2}
DROPOUT=${7:-0.1}

# 固定参数
MODEL=cbramod
EPOCHS=50
SEED=3407
N_LAYER=12
DIM_FF=800
NHEAD=8
LABEL_SMOOTHING=0.1
CLIP_VALUE=1.0

# WandB配置
WANDB_PROJECT="eeg-msft-improved-ablation"
WANDB_ENTITY=""  # 填入你的entity

# 目录设置
SCRIPT_DIR="/home/infres/yinwang/eeg2025/NIPS"
LOG_DIR="${SCRIPT_DIR}/logs"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"

# 创建日志目录
mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}

# 打印作业信息
echo "======================================================================"
echo "SLURM Job Information"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Working Directory: ${SLURM_SUBMIT_DIR}"
echo "Start Time: $(date)"
echo "======================================================================"
echo ""

# 打印实验配置
echo "======================================================================"
echo "Experiment Configuration"
echo "======================================================================"
echo "Dataset: ${DATASET}"
echo "Variant: ${VARIANT}"
echo "Num Scales: ${NUM_SCALES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Dropout: ${DROPOUT}"
echo "======================================================================"
echo ""

# 设置variant对应的参数
case ${VARIANT} in
    baseline)
        USE_POS_REFINER="--no_pos_refiner"
        USE_CRISS_CROSS_AGG="--no_criss_cross_agg"
        ;;
    pos_refiner)
        USE_POS_REFINER="--use_pos_refiner"
        USE_CRISS_CROSS_AGG="--no_criss_cross_agg"
        ;;
    criss_cross_agg)
        USE_POS_REFINER="--no_pos_refiner"
        USE_CRISS_CROSS_AGG="--use_criss_cross_agg"
        ;;
    full)
        USE_POS_REFINER="--use_pos_refiner"
        USE_CRISS_CROSS_AGG="--use_criss_cross_agg"
        ;;
    *)
        echo "Error: Unknown variant '${VARIANT}'"
        exit 1
        ;;
esac

# 生成run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="MSFT_${VARIANT}_${MODEL}_${DATASET}_s${NUM_SCALES}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}_do${DROPOUT}_${TIMESTAMP}_job${SLURM_JOB_ID}"

# 加载环境（根据你的集群配置修改）
# module load cuda/11.8
# module load python/3.9
# source /path/to/your/venv/bin/activate

# 或使用conda
# module load anaconda3
# source activate your_env_name

echo "Loading environment..."
# TODO: 在这里添加你的环境加载命令
echo ""

# 打印GPU信息
echo "======================================================================"
echo "GPU Information"
echo "======================================================================"
nvidia-smi
echo "======================================================================"
echo ""

# 构建训练命令
TRAIN_CMD="python ${SCRIPT_DIR}/finetune_msft_improved.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
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
    --wandb_run_name ${RUN_NAME} \
    ${USE_POS_REFINER} \
    ${USE_CRISS_CROSS_AGG}"

# 添加WandB entity（如果设置了）
if [ -n "${WANDB_ENTITY}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --wandb_entity ${WANDB_ENTITY}"
fi

# 打印命令
echo "======================================================================"
echo "Training Command"
echo "======================================================================"
echo "${TRAIN_CMD}"
echo "======================================================================"
echo ""

# 执行训练
echo "Starting training..."
eval ${TRAIN_CMD}

EXIT_CODE=$?

# 打印结束信息
echo ""
echo "======================================================================"
echo "Job Complete"
echo "======================================================================"
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "======================================================================"

exit ${EXIT_CODE}
