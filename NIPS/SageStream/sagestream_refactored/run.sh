#!/bin/sh
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G
# Set the conda environment
CONDA_ENV="/home/infres/yinwang/anaconda3/envs/sage/bin/python"

# Change to project directory
cd ~/eeg2025/NIPS/SageStream/sagestream_refactored

# Run the main script with all arguments
python main.py --mode kfold \
    --dataset APAVA \
    --k 5 \
    --epochs 30 \
    --disable_iib \
    --wandb_run_name SageStream_APAVA_kfold
    # --enable_iib \
    # --iib_kl_weight 0.1 \
    # --iib_adv_weight 0.1
