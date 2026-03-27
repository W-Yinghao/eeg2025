#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G
#SBATCH --job-name=smoke_top2_selector
#SBATCH --output=experiments/deb/slurm_log/smoke_test/%j_top2_selector.out
#SBATCH --error=experiments/deb/slurm_log/smoke_test/%j_top2_selector.err

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg2025

cd /home/infres/yinwang/eeg2025/NIPS_finetune

python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector --regime top2 \
    --epochs 5 --cuda 0
