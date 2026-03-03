#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

# Full fine-tuning with gradient stability fixes:
#   - clip_value=1.0 (reduced from 5.0)
#   - backbone_warmup_epochs=5 (freeze backbone for first 5 epochs)
#   - backbone_lr_ratio=0.01 (backbone lr = 1e-5 when lr=1e-3)
python train_hyperbolic_finetuning.py --dataset TUAB --cuda 0 --seed 42 --full_finetune \
    --clip_value 1.0 --backbone_warmup_epochs 5 --backbone_lr_ratio 0.01 --lr 1e-3
python train_hyperbolic_finetuning.py --dataset TUEV --cuda 0 --seed 42 --full_finetune \
    --clip_value 1.0 --backbone_warmup_epochs 5 --backbone_lr_ratio 0.01 --lr 1e-3
