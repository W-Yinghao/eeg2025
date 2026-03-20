#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

# python Downstream/finetune_main_lmdb.py --downstream_dataset TUAB --cuda 0 --batch_size  64 --lr 2e-5 --weight_decay 5e-4 --dropout 0.3
python Downstream/finetune_main_lmdb.py --downstream_dataset TUAB --cuda 0 --seed 3407
