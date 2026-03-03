#!/bin/sh
#SBATCH --gpus=1
#SBATCH --partition=A40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# python ${PROJECT_DIR}/finetune_tuev_lmdb.py --dataset AD_DIAGNOSIS --cuda 0 --epochs 30 --wandb_project diagnosis_foundation_model --wandb_run_name Cbramod_all_finetuning --seed 0
python ${PROJECT_DIR}/finetune_tuev_lmdb.py \
    --dataset CHB-MIT \
    --finetuned_ckpt ${PROJECT_DIR}/checkpoints/ad_diagnosis_epoch6_bal_acc_0.74374_kappa_0.76930_f1_0.86524.pth \
    --linear_probe \
    --cuda 0 \
    --epochs 20 \
    --wandb_project diagnosis_foundation_model \
    --wandb_run_name Cbramod_finetuned_CHB-MIT_linear \
    --seed 0

python ${PROJECT_DIR}/finetune_tuev_lmdb.py \
    --dataset CHB-MIT \
    --finetuned_ckpt ${PROJECT_DIR}/checkpoints/ad_diagnosis_epoch6_bal_acc_0.74374_kappa_0.76930_f1_0.86524.pth \
    --cuda 0 \
    --epochs 20 \
    --wandb_project diagnosis_foundation_model \
    --wandb_run_name Cbramod_finetuned_CHB-MIT_finetune \
    --seed 0

python ${PROJECT_DIR}/finetune_tuev_lmdb.py \
    --dataset CHB-MIT \
    --cuda 0 \
    --epochs 20 \
    --linear_probe \
    --wandb_project diagnosis_foundation_model \
    --wandb_run_name Cbramod_finetuned_CHB-MIT_linear\
    --seed 0