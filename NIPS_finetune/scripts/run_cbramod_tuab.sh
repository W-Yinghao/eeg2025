#!/bin/sh
#SBATCH --gpus=1
#SBATCH --partition=A40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 100G

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

python ${PROJECT_DIR}/finetune_tuev_lmdb.py --cuda 0 --dataset TUAB --wandb_project diagnosis_foundation_model --wandb_run_name Cbramod_TUAB_finetuning
