"""
CodeBrain finetune using LMDB data from finetune_tuev_lmdb.py.

This is a copy of finetune_main.py adapted to use LMDB datasets instead of
the original pickle-based datasets.

Usage:
    cd /home/infres/yinwang/eeg2025/NIPS_finetune/CodeBrain
    python Downstream/finetune_main_lmdb.py --downstream_dataset TUAB --cuda 0
    python Downstream/finetune_main_lmdb.py --downstream_dataset TUEV --cuda 0
    python Downstream/finetune_main_lmdb.py --downstream_dataset TUAB --frozen --cuda 0
"""
import argparse
import random
import datetime
import sys
import os
import numpy as np
import torch
import wandb
import warnings
warnings.filterwarnings('ignore')

# Add parent project to path for LMDB data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from finetune_tuev_lmdb import DATASET_CONFIGS, EEGLMDBDataset, load_data
from torch.utils.data import DataLoader

from Downstream.finetune_trainer import Trainer
from Models.model_for_tuab import Model as ModelTUAB
from Models.model_for_tuev import Model as ModelTUEV


def collate_fn_float_label(batch):
    """Collate that returns float labels (needed for BCEWithLogitsLoss)."""
    x_data = np.array([x[0] for x in batch])
    y_label = np.array([x[1] for x in batch])
    return torch.from_numpy(x_data).float(), torch.from_numpy(y_label).float()


def collate_fn_long_label(batch):
    """Collate that returns long labels (needed for CrossEntropyLoss)."""
    x_data = np.array([x[0] for x in batch])
    y_label = np.array([x[1] for x in batch])
    return torch.from_numpy(x_data).float(), torch.from_numpy(y_label).long()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='CodeBrain Downstream (LMDB)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--clip_value', type=float, default=5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--n_layer', type=int, default=8)

    parser.add_argument('--downstream_dataset', type=str, default='TUAB',
                        choices=['TUEV', 'TUAB'])
    parser.add_argument('--num_of_classes', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='Checkpoints/finetune/')
    parser.add_argument('--log_dir', type=str, default='logs/')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--frozen', action='store_true', help='Freeze backbone')
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    parser.add_argument('--foundation_dir', type=str,
                        default='/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth')

    parser.add_argument('--codebook_size_t', type=int, default=4096)
    parser.add_argument('--codebook_size_f', type=int, default=4096)
    parser.add_argument('--codebook_dim', type=int, default=32)

    # LMDB data loading params (needed by load_data)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--datasets_dir', type=str, default='')
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)
    parser.add_argument('--cross_subject', type=bool, default=None)

    # WandB
    parser.add_argument('--wandb_project', type=str, default='eeg_codebrain_finetune')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    params = parser.parse_args()

    # Trainer checks params.multi_lr for SGD optimizer
    params.multi_lr = False

    # Setup dataset-specific params
    if params.downstream_dataset == 'TUAB':
        params.num_of_classes = 2
        params.dataset = 'TUAB'
        task_type = 'binary'
    elif params.downstream_dataset == 'TUEV':
        params.num_of_classes = 6
        params.dataset = 'TUEV'
        task_type = 'multiclass'

    # Setup directories
    params.model_dir = params.model_dir + params.downstream_dataset + '/'
    params.log_dir = params.log_dir + params.downstream_dataset + '/'
    os.makedirs(params.model_dir, exist_ok=True)
    os.makedirs(params.log_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    params.file_name = str(params.log_dir) + str(current_time) + "_" + str(params.cuda) + ".txt"
    print(params)
    with open(params.file_name, "a") as file:
        file.write(str(params) + "\n")

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)

    # WandB init
    mode_tag = "frozen" if params.frozen else "full"
    run_name = params.wandb_run_name or f"CodeBrain_{mode_tag}_{params.downstream_dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if params.wandb_project:
        wandb.init(
            project=params.wandb_project,
            name=run_name,
            config=vars(params),
        )

    # Load LMDB datasets
    dataset_config = DATASET_CONFIGS[params.dataset].copy()
    if params.datasets_dir:
        dataset_config['data_dir'] = params.datasets_dir

    # Build datasets via load_data helper, but we need to rebuild DataLoaders
    # with the correct collate_fn (float labels for binary, long for multiclass)
    data_dir = dataset_config['data_dir']
    splits = dataset_config['splits']
    from pathlib import Path
    data_path = Path(data_dir)

    val_from_train = (splits['train'] == splits['val'])

    train_dataset = EEGLMDBDataset(
        data_path / splits['train'], dataset_config, split='train',
        val_ratio=params.val_ratio, is_val_from_train=val_from_train)

    if val_from_train:
        val_dataset = EEGLMDBDataset(
            data_path / splits['val'], dataset_config, split='val',
            val_ratio=params.val_ratio, is_val_from_train=True)
    else:
        val_dataset = EEGLMDBDataset(
            data_path / splits['val'], dataset_config, split='val')

    test_dataset = EEGLMDBDataset(
        data_path / splits['test'], dataset_config, split='test')

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Use float labels for binary (BCEWithLogitsLoss), long for multiclass (CrossEntropyLoss)
    collate_fn = collate_fn_float_label if task_type == 'binary' else collate_fn_long_label

    data_loader = {
        'train': DataLoader(
            train_dataset, batch_size=params.batch_size, collate_fn=collate_fn,
            shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True),
        'val': DataLoader(
            val_dataset, batch_size=params.batch_size, collate_fn=collate_fn,
            shuffle=False, num_workers=params.num_workers, pin_memory=True),
        'test': DataLoader(
            test_dataset, batch_size=params.batch_size, collate_fn=collate_fn,
            shuffle=False, num_workers=params.num_workers, pin_memory=True),
    }

    print(f'The downstream dataset is {params.downstream_dataset}')
    with open(params.file_name, "a") as file:
        file.write(f'The downstream dataset is {params.downstream_dataset}\n')

    # Create model
    if params.downstream_dataset == 'TUAB':
        model = ModelTUAB(params)
    elif params.downstream_dataset == 'TUEV':
        model = ModelTUEV(params)

    # Train
    t = Trainer(params, data_loader, model)
    if params.downstream_dataset == 'TUAB':
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        t.train_for_multiclass()

    if params.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    main()
