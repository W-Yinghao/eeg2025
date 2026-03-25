#!/usr/bin/env python3
"""
Training entry point for Experiments 6 & 7:
  - Exp 6: true partial FT boundary search (baseline vs selector)
  - Exp 7: selector + sparse/consistency regularization

Usage:
    # Exp 6: frozen baseline
    python experiments/deb/scripts/train_partial_ft.py \
        --dataset TUAB --model codebrain --mode baseline --regime frozen

    # Exp 6: frozen selector
    python experiments/deb/scripts/train_partial_ft.py \
        --dataset TUAB --model codebrain --mode selector --regime frozen

    # Exp 6: top2 selector
    python experiments/deb/scripts/train_partial_ft.py \
        --dataset TUAB --model codebrain --mode selector --regime top2

    # Exp 7: selector + sparse
    python experiments/deb/scripts/train_partial_ft.py \
        --dataset TUAB --model codebrain --mode selector --regime top2 \
        --enable_sparse --sparse_lambda 1e-3 --sparse_type l1

    # Exp 7: selector + consistency
    python experiments/deb/scripts/train_partial_ft.py \
        --dataset TUAB --model codebrain --mode selector --regime top2 \
        --enable_consistency --consistency_lambda 1e-2 --consistency_type l2

    # Exp 7: selector + sparse + consistency
    python experiments/deb/scripts/train_partial_ft.py \
        --dataset TUAB --model codebrain --mode selector --regime top2 \
        --enable_sparse --sparse_lambda 1e-3 \
        --enable_consistency --consistency_lambda 1e-2
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from finetune_tuev_lmdb import DATASET_CONFIGS, setup_seed
from backbone_factory import create_backbone

from experiments.deb.data.batch_protocol import load_deb_data
from experiments.deb.models.backbone_wrapper import BackboneWrapper
from experiments.deb.models.selector_model import SelectorModel
from experiments.deb.training.selector_trainer import SelectorTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Selector Experiments (Exp 6/7): true partial FT + interpretability'
    )

    # Core
    parser.add_argument('--mode', type=str, default='selector',
                        choices=['baseline', 'selector'])
    parser.add_argument('--dataset', type=str, default='TUAB',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod', 'luna'])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)

    # True partial FT regime
    parser.add_argument('--regime', type=str, default='frozen',
                        choices=['frozen', 'top1', 'top2', 'top4', 'full'],
                        help='Partial FT regime: how many top backbone blocks to unfreeze')
    parser.add_argument('--freeze_patch_embed', action='store_true', default=True)
    parser.add_argument('--no_freeze_patch_embed', dest='freeze_patch_embed',
                        action='store_false')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_backbone', type=float, default=None)
    parser.add_argument('--lr_ratio', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'])

    # Head config
    parser.add_argument('--head_type', type=str, default='pool',
                        choices=['pool', 'flatten'])
    parser.add_argument('--head_hidden', type=int, default=512)
    parser.add_argument('--head_dropout', type=float, default=0.1)

    # Gate config (selector mode)
    parser.add_argument('--gate_hidden', type=int, default=64)
    parser.add_argument('--fusion', type=str, default='concat',
                        choices=['concat', 'add'])
    parser.add_argument('--no_temporal_gate', action='store_true')
    parser.add_argument('--no_frequency_gate', action='store_true')

    # Sparse regularization (Exp 7)
    parser.add_argument('--enable_sparse', action='store_true',
                        help='Enable gate sparse regularization')
    parser.add_argument('--sparse_lambda', type=float, default=1e-3)
    parser.add_argument('--sparse_type', type=str, default='l1',
                        choices=['l1', 'entropy', 'coverage'])

    # Consistency regularization (Exp 7)
    parser.add_argument('--enable_consistency', action='store_true',
                        help='Enable gate consistency regularization')
    parser.add_argument('--consistency_lambda', type=float, default=1e-2)
    parser.add_argument('--consistency_type', type=str, default='l2',
                        choices=['l2', 'cosine', 'kl'])
    parser.add_argument('--aug_jitter_std', type=float, default=0.05)
    parser.add_argument('--aug_mask_ratio', type=float, default=0.1)
    parser.add_argument('--aug_time_shift_max', type=int, default=1)

    # VIB extension (disabled by default)
    parser.add_argument('--enable_vib', action='store_true',
                        help='Enable VIB bottleneck (extension point)')
    parser.add_argument('--vib_latent_dim', type=int, default=64)
    parser.add_argument('--vib_beta', type=float, default=1e-4)
    parser.add_argument('--vib_warmup_epochs', type=int, default=5)

    # Split
    parser.add_argument('--split_strategy', type=str, default='subject',
                        choices=['subject', 'random', 'site_held_out'])
    parser.add_argument('--val_ratio', type=float, default=0.15)

    # Backbone
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_layer_cbramod', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--luna_size', type=str, default='base')
    parser.add_argument('--pretrained_weights', type=str, default=None)

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_selector')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_test_every_epoch', action='store_true')

    # Label filtering
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    cfg = vars(args).copy()
    cfg['enable_temporal_gate'] = not args.no_temporal_gate
    cfg['enable_frequency_gate'] = not args.no_frequency_gate

    if cfg.get('lr_backbone') is None:
        cfg['lr_backbone'] = cfg['lr_head'] / cfg['lr_ratio']

    setup_seed(cfg['seed'])
    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset_config = DATASET_CONFIGS[cfg['dataset']]
    cfg['num_classes'] = dataset_config['num_classes']

    # Load data
    data_loaders, num_classes, seq_len, num_subjects = load_deb_data(cfg, dataset_config)
    print(f"[Data] {cfg['dataset']}: {num_classes} classes, "
          f"seq_len={seq_len}, {num_subjects} subjects")

    # Create backbone
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']

    weights_path = cfg.get('pretrained_weights')
    if weights_path is None:
        luna_size = cfg.get('luna_size', 'base')
        weights_map = {
            'codebrain': os.path.join(REPO_ROOT, 'CodeBrain/Checkpoints/CodeBrain.pth'),
            'cbramod': os.path.join(REPO_ROOT, 'Cbramod_pretrained_weights.pth'),
            'luna': os.path.join(REPO_ROOT, f'BioFoundation/checkpoints/LUNA/LUNA_{luna_size}.safetensors'),
        }
        weights_path = weights_map.get(cfg['model'])

    backbone, backbone_out_dim, token_dim = create_backbone(
        model_type=cfg['model'],
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        n_layer=cfg.get('n_layer', 8),
        n_layer_cbramod=cfg.get('n_layer_cbramod', 12),
        nhead=cfg.get('nhead', 8),
        dim_feedforward=cfg.get('dim_feedforward', 800),
        pretrained_weights_path=weights_path,
        device=str(device),
    )

    bb_wrapper = BackboneWrapper(
        backbone=backbone,
        model_type=cfg['model'],
        n_channels=n_channels,
        seq_len=seq_len,
        token_dim=token_dim,
    )

    # Build model with true partial FT
    model = SelectorModel(
        backbone_wrapper=bb_wrapper,
        mode=cfg['mode'],
        num_classes=num_classes,
        cfg=cfg,
    )

    # Train
    trainer = SelectorTrainer(
        model=model,
        cfg=cfg,
        data_loaders=data_loaders,
        dataset_config=dataset_config,
        device=device,
    )
    trainer.train()


if __name__ == '__main__':
    main()
