#!/usr/bin/env python3
"""
Training entry point for selector-only ablation.

Selector mode = gates + fusion + classifier, NO variational bottleneck.
Loss = CE only (no KL, no sparse regularisation).

Usage:
    python experiments/deb/scripts/train_selector.py --dataset TUAB --model codebrain --finetune frozen
    python experiments/deb/scripts/train_selector.py --dataset TUAB --model codebrain --finetune partial
    python experiments/deb/scripts/train_selector.py --dataset TUAB --model codebrain --finetune full
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

from experiments.deb.configs.defaults import make_config
from experiments.deb.data.batch_protocol import load_deb_data
from experiments.deb.models.backbone_wrapper import BackboneWrapper
from experiments.deb.models.deb_model_selector import DEBModelSelector
from experiments.deb.training.trainer import DEBTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Selector-only ablation')

    # Core
    parser.add_argument('--dataset', type=str, default='TUAB',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod', 'luna'])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)

    # Fine-tune
    parser.add_argument('--finetune', type=str, default='frozen',
                        choices=['frozen', 'partial', 'full'])
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
    parser.add_argument('--head_dropout', type=float, default=0.1)

    # Gate / fusion (reuse deb_* naming for consistency)
    parser.add_argument('--deb_gate_hidden', type=int, default=64)
    parser.add_argument('--deb_fusion', type=str, default='concat',
                        choices=['concat', 'add'])

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
    parser.add_argument('--save_dir', type=str, default='checkpoints_deb')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_test_every_epoch', action='store_true')

    # Label filtering
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config — use baseline as base (no KL/sparse params needed)
    cfg = make_config(base='baseline')
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    # Force selector mode
    cfg['mode'] = 'selector'
    cfg['deb_temporal_gate'] = True
    cfg['deb_frequency_gate'] = True
    # Zero out DEB loss terms so the trainer's loss is pure CE
    cfg['beta'] = 0.0
    cfg['sparse_lambda'] = 0.0
    cfg['enable_sparse_reg'] = False

    if cfg.get('lr_backbone') is None:
        cfg['lr_backbone'] = cfg['lr_head'] / cfg['lr_ratio']

    setup_seed(cfg['seed'])
    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset_config = DATASET_CONFIGS[cfg['dataset']]

    # ── Load data ─────────────────────────────────────────────────────
    data_loaders, num_classes, seq_len, num_subjects = load_deb_data(
        cfg, dataset_config
    )
    print(f"[Data] {cfg['dataset']}: {num_classes} classes, "
          f"seq_len={seq_len}, {num_subjects} subjects")

    # ── Create backbone ───────────────────────────────────────────────
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

    # ── Build model (selector mode) ──────────────────────────────────
    model = DEBModelSelector(
        backbone_wrapper=bb_wrapper,
        mode='selector',
        num_classes=num_classes,
        cfg=cfg,
    )

    # ── Train ─────────────────────────────────────────────────────────
    # Trainer sees mode='selector', which behaves like 'deb' in the
    # training loop (calls return_gates=True) but the loss only has CE
    # because beta=0 and sparse_lambda=0.
    trainer = DEBTrainer(
        model=model,
        cfg=cfg,
        data_loaders=data_loaders,
        dataset_config=dataset_config,
        device=device,
    )
    trainer.train()


if __name__ == '__main__':
    main()
