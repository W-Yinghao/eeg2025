#!/usr/bin/env python3
"""
Evaluation entry point for DEB experiments.

Usage:
    python experiments/deb/scripts/evaluate.py \
        --checkpoint checkpoints_deb/best_TUEV_codebrain_deb_acc0.9500.pth \
        --dataset TUEV --model codebrain --mode deb
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
from experiments.deb.models.deb_model import DEBModel
from experiments.deb.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='DEB Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='TUEV')
    parser.add_argument('--model', type=str, default='codebrain')
    parser.add_argument('--mode', type=str, default='deb')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'])
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--luna_size', type=str, default='base')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')

    # Load checkpoint to get saved config
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_cfg = ckpt.get('cfg', {})

    # Merge with defaults
    cfg = make_config(base=args.mode)
    cfg.update(saved_cfg)
    cfg['dataset'] = args.dataset
    cfg['model'] = args.model
    cfg['mode'] = args.mode
    cfg['batch_size'] = args.batch_size
    cfg['cuda'] = args.cuda

    dataset_config = DATASET_CONFIGS[cfg['dataset']]

    # Load data
    data_loaders, num_classes, seq_len, _ = load_deb_data(cfg, dataset_config)

    # Create backbone
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']

    weights_map = {
        'codebrain': os.path.join(REPO_ROOT, 'CodeBrain/Checkpoints/CodeBrain.pth'),
        'cbramod': os.path.join(REPO_ROOT, 'Cbramod_pretrained_weights.pth'),
        'luna': os.path.join(REPO_ROOT, f'BioFoundation/checkpoints/LUNA/LUNA_{args.luna_size}.safetensors'),
    }
    backbone, _, token_dim = create_backbone(
        model_type=args.model,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        n_layer=args.n_layer,
        pretrained_weights_path=weights_map.get(args.model),
        device=str(device),
    )

    bb_wrapper = BackboneWrapper(backbone, args.model, n_channels, seq_len, token_dim)
    model = DEBModel(bb_wrapper, cfg['mode'], num_classes, cfg).to(device)

    # Load trained weights
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    results = Evaluator.evaluate(model, data_loaders[args.split], device)
    Evaluator.print_results(results, args.split, dataset_config.get('label_names'))


if __name__ == '__main__':
    main()
