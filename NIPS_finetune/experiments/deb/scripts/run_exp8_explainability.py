#!/usr/bin/env python3
"""
Experiment 8: Formal Explainability Evaluation.

Evaluates trained selector models with:
  1. Insertion/Deletion AUC
  2. Augmentation Consistency
  3. Abnormal-Focused Analysis

Usage:
    # Evaluate a single checkpoint
    python experiments/deb/scripts/run_exp8_explainability.py \
        --checkpoint checkpoints_selector/best_TUAB_codebrain_selector_top2_acc0.85_s3407.pth \
        --dataset TUAB --model codebrain --mode selector

    # Evaluate all checkpoints in a directory
    python experiments/deb/scripts/run_exp8_explainability.py \
        --checkpoint_dir checkpoints_selector/ \
        --dataset TUAB --model codebrain

    # With custom parameters
    python experiments/deb/scripts/run_exp8_explainability.py \
        --checkpoint path/to/model.pth \
        --dataset TUAB --model codebrain --mode selector \
        --n_insertion_steps 20 --n_augmentations 10 \
        --abnormal_class 0 --output_dir results_exp8/
"""

import argparse
import os
import sys
import json
import glob
from pathlib import Path

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
from experiments.deb.evaluation.explainability import run_full_explainability_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description='Experiment 8: Explainability Evaluation'
    )

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with checkpoints (evaluate all)')

    # Model config
    parser.add_argument('--dataset', type=str, default='TUAB')
    parser.add_argument('--model', type=str, default='codebrain')
    parser.add_argument('--mode', type=str, default='selector',
                        choices=['baseline', 'selector'])
    parser.add_argument('--regime', type=str, default=None,
                        help='Override regime (otherwise inferred from checkpoint)')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=64)

    # Evaluation parameters
    parser.add_argument('--n_insertion_steps', type=int, default=10)
    parser.add_argument('--n_augmentations', type=int, default=5)
    parser.add_argument('--abnormal_class', type=int, default=0,
                        help='Index of abnormal/disease class')

    # Output
    parser.add_argument('--output_dir', type=str, default='results_explainability')

    # Backbone
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_layer_cbramod', type=int, default=12)
    parser.add_argument('--luna_size', type=str, default='base')

    # Split
    parser.add_argument('--split_strategy', type=str, default='subject')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--num_workers', type=int, default=0)

    return parser.parse_args()


def evaluate_checkpoint(ckpt_path, args, data_loaders, dataset_config, device):
    """Evaluate a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {ckpt_path}")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get('cfg', {})

    # Determine mode and regime
    mode = args.mode or saved_cfg.get('mode', 'selector')
    regime = args.regime or saved_cfg.get('regime', 'frozen')

    # Build config
    cfg = {
        'mode': mode,
        'regime': regime,
        'dataset': args.dataset,
        'model': args.model,
        'seed': args.seed,
        'freeze_patch_embed': saved_cfg.get('freeze_patch_embed', True),
        'head_type': saved_cfg.get('head_type', 'pool'),
        'head_hidden': saved_cfg.get('head_hidden', 512),
        'head_dropout': saved_cfg.get('head_dropout', 0.1),
        'gate_hidden': saved_cfg.get('gate_hidden', 64),
        'fusion': saved_cfg.get('fusion', 'concat'),
        'enable_temporal_gate': saved_cfg.get('enable_temporal_gate', True),
        'enable_frequency_gate': saved_cfg.get('enable_frequency_gate', True),
        'enable_vib': saved_cfg.get('enable_vib', False),
        'vib_latent_dim': saved_cfg.get('vib_latent_dim', 64),
    }

    # Create backbone
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']
    seg_dur = dataset_config['segment_duration']
    seq_len = int(seg_dur * dataset_config['sampling_rate'] / patch_size)

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
        n_layer_cbramod=args.n_layer_cbramod,
        pretrained_weights_path=weights_map.get(args.model),
        device=str(device),
    )

    bb_wrapper = BackboneWrapper(backbone, args.model, n_channels, seq_len, token_dim)

    num_classes = dataset_config['num_classes']
    model = SelectorModel(bb_wrapper, mode, num_classes, cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")

    # Determine model name from checkpoint filename
    ckpt_name = Path(ckpt_path).stem

    # Run evaluation
    label_names = dataset_config.get('label_names', {})
    results = run_full_explainability_eval(
        model=model,
        dataloader=data_loaders['test'],
        device=device,
        output_dir=args.output_dir,
        label_names=label_names,
        abnormal_class=args.abnormal_class,
        n_insertion_steps=args.n_insertion_steps,
        n_augmentations=args.n_augmentations,
        model_name=ckpt_name,
    )

    return results


def main():
    args = parse_args()

    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')

    dataset_config = DATASET_CONFIGS[args.dataset]

    # Load data
    cfg_data = {
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'split_strategy': args.split_strategy,
        'val_ratio': args.val_ratio,
        'num_workers': args.num_workers,
    }
    data_loaders, _, _, _ = load_deb_data(cfg_data, dataset_config)

    # Collect checkpoints
    ckpt_paths = []
    if args.checkpoint:
        ckpt_paths.append(args.checkpoint)
    elif args.checkpoint_dir:
        ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')))
        print(f"Found {len(ckpt_paths)} checkpoints in {args.checkpoint_dir}")

    if not ckpt_paths:
        print("ERROR: No checkpoints specified. Use --checkpoint or --checkpoint_dir")
        return

    # Evaluate each
    all_results = {}
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found, skipping")
            continue
        results = evaluate_checkpoint(
            ckpt_path, args, data_loaders, dataset_config, device
        )
        all_results[Path(ckpt_path).stem] = results

    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_file = output_dir / 'explainability_combined.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_file}")


if __name__ == '__main__':
    main()
