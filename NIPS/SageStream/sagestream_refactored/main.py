#!/usr/bin/env python3
"""
Main entry point for SageStream experiments.

This script provides a unified interface for:
- Training and evaluation
- K-fold cross-validation
- Test-time adaptation
- Ablation studies

Usage:
    python main.py --mode train --config configs/default.yaml
    python main.py --mode kfold --k 5
    python main.py --mode ablation --experiments full,no_tta,no_subject_embedding
"""

import warnings
# Filter PyTorch warnings about checkpoint and deprecations
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sagestream_refactored.configs.config import (
    SageStreamConfig, create_default_config, get_ablation_preset
)
from sagestream_refactored.data import DataLoaderFactory
from sagestream_refactored.trainers.sagestream_trainer import KFoldTrainer
from sagestream_refactored.tta import TTAManager, initialize_unknown_subject_embeddings
from sagestream_refactored.ablation import AblationStudy, create_ablation_configs
from sagestream_refactored.utils import set_all_seeds, clear_gpu_memory

# Import the model
from MoE_moment.momentfm.models.SS_MOMENT import SageStreamPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SageStream: Cross-subject EEG Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='kfold',
        choices=['train', 'kfold', 'ablation', 'eval'],
        help='Running mode'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (JSON)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=2025,
        help='Random seed'
    )

    # Dataset specific
    parser.add_argument(
        '--dataset',
        type=str,
        default='APAVA',
        choices=['APAVA', 'TUAB', 'SEEDV', 'PTB'],
        help='Dataset to use'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to dataset (optional, uses default if not specified)'
    )

    # K-fold specific
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )

    # Ablation specific
    parser.add_argument(
        '--experiments',
        type=str,
        default=None,
        help='Comma-separated list of ablation experiments to run'
    )

    parser.add_argument(
        '--list_experiments',
        action='store_true',
        help='List available ablation experiments'
    )

    # Training specific
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate'
    )

    # TTA specific
    parser.add_argument(
        '--no_tta',
        action='store_true',
        help='Disable test-time adaptation'
    )

    # Wandb specific
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable wandb logging'
    )

    parser.add_argument(
        '--wandb_project',
        type=str,
        default='sagestream',
        help='Wandb project name'
    )

    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='Wandb entity (username or team)'
    )

    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Wandb run name'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )

    # SA-MoE control
    parser.add_argument(
        '--disable_moe',
        action='store_true',
        help='Disable SA-MoE (use plain encoder + IIB only)'
    )

    # IIB (Information Invariant Bottleneck) specific
    parser.add_argument(
        '--enable_iib',
        action='store_true',
        default=True,
        help='Enable IIB module for subject-invariant learning'
    )

    parser.add_argument(
        '--disable_iib',
        action='store_true',
        help='Disable IIB module'
    )

    parser.add_argument(
        '--iib_hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for IIB variational encoder'
    )

    parser.add_argument(
        '--iib_latent_dim',
        type=int,
        default=256,
        help='Latent dimension for IIB bottleneck'
    )

    parser.add_argument(
        '--iib_kl_weight',
        type=float,
        default=0.1,
        help='Weight for KL divergence loss (alpha)'
    )

    parser.add_argument(
        '--iib_adv_weight',
        type=float,
        default=0.1,
        help='Weight for adversarial loss (beta)'
    )

    parser.add_argument(
        '--iib_grl_alpha',
        type=float,
        default=1.0,
        help='Gradient reversal layer alpha'
    )

    parser.add_argument(
        '--iib_grl_schedule',
        action='store_true',
        help='Enable progressive GRL alpha schedule'
    )

    # IIB variant selection
    parser.add_argument(
        '--iib_variant',
        type=str,
        default='nips',
        choices=['nips', 'icml'],
        help='IIB variant: "nips" (GRL+Discriminator) or "icml" (dual-head+CI loss)'
    )

    # ICML IIB specific
    parser.add_argument(
        '--icml_lambda_ib',
        type=float,
        default=0.1,
        help='ICML IIB: weight for IB (KL divergence) loss'
    )

    parser.add_argument(
        '--icml_beta_ci',
        type=float,
        default=10.0,
        help='ICML IIB: weight for CI (conditional independence) loss'
    )

    return parser.parse_args()


def setup_config(args) -> SageStreamConfig:
    """Setup configuration from args."""
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SageStreamConfig.from_dict(config_dict)
    else:
        # Use default config with automatic path resolution for specified dataset
        config = create_default_config(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path
        )

    # Override with command line args
    config.training.random_state = args.seed
    config.training.k_folds = args.k
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.device = args.device
    config.training.num_workers = args.num_workers

    if args.no_tta:
        config.tta.enable_tta = False
        config.ablation.use_tta = False

    # SA-MoE settings
    if args.disable_moe:
        config.ablation.use_moe = False
        # __post_init__ won't re-run, so propagate manually
        config.ablation.use_subject_embedding = False
        config.ablation.use_style_alignment = False
        config.ablation.use_hypernetwork = False
        config.ablation.use_aux_loss = False
        config.moe.enable_subject_style_normalization = False

    # IIB settings
    if args.disable_iib:
        config.iib.enable_iib = False
        config.ablation.use_iib = False
    else:
        config.iib.enable_iib = args.enable_iib
        config.ablation.use_iib = args.enable_iib

    config.iib.iib_variant = args.iib_variant
    config.iib.hidden_dim = args.iib_hidden_dim
    config.iib.latent_dim = args.iib_latent_dim
    config.iib.kl_loss_weight = args.iib_kl_weight
    config.iib.adv_loss_weight = args.iib_adv_weight
    config.iib.grl_alpha = args.iib_grl_alpha
    config.iib.use_grl_schedule = args.iib_grl_schedule
    config.iib.icml_lambda_ib = args.icml_lambda_ib
    config.iib.icml_beta_ci = args.icml_beta_ci

    # Wandb settings
    if args.no_wandb:
        config.wandb.enable = False
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity

    # Set wandb run name: use user-specified name or generate automatically
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    else:
        # Auto-generate run name based on dataset, IIB variant, and model name
        if config.iib.enable_iib:
            iib_tag = f"iib_{config.iib.iib_variant}"
        else:
            iib_tag = "no_iib"
        model_name = config.model.model_name
        dataset_name = config.data.dataset_name
        config.wandb.run_name = f"{dataset_name}_{model_name}_{iib_tag}"

    config.output_dir = args.output_dir

    # Print resolved paths for debugging
    print(f"\nResolved configuration:")
    print(f"  Dataset: {config.data.dataset_name}")
    print(f"  Model path: {config.model.model_path}")
    print(f"  Dataset path: {config.data.dataset_path}")
    print(f"  Seq length: {config.model.seq_len}")
    print(f"  Num channels: {config.model.input_channels}")
    print(f"  Wandb enabled: {config.wandb.enable}")
    print(f"  Num classes: {config.model.num_classes}")
    print(f"  SA-MoE enabled: {config.ablation.use_moe}")
    print(f"  IIB enabled: {config.iib.enable_iib}")
    if config.iib.enable_iib:
        print(f"    - IIB variant: {config.iib.iib_variant}")
        print(f"    - IIB hidden dim: {config.iib.hidden_dim}")
        print(f"    - IIB latent dim: {config.iib.latent_dim}")
        if config.iib.iib_variant == 'nips':
            print(f"    - IIB KL weight (α): {config.iib.kl_loss_weight}")
            print(f"    - IIB Adv weight (β): {config.iib.adv_loss_weight}")
            print(f"    - IIB GRL alpha: {config.iib.grl_alpha}")
            print(f"    - IIB GRL schedule: {config.iib.use_grl_schedule}")
        elif config.iib.iib_variant == 'icml':
            print(f"    - ICML lambda_ib: {config.iib.icml_lambda_ib}")
            print(f"    - ICML beta_ci: {config.iib.icml_beta_ci}")
    sys.stdout.flush()

    return config


def run_single_experiment(config: SageStreamConfig) -> dict:
    """Run a single training/evaluation experiment."""
    print("\n" + "="*60)
    print(f"Running experiment: {config.experiment_name}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"TTA enabled: {config.tta.enable_tta}")
    print("="*60)
    sys.stdout.flush()

    # Set seed
    set_all_seeds(config.training.random_state)

    # Create data loaders
    print("\nLoading data...")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Num workers: {config.training.num_workers}")
    data_factory = DataLoaderFactory(
        dataset_path=config.data.dataset_path,
        dataset_name=config.data.dataset_name,
        cache_dir=config.data.cache_dir,
        use_cache=config.data.use_cache,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        target_seq_len=config.model.seq_len,
        tuab_data_format=config.data.tuab_data_format,
        tuab_normalize=config.data.tuab_normalize
    )

    fold_loaders = data_factory.get_k_fold_loaders(
        k=config.training.k_folds,
        random_state=config.training.random_state,
        val_split_ratio=config.training.val_split_ratio
    )

    # Create trainer
    trainer = KFoldTrainer(
        config=config,
        model_class=SageStreamPipeline,
        output_dir=config.output_dir
    )

    # Run k-fold
    results = trainer.run_k_fold(fold_loaders)

    # Run TTA if enabled
    if config.tta.enable_tta:
        print("\n" + "="*60)
        print("Running Test-Time Adaptation")
        print("="*60)

        tta_manager = TTAManager(config.tta, trainer.device)

        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
            if results['fold_results'][fold_idx].get('best_model_state') is not None:
                # Load best model
                model = trainer._create_model()
                model.load_state_dict(
                    results['fold_results'][fold_idx]['best_model_state']['model_state_dict']
                )

                # Initialize unknown subject embeddings
                train_subject_ids = []
                test_subject_ids = []
                for batch_data in train_loader:
                    if len(batch_data) == 3:
                        train_subject_ids.extend(batch_data[2].tolist())
                for batch_data in test_loader:
                    if len(batch_data) == 3:
                        test_subject_ids.extend(batch_data[2].tolist())

                initialize_unknown_subject_embeddings(
                    model,
                    sorted(list(set(train_subject_ids))),
                    sorted(list(set(test_subject_ids)))
                )

                # Run TTA - get d_model from model config
                feature_dim = getattr(model.config, 'd_model', 512) if hasattr(model, 'config') else 512
                tta_result = tta_manager.run_tta(
                    model=model,
                    test_loader=test_loader,
                    num_classes=config.model.num_classes,
                    num_channels=config.model.input_channels,
                    feature_dim=feature_dim
                )

                if tta_result:
                    results['fold_results'][fold_idx]['tta_metrics'] = tta_result['metrics']

                clear_gpu_memory()

    return results


def run_ablation_study(config: SageStreamConfig, experiment_names: list = None):
    """Run ablation study."""
    print("\n" + "="*60)
    print("Running Ablation Study")
    print("="*60)

    study = AblationStudy(config, os.path.join(config.output_dir, "ablation"))

    if experiment_names is None:
        experiment_names = list(AblationStudy.STANDARD_EXPERIMENTS.keys())

    for name in experiment_names:
        print(f"\n{'='*60}")
        print(f"Ablation Experiment: {name}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Get config for this experiment
        exp_config = study.get_experiment_config(name)

        # Run experiment
        result = run_single_experiment(exp_config)

        # Save result
        study.save_result(name, result)

    # Print comparison
    print("\n" + study.get_comparison_table())
    study.save_comparison()
    study.export_results()

    return study.results


def list_ablation_experiments():
    """List available ablation experiments."""
    print("\nAvailable Ablation Experiments:")
    print("="*60)

    for name, exp in AblationStudy.STANDARD_EXPERIMENTS.items():
        print(f"\n{name}:")
        print(f"  Description: {exp.description}")
        print(f"  Tags: {exp.tags}")
        print(f"  Config:")
        from dataclasses import asdict
        for key, value in asdict(exp.ablation_config).items():
            if value is False:  # Only show disabled components
                print(f"    - {key}: {value}")


def main():
    """Main entry point."""
    args = parse_args()

    # List experiments if requested
    if args.list_experiments:
        list_ablation_experiments()
        return

    # Setup config
    config = setup_config(args)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, f"{config.experiment_name}_{timestamp}")
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"\nOutput directory: {config.output_dir}")
    print(f"Config saved to: {config_path}")

    # Run based on mode
    if args.mode == 'train' or args.mode == 'kfold':
        results = run_single_experiment(config)

        # Save results
        results_path = os.path.join(config.output_dir, "results.json")

        def make_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj

        with open(results_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2)

        print(f"\nResults saved to: {results_path}")

    elif args.mode == 'ablation':
        experiment_names = None
        if args.experiments:
            experiment_names = [e.strip() for e in args.experiments.split(',')]

        results = run_ablation_study(config, experiment_names)

    elif args.mode == 'eval':
        # Evaluation only mode
        print("Evaluation mode not yet implemented")
        # TODO: Implement evaluation on pretrained model


if __name__ == "__main__":
    main()
