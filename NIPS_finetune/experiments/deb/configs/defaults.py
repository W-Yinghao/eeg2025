"""
Default configurations for DEB experiments.

Two modes:
  - BASELINE: frozen backbone + simple classifier head
  - DEB: frozen backbone + Disease Evidence Bottleneck head
"""

from copy import deepcopy


BASELINE_CONFIG = {
    # ── Mode ──────────────────────────────────────────────────────────
    'mode': 'baseline',          # 'baseline' | 'deb'

    # ── Dataset ───────────────────────────────────────────────────────
    'dataset': 'TUEV',
    'seed': 3407,

    # ── Backbone ──────────────────────────────────────────────────────
    'model': 'codebrain',        # 'codebrain' | 'cbramod' | 'luna'
    'n_layer': 8,                # CodeBrain SSSM layers
    'n_layer_cbramod': 12,
    'nhead': 8,
    'dim_feedforward': 800,
    'luna_size': 'base',
    'pretrained_weights': None,  # None → auto-resolve from model type

    # ── Fine-tune strategy ────────────────────────────────────────────
    'finetune': 'partial',       # 'frozen' | 'partial' | 'full'
    'freeze_patch_embed': True,  # freeze tokenizer / patch embedding layer

    # ── Head ──────────────────────────────────────────────────────────
    'head_hidden': 512,
    'head_dropout': 0.1,

    # ── Training ──────────────────────────────────────────────────────
    'epochs': 100,
    'batch_size': 64,
    'lr_head': 1e-3,
    'lr_backbone': 1e-5,
    'lr_ratio': 10.0,            # head_lr = backbone_lr * ratio (when partial)
    'weight_decay': 1e-3,
    'clip_value': 5.0,
    'patience': 15,
    'scheduler': 'cosine',       # 'cosine' | 'none'

    # ── Loss ──────────────────────────────────────────────────────────
    'class_weights': None,       # None → auto from dataset config if available
    'label_smoothing': 0.0,

    # ── Split ─────────────────────────────────────────────────────────
    'split_strategy': 'subject', # 'subject' | 'random' | 'site_held_out'
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'cross_subject': True,

    # ── Evaluation ────────────────────────────────────────────────────
    'primary_metric': 'balanced_accuracy',  # early stop metric
    'eval_test_every_epoch': False,

    # ── Logging ───────────────────────────────────────────────────────
    'wandb_project': None,
    'wandb_run_name': None,
    'save_dir': 'checkpoints_deb',
    'num_workers': 0,
    'cuda': 0,

    # ── Label filtering ───────────────────────────────────────────────
    'include_labels': None,
    'exclude_labels': None,
}


DEB_CONFIG = {
    **BASELINE_CONFIG,

    'mode': 'deb',

    # ── DEB Head ──────────────────────────────────────────────────────
    'deb_latent_dim': 64,
    'deb_temporal_gate': True,
    'deb_frequency_gate': True,
    'deb_gate_hidden': 64,
    'deb_fusion': 'concat',      # 'concat' | 'add'

    # ── DEB Loss ──────────────────────────────────────────────────────
    'beta': 1e-4,                # KL weight
    'beta_warmup_epochs': 5,     # linearly ramp beta from 0
    'sparse_lambda': 1e-3,       # gate sparsity L1 weight
    'enable_sparse_reg': True,

    # ── Consistency (interface only, default off) ─────────────────────
    'enable_consistency': False,
    'consistency_lambda': 0.0,
}


def make_config(overrides: dict = None, base: str = 'baseline') -> dict:
    """Create config from base with optional overrides."""
    cfg = deepcopy(DEB_CONFIG if base == 'deb' else BASELINE_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg
