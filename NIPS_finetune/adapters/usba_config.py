"""
USBAConfig — Configuration dataclass for Universal Sufficient Bottleneck Adapter.

All hyperparameters with sensible defaults. Can be constructed from argparse or dict.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union


@dataclass
class USBAConfig:
    """
    Configuration for USBA (Universal Sufficient Bottleneck Adapter).

    Core USBA parameters control the adapter architecture;
    loss parameters control the training objective;
    extension flags reserve interface for BILO / SPCBA.
    """

    # ── Master switch ──────────────────────────────────────────────────
    enabled: bool = True

    # ── Layer selection ────────────────────────────────────────────────
    # 'all' = inject after every backbone layer; list of ints = specific layers
    # 'output' = inject only after the final backbone output (default, simplest)
    selected_layers: Union[str, List[int]] = 'output'

    # ── Branch architecture ────────────────────────────────────────────
    # temporal: 'depthwise_conv' | 'low_rank_mix'
    temporal_branch_type: str = 'depthwise_conv'
    temporal_kernel_size: int = 5
    temporal_rank: int = 16  # for low_rank_mix

    # spatial: 'channel_attention' | 'grouped_mlp'
    spatial_branch_type: str = 'channel_attention'
    spatial_reduction: int = 4  # reduction ratio for channel attention

    # ── Fusion ─────────────────────────────────────────────────────────
    # 'gated' (default, with interpretable gate stats)
    fusion_type: str = 'gated'

    # ── Variational bottleneck ─────────────────────────────────────────
    latent_dim: int = 64  # bottleneck dimension (d' per token)
    bottleneck_hidden_dim: Optional[int] = None  # if None, = input_dim

    # ── Gate type for residual write-back ──────────────────────────────
    # 'layer_wise' | 'token_wise' | 'channel_wise'
    gate_type: str = 'layer_wise'
    gate_init: float = 0.0  # initial gate value (0 → identity at init)

    # ── Loss weights ───────────────────────────────────────────────────
    beta: float = 1e-4  # KL weight (global or per-layer)
    per_layer_beta: Optional[List[float]] = None  # if set, overrides beta per layer
    beta_warmup_epochs: int = 5  # linearly warm up beta from 0

    lambda_cc_inv: float = 0.01  # class-conditional invariance weight
    eta_budget: float = 1e-3  # budget / sparsity regularization weight

    # ── Feature toggles ────────────────────────────────────────────────
    enable_cc_inv: bool = True  # auto-disabled if no subject_id in batch
    enable_budget_reg: bool = True
    freeze_backbone: bool = True

    # ── Ablation switches ──────────────────────────────────────────────
    factorized: bool = True  # True = temporal+spatial branches; False = single MLP
    non_factorized_baseline: bool = False  # ablation: plain bottleneck adapter

    # ── Logging ────────────────────────────────────────────────────────
    log_gate_stats: bool = True
    log_kl_per_layer: bool = True
    log_budget_stats: bool = True

    # ── KL reduction ─────────────────────────────────────────────────
    kl_reduction: str = 'mean'  # 'mean' (KL per token) or 'total' (sum over tokens)

    # ── Budget warmup ────────────────────────────────────────────────
    budget_warmup_epochs: int = 10  # budget reg is zero before this epoch, then ramps

    # ── Dropout ────────────────────────────────────────────────────────
    dropout: float = 0.1

    # ── Extension flags (BILO / SPCBA) — interface only ────────────────
    # BILO: budgeted information LoRA — future
    bilo_enabled: bool = False
    # SPCBA: shared-private conditional bottleneck — future
    spcba_enabled: bool = False

    # ── Task ───────────────────────────────────────────────────────────
    task_type: str = 'multiclass'  # 'multiclass' | 'binary'
    num_classes: int = 2
    class_weights: Optional[List[float]] = None  # for weighted CE

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'USBAConfig':
        """Create config from dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def add_argparse_args(cls, parser):
        """Add USBA arguments to an existing argparse parser."""
        g = parser.add_argument_group('USBA')
        g.add_argument('--usba', action='store_true', default=False,
                        help='Enable USBA adapter')
        g.add_argument('--usba_latent_dim', type=int, default=64)
        g.add_argument('--usba_gate_type', type=str, default='layer_wise',
                        choices=['layer_wise', 'token_wise', 'channel_wise'])
        g.add_argument('--usba_temporal', type=str, default='depthwise_conv',
                        choices=['depthwise_conv', 'low_rank_mix'])
        g.add_argument('--usba_spatial', type=str, default='channel_attention',
                        choices=['channel_attention', 'grouped_mlp'])
        g.add_argument('--usba_beta', type=float, default=1e-4,
                        help='KL loss weight')
        g.add_argument('--usba_beta_warmup', type=int, default=5,
                        help='Beta warmup epochs')
        g.add_argument('--usba_lambda_cc', type=float, default=0.01,
                        help='Class-conditional invariance weight')
        g.add_argument('--usba_eta_budget', type=float, default=1e-3,
                        help='Budget regularization weight')
        g.add_argument('--usba_no_cc_inv', action='store_true',
                        help='Disable class-conditional invariance')
        g.add_argument('--usba_no_budget', action='store_true',
                        help='Disable budget regularization')
        g.add_argument('--usba_no_factorize', action='store_true',
                        help='Disable factorized branches (ablation)')
        g.add_argument('--usba_selected_layers', type=str, default='output',
                        help='Layer selection: "output", "all", or comma-separated ints')
        g.add_argument('--usba_dropout', type=float, default=0.1)
        g.add_argument('--usba_kl_reduction', type=str, default='mean',
                        choices=['mean', 'total'],
                        help='KL reduction: mean (per token) or total (sum)')
        g.add_argument('--usba_budget_warmup', type=int, default=10,
                        help='Budget regularization warmup epochs')
        return parser

    @classmethod
    def from_args(cls, args) -> 'USBAConfig':
        """Build USBAConfig from parsed argparse namespace."""
        # Parse selected_layers
        sl = getattr(args, 'usba_selected_layers', 'output')
        if sl not in ('output', 'all'):
            try:
                sl = [int(x.strip()) for x in sl.split(',')]
            except ValueError:
                sl = 'output'

        return cls(
            enabled=getattr(args, 'usba', False),
            selected_layers=sl,
            latent_dim=getattr(args, 'usba_latent_dim', 64),
            gate_type=getattr(args, 'usba_gate_type', 'layer_wise'),
            temporal_branch_type=getattr(args, 'usba_temporal', 'depthwise_conv'),
            spatial_branch_type=getattr(args, 'usba_spatial', 'channel_attention'),
            beta=getattr(args, 'usba_beta', 1e-4),
            beta_warmup_epochs=getattr(args, 'usba_beta_warmup', 5),
            lambda_cc_inv=getattr(args, 'usba_lambda_cc', 0.01),
            eta_budget=getattr(args, 'usba_eta_budget', 1e-3),
            enable_cc_inv=not getattr(args, 'usba_no_cc_inv', False),
            enable_budget_reg=not getattr(args, 'usba_no_budget', False),
            factorized=not getattr(args, 'usba_no_factorize', False),
            dropout=getattr(args, 'usba_dropout', 0.1),
            kl_reduction=getattr(args, 'usba_kl_reduction', 'mean'),
            budget_warmup_epochs=getattr(args, 'usba_budget_warmup', 10),
        )
