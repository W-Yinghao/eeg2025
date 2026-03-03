"""
Configuration management for SageStream.

This module provides dataclass-based configurations for:
- Model architecture
- Training parameters
- Data settings
- Ablation study flags
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base model
    model_name: str = "MOMENT-1-small"
    model_path: str = ""  # Will be set dynamically based on project root

    # Input dimensions
    seq_len: int = 256
    input_channels: int = 16
    num_classes: int = 2
    sampling_rate: float = 256.0

    # Architecture
    reduction: str = "concat"  # "concat" or "mean"
    freeze_embedder: bool = True
    freeze_encoder: bool = True
    freeze_head: bool = False
    add_positional_embedding: bool = False


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    # Expert settings
    num_experts: int = 5
    top_k: int = 2
    expert_dim_ratio: float = 1.0 / 8
    dropout: float = 0.1

    # Frequency learning
    freq_learning_mode: str = "lightweight_biomedical_filter"
    max_freq: float = 100.0
    routing_strategy: str = "simple"

    # Loss
    aux_loss_weight: float = 1.0

    # Subject-aware components
    enable_subject_style_normalization: bool = True
    num_subjects: int = 23
    subject_embedding_dim: int = 64
    expert_embedding_dim: int = 32
    hyper_expert_hidden_dim: int = 64
    moe_conditioning_dim: int = 64


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    aux_loss_weight: float = 0.001

    # Optimizer & Scheduler
    optimizer: str = "adamw"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    # Early stopping
    early_stop_patience: int = 5
    early_stop_metric: str = "balanced_accuracy"

    # Validation
    val_split_ratio: float = 0.25

    # Cross-validation
    enable_k_fold: bool = True
    k_folds: int = 5
    random_state: int = 2025

    # Device
    device: str = "cuda:0"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TTAConfig:
    """Test-Time Adaptation configuration."""
    enable_tta: bool = True
    tta_method: str = "STSA"  # "STSA" or None
    tta_learning_rate: float = 5e-4
    tta_batch_size: int = 64
    tta_steps_per_batch: int = 1


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    enable: bool = True
    project: str = "sagestream"
    entity: str = None  # Your wandb username or team
    run_name: str = None  # Wandb run name
    log_model: bool = False  # Log model checkpoints
    log_freq: int = 10  # Log every N batches


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "APAVA"
    dataset_path: str = ""  # Will be set dynamically
    cache_dir: str = "./cache"
    use_cache: bool = True

    # TUAB specific settings
    tuab_data_format: str = "lmdb"  # "lmdb" or "pickle"
    tuab_normalize: bool = True  # Divide by 100 for uV scaling


@dataclass
class IIBConfig:
    """
    Information Invariant Bottleneck (IIB) configuration.

    Supports two variants:
    - "nips": GRL + Subject Discriminator adversarial approach
      L_total = L_task + alpha * L_KL + beta * L_adv
    - "icml": Dual-head + CI loss approach
      L_total = L_inv + L_env + lambda * L_IB + beta * L_CI
    """
    # Enable/disable IIB
    enable_iib: bool = True  # If False, skip IIB and use original features

    # IIB variant selection
    iib_variant: str = "nips"  # "nips" (GRL-based) or "icml" (dual-head + CI loss)

    # Architecture (shared)
    hidden_dim: int = 256  # Hidden dimension for variational encoder
    latent_dim: int = 256  # Dimension of latent representation Z
    dropout: float = 0.1  # Dropout probability

    # NIPS variant specific
    discriminator_hidden_dim: Optional[int] = None  # Hidden dim for discriminator (default: latent_dim // 2)
    grl_alpha: float = 1.0  # Gradient reversal strength
    kl_loss_weight: float = 0.1  # Weight for KL divergence loss (alpha)
    adv_loss_weight: float = 0.1  # Weight for adversarial loss (beta)
    use_grl_schedule: bool = False  # If True, gradually increase GRL alpha
    grl_alpha_max: float = 1.0  # Maximum GRL alpha value
    grl_schedule_epochs: int = 10  # Number of epochs to reach max alpha

    # ICML variant specific
    icml_head_hidden_dim: int = 16  # Hidden dim for inv/env prediction heads
    icml_domain_dim: int = 1  # Dimension of domain label input
    icml_lambda_ib: float = 0.1  # Weight for IB (KL divergence) loss
    icml_beta_ci: float = 10.0  # Weight for CI (conditional independence) loss


@dataclass
class AblationConfig:
    """
    Ablation study configuration.

    Set flags to False to disable specific components for ablation studies.
    """
    # Core components
    use_moe: bool = True  # If False, use simple FFN instead of MoE
    use_subject_embedding: bool = True  # If False, no subject-aware components
    use_style_alignment: bool = True  # If False, skip style alignment
    use_hypernetwork: bool = True  # If False, use static style parameters

    # IIB (Information Invariant Bottleneck)
    use_iib: bool = True  # If False, skip IIB module

    # MoE variants
    use_shared_router: bool = True  # If False, each layer has its own router
    use_shared_experts: bool = True  # If False, each layer has its own experts
    use_expert_conditioning: bool = True  # If False, no conditioning on experts

    # Training variants
    use_aux_loss: bool = True  # If False, no load balancing loss
    use_tta: bool = True  # If False, no test-time adaptation

    # Frequency learning
    use_freq_learning: bool = True  # If False, standard MoE without frequency awareness


@dataclass
class SageStreamConfig:
    """Main configuration combining all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tta: TTAConfig = field(default_factory=TTAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    iib: IIBConfig = field(default_factory=IIBConfig)

    # Output
    output_dir: str = "./outputs"
    experiment_name: str = "sagestream"

    def __post_init__(self):
        """Validate and adjust configurations based on ablation settings."""
        # If MoE is disabled, disable all SA-MoE related components
        if not self.ablation.use_moe:
            self.ablation.use_subject_embedding = False
            self.ablation.use_style_alignment = False
            self.ablation.use_hypernetwork = False
            self.ablation.use_aux_loss = False
            self.moe.enable_subject_style_normalization = False

        # If subject embedding is disabled, related features should be disabled
        if not self.ablation.use_subject_embedding:
            self.moe.enable_subject_style_normalization = False

        # If TTA is disabled in ablation, update TTA config
        if not self.ablation.use_tta:
            self.tta.enable_tta = False

        # If IIB is disabled in ablation, update IIB config
        if not self.ablation.use_iib:
            self.iib.enable_iib = False

    def get_decoupling_config(self) -> Dict[str, Any]:
        """Generate decoupling config for model initialization."""
        return {
            'use_moe': self.ablation.use_moe,
            'shared_config': {
                'num_experts': self.moe.num_experts,
                'top_k': self.moe.top_k,
                'dropout': self.moe.dropout,
                'freq_learning_mode': self.moe.freq_learning_mode,
                'routing_strategy': self.moe.routing_strategy,
                'expert_dim_ratio': self.moe.expert_dim_ratio,
                'max_freq': self.moe.max_freq,
                'sampling_rate': self.model.sampling_rate,
                'aux_loss_weight': self.moe.aux_loss_weight,
                'enable_shared_backbone_hypernetwork': (
                    self.ablation.use_hypernetwork and
                    self.ablation.use_subject_embedding
                ),
                'num_subjects': self.moe.num_subjects,
                'subject_embedding_dim': self.moe.subject_embedding_dim,
                'expert_embedding_dim': self.moe.expert_embedding_dim,
                'hyper_expert_hidden_dim': self.moe.hyper_expert_hidden_dim,
                'num_channels': self.model.input_channels,
                'moe_conditioning_dim': self.moe.moe_conditioning_dim,
            },
            'iib_config': {
                'enable_iib': self.iib.enable_iib and self.ablation.use_iib,
                'iib_variant': self.iib.iib_variant,
                # Shared
                'hidden_dim': self.iib.hidden_dim,
                'latent_dim': self.iib.latent_dim,
                'dropout': self.iib.dropout,
                'num_subjects': self.moe.num_subjects,
                'num_classes': self.model.num_classes,
                # NIPS variant
                'discriminator_hidden_dim': self.iib.discriminator_hidden_dim,
                'grl_alpha': self.iib.grl_alpha,
                'kl_loss_weight': self.iib.kl_loss_weight,
                'adv_loss_weight': self.iib.adv_loss_weight,
                'use_grl_schedule': self.iib.use_grl_schedule,
                'grl_alpha_max': self.iib.grl_alpha_max,
                'grl_schedule_epochs': self.iib.grl_schedule_epochs,
                # ICML variant
                'icml_head_hidden_dim': self.iib.icml_head_hidden_dim,
                'icml_domain_dim': self.iib.icml_domain_dim,
                'icml_lambda_ib': self.iib.icml_lambda_ib,
                'icml_beta_ci': self.iib.icml_beta_ci,
            }
        }

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Generate model kwargs for model initialization."""
        return {
            "task_name": "classification",
            "n_channels": self.model.input_channels,
            "num_class": self.model.num_classes,
            "freeze_embedder": self.model.freeze_embedder,
            "freeze_encoder": self.model.freeze_encoder,
            "freeze_head": self.model.freeze_head,
            "seq_len": self.model.seq_len,
            "reduction": self.model.reduction,
            "add_positional_embedding": self.model.add_positional_embedding
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SageStreamConfig":
        """Create config from dictionary."""
        model = ModelConfig(**config_dict.get('model', {}))
        moe = MoEConfig(**config_dict.get('moe', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        tta = TTAConfig(**config_dict.get('tta', {}))
        data = DataConfig(**config_dict.get('data', {}))
        ablation = AblationConfig(**config_dict.get('ablation', {}))
        wandb = WandbConfig(**config_dict.get('wandb', {}))
        iib = IIBConfig(**config_dict.get('iib', {}))

        return cls(
            model=model,
            moe=moe,
            training=training,
            tta=tta,
            data=data,
            ablation=ablation,
            wandb=wandb,
            iib=iib,
            output_dir=config_dict.get('output_dir', './outputs'),
            experiment_name=config_dict.get('experiment_name', 'sagestream')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return {
            'model': asdict(self.model),
            'moe': asdict(self.moe),
            'training': asdict(self.training),
            'tta': asdict(self.tta),
            'data': asdict(self.data),
            'ablation': asdict(self.ablation),
            'wandb': asdict(self.wandb),
            'iib': asdict(self.iib),
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name
        }


# Preset configurations for common ablation studies
ABLATION_PRESETS = {
    "full": AblationConfig(),  # All components enabled

    "no_subject_embedding": AblationConfig(
        use_subject_embedding=False,
        use_style_alignment=False,
        use_hypernetwork=False
    ),

    "no_moe": AblationConfig(
        use_moe=False
    ),

    "no_style_alignment": AblationConfig(
        use_style_alignment=False
    ),

    "no_hypernetwork": AblationConfig(
        use_hypernetwork=False
    ),

    "no_aux_loss": AblationConfig(
        use_aux_loss=False
    ),

    "no_tta": AblationConfig(
        use_tta=False
    ),

    "no_freq_learning": AblationConfig(
        use_freq_learning=False
    ),

    "minimal": AblationConfig(
        use_subject_embedding=False,
        use_style_alignment=False,
        use_hypernetwork=False,
        use_aux_loss=False,
        use_tta=False,
        use_freq_learning=False
    ),

    "iib_only": AblationConfig(
        use_moe=False,
        use_subject_embedding=False,
        use_style_alignment=False,
        use_hypernetwork=False,
        use_aux_loss=False,
        use_iib=True,
    ),
}


def get_ablation_preset(name: str) -> AblationConfig:
    """Get a predefined ablation configuration by name."""
    if name not in ABLATION_PRESETS:
        available = list(ABLATION_PRESETS.keys())
        raise ValueError(f"Unknown ablation preset: {name}. Available: {available}")
    return ABLATION_PRESETS[name]


def _get_project_root() -> Path:
    """Get the project root directory."""
    # sagestream_refactored is inside SageStream directory
    current_file = Path(__file__).resolve()
    # Go up: config.py -> configs -> sagestream_refactored -> SageStream
    return current_file.parent.parent.parent


# Dataset-specific configurations
DATASET_CONFIGS = {
    "APAVA": {
        "seq_len": 256,
        "input_channels": 16,
        "num_classes": 2,
        "num_subjects": 23,
        "sampling_rate": 256.0,
        "default_path": "../datasets/APAVA",
    },
    "TUAB": {
        "seq_len": 256,  # Resampled from 2000
        "input_channels": 16,  # Bipolar montage
        "num_classes": 2,  # Normal vs Abnormal
        "num_subjects": 1000,  # Increased for TUAB which has many subjects
        "sampling_rate": 200.0,  # Original 200Hz
        "default_path": "/projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed",
    },
    "SEEDV": {
        "seq_len": 200,  # 1 second at 200Hz
        "input_channels": 62,  # 62 EEG channels (removed M1, M2, VEO, HEO)
        "num_classes": 5,  # 5 emotion classes
        "num_subjects": 16,  # 16 subjects in SEED-V
        "sampling_rate": 200.0,  # Resampled to 200Hz
        "default_path": "/projects/EEG-foundation-model/SEED-V-cbramod/processed_kiet",
    },
    "PTB": {
        "seq_len": 300,  # 300 time steps per sample
        "input_channels": 15,  # 15 channels
        "num_classes": 2,  # Binary classification
        "num_subjects": 198,  # 198 subjects
        "sampling_rate": 100.0,  # Estimated ~100Hz
        "default_path": "../datasets/PTB",
    },
}


def create_default_config(
    model_path: str = None,
    dataset_path: str = None,
    cache_dir: str = None,
    dataset_name: str = "APAVA"
) -> SageStreamConfig:
    """
    Create a default configuration for a specific dataset.

    Args:
        model_path: Path to pretrained model. If None, uses default location.
        dataset_path: Path to dataset. If None, uses default location.
        cache_dir: Cache directory. If None, uses default location.
        dataset_name: Name of dataset ("APAVA" or "TUAB")

    Returns:
        SageStreamConfig with appropriate paths set.
    """
    config = SageStreamConfig()

    # Set project root based paths
    project_root = _get_project_root()

    # Get dataset-specific settings
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    ds_config = DATASET_CONFIGS[dataset_name]

    # Update model config with dataset-specific settings
    config.model.seq_len = ds_config["seq_len"]
    config.model.input_channels = ds_config["input_channels"]
    config.model.num_classes = ds_config["num_classes"]
    config.model.sampling_rate = ds_config["sampling_rate"]

    # Update MoE config
    config.moe.num_subjects = ds_config["num_subjects"]

    # Update data config
    config.data.dataset_name = dataset_name

    if model_path is None:
        config.model.model_path = str(project_root / "MOMENT-1-small")
    else:
        config.model.model_path = model_path

    if dataset_path is None:
        config.data.dataset_path = ds_config["default_path"]
    else:
        config.data.dataset_path = dataset_path

    if cache_dir is None:
        config.data.cache_dir = str(project_root / "cache")
    else:
        config.data.cache_dir = cache_dir

    config.experiment_name = f"sagestream_{dataset_name.lower()}"

    return config


def create_apava_config(**kwargs) -> SageStreamConfig:
    """Create configuration for APAVA dataset."""
    return create_default_config(dataset_name="APAVA", **kwargs)


def create_tuab_config(**kwargs) -> SageStreamConfig:
    """Create configuration for TUAB dataset."""
    return create_default_config(dataset_name="TUAB", **kwargs)


def create_seedv_config(**kwargs) -> SageStreamConfig:
    """Create configuration for SEED-V dataset."""
    return create_default_config(dataset_name="SEEDV", **kwargs)


def create_ptb_config(**kwargs) -> SageStreamConfig:
    """Create configuration for PTB dataset."""
    return create_default_config(dataset_name="PTB", **kwargs)
