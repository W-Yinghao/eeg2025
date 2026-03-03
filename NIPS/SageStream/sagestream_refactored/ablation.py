"""
Ablation study utilities for SageStream.

This module provides tools for running systematic ablation studies
to analyze the contribution of different components.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime

import torch
import numpy as np

from .configs.config import (
    SageStreamConfig, AblationConfig, ABLATION_PRESETS,
    ModelConfig, MoEConfig, TrainingConfig, TTAConfig, DataConfig
)


class AblationType(Enum):
    """Types of ablation studies."""
    COMPONENT = "component"  # Remove single components
    COMBINATION = "combination"  # Remove combinations of components
    GRID = "grid"  # Grid search over component settings


@dataclass
class AblationExperiment:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    ablation_config: AblationConfig
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return {
            'name': self.name,
            'description': self.description,
            'ablation_config': asdict(self.ablation_config),
            'tags': self.tags
        }


class AblationStudy:
    """
    Manager for ablation studies.

    Provides utilities for:
    - Defining ablation experiments
    - Running experiments systematically
    - Aggregating and comparing results
    """

    # Standard ablation experiments
    STANDARD_EXPERIMENTS = {
        # Full model (baseline)
        "full": AblationExperiment(
            name="full",
            description="Full model with all components enabled",
            ablation_config=AblationConfig(),
            tags=["baseline"]
        ),

        # Single component ablations
        "no_subject_embedding": AblationExperiment(
            name="no_subject_embedding",
            description="Without subject embedding (removes subject-aware components)",
            ablation_config=AblationConfig(
                use_subject_embedding=False,
                use_style_alignment=False,
                use_hypernetwork=False
            ),
            tags=["subject", "embedding"]
        ),

        "no_style_alignment": AblationExperiment(
            name="no_style_alignment",
            description="Without style alignment",
            ablation_config=AblationConfig(use_style_alignment=False),
            tags=["style", "alignment"]
        ),

        "no_hypernetwork": AblationExperiment(
            name="no_hypernetwork",
            description="Without hypernetwork (static style parameters)",
            ablation_config=AblationConfig(use_hypernetwork=False),
            tags=["hypernetwork"]
        ),

        "no_aux_loss": AblationExperiment(
            name="no_aux_loss",
            description="Without auxiliary load balancing loss",
            ablation_config=AblationConfig(use_aux_loss=False),
            tags=["loss", "regularization"]
        ),

        "no_tta": AblationExperiment(
            name="no_tta",
            description="Without test-time adaptation",
            ablation_config=AblationConfig(use_tta=False),
            tags=["tta", "adaptation"]
        ),

        "no_freq_learning": AblationExperiment(
            name="no_freq_learning",
            description="Without frequency-aware learning",
            ablation_config=AblationConfig(use_freq_learning=False),
            tags=["frequency", "learning"]
        ),

        # Combinations
        "minimal": AblationExperiment(
            name="minimal",
            description="Minimal model (most components disabled)",
            ablation_config=AblationConfig(
                use_subject_embedding=False,
                use_style_alignment=False,
                use_hypernetwork=False,
                use_aux_loss=False,
                use_tta=False,
                use_freq_learning=False
            ),
            tags=["minimal", "baseline"]
        ),

        # Subject components only
        "subject_only": AblationExperiment(
            name="subject_only",
            description="Only subject embedding, no style/hypernetwork",
            ablation_config=AblationConfig(
                use_subject_embedding=True,
                use_style_alignment=False,
                use_hypernetwork=False,
                use_tta=False
            ),
            tags=["subject", "minimal"]
        ),

        # TTA only (no subject embedding during training)
        "tta_only": AblationExperiment(
            name="tta_only",
            description="Only TTA at test time, no subject embedding during training",
            ablation_config=AblationConfig(
                use_subject_embedding=False,
                use_style_alignment=False,
                use_hypernetwork=False,
                use_tta=True
            ),
            tags=["tta", "adaptation"]
        ),
    }

    def __init__(
        self,
        base_config: SageStreamConfig,
        output_dir: str = "./ablation_results"
    ):
        """
        Initialize ablation study.

        Args:
            base_config: Base configuration to modify for ablations
            output_dir: Directory to save results
        """
        self.base_config = base_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results: Dict[str, Dict[str, Any]] = {}

    def get_experiment_config(self, experiment_name: str) -> SageStreamConfig:
        """
        Get configuration for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Modified SageStreamConfig for the experiment
        """
        if experiment_name not in self.STANDARD_EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment_name}. "
                           f"Available: {list(self.STANDARD_EXPERIMENTS.keys())}")

        experiment = self.STANDARD_EXPERIMENTS[experiment_name]

        # Create new config with modified ablation settings
        from dataclasses import asdict
        config_dict = {
            'model': asdict(self.base_config.model),
            'moe': asdict(self.base_config.moe),
            'training': asdict(self.base_config.training),
            'tta': asdict(self.base_config.tta),
            'data': asdict(self.base_config.data),
            'ablation': asdict(experiment.ablation_config),
            'output_dir': self.output_dir,
            'experiment_name': experiment_name
        }

        return SageStreamConfig.from_dict(config_dict)

    def add_custom_experiment(
        self,
        name: str,
        description: str,
        ablation_config: AblationConfig,
        tags: List[str] = None
    ):
        """
        Add a custom ablation experiment.

        Args:
            name: Experiment name
            description: Description of the experiment
            ablation_config: Ablation configuration
            tags: Optional tags for categorization
        """
        self.STANDARD_EXPERIMENTS[name] = AblationExperiment(
            name=name,
            description=description,
            ablation_config=ablation_config,
            tags=tags or []
        )

    def save_result(
        self,
        experiment_name: str,
        result: Dict[str, Any]
    ):
        """
        Save result for an experiment.

        Args:
            experiment_name: Name of the experiment
            result: Result dictionary
        """
        self.results[experiment_name] = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.STANDARD_EXPERIMENTS.get(
                experiment_name,
                AblationExperiment(name=experiment_name, description="Custom", ablation_config=AblationConfig())
            ).to_dict(),
            'result': result
        }

        # Save to file
        filepath = os.path.join(self.output_dir, f'{experiment_name}_result.json')
        with open(filepath, 'w') as f:
            json.dump(self._make_serializable(self.results[experiment_name]), f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and tensors to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    def get_comparison_table(self) -> str:
        """
        Generate a comparison table of all results.

        Returns:
            Formatted string table
        """
        if not self.results:
            return "No results available."

        # Get all metric keys
        metric_keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

        # Build table
        lines = []
        lines.append("=" * 100)
        lines.append("ABLATION STUDY RESULTS COMPARISON")
        lines.append("=" * 100)
        lines.append("")

        # Header
        header = f"{'Experiment':<25}"
        for key in metric_keys:
            header += f"{key:<20}"
        lines.append(header)
        lines.append("-" * 100)

        # Results
        for name, data in self.results.items():
            result = data.get('result', {})
            metrics = result.get('mean_metrics', result.get('test_metrics', {}))

            row = f"{name:<25}"
            for key in metric_keys:
                value = metrics.get(key, 0.0)
                row += f"{value:.4f}              "[:20]
            lines.append(row)

        lines.append("=" * 100)

        return "\n".join(lines)

    def save_comparison(self, filename: str = "comparison.txt"):
        """Save comparison table to file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(self.get_comparison_table())

    def export_results(self, filename: str = "all_results.json"):
        """Export all results to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self._make_serializable(self.results), f, indent=2)


def create_ablation_configs(
    base_config: SageStreamConfig,
    experiment_names: List[str] = None
) -> Dict[str, SageStreamConfig]:
    """
    Create configurations for multiple ablation experiments.

    Args:
        base_config: Base configuration to modify
        experiment_names: List of experiment names (default: all standard experiments)

    Returns:
        Dictionary mapping experiment names to configurations
    """
    study = AblationStudy(base_config)

    if experiment_names is None:
        experiment_names = list(AblationStudy.STANDARD_EXPERIMENTS.keys())

    configs = {}
    for name in experiment_names:
        try:
            configs[name] = study.get_experiment_config(name)
        except ValueError as e:
            print(f"Warning: {e}")

    return configs


def run_ablation_study(
    base_config: SageStreamConfig,
    run_fn,  # Function that takes config and returns results
    experiment_names: List[str] = None,
    output_dir: str = "./ablation_results"
) -> Dict[str, Any]:
    """
    Run a complete ablation study.

    Args:
        base_config: Base configuration
        run_fn: Function that runs experiment and returns results
        experiment_names: Experiments to run
        output_dir: Output directory

    Returns:
        Dictionary with all results
    """
    study = AblationStudy(base_config, output_dir)

    if experiment_names is None:
        experiment_names = list(AblationStudy.STANDARD_EXPERIMENTS.keys())

    for name in experiment_names:
        print(f"\n{'='*60}")
        print(f"Running ablation experiment: {name}")
        print(f"{'='*60}")

        config = study.get_experiment_config(name)
        result = run_fn(config)
        study.save_result(name, result)

    # Generate comparison
    print("\n" + study.get_comparison_table())
    study.save_comparison()

    return study.results
