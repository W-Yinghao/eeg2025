"""
USBA — Universal Sufficient Bottleneck Adapter package.

Main entry points:
    from adapters import USBAConfig, USBAInjector, USBALoss, collect_usba_metrics
"""

from .usba_config import USBAConfig
from .usba import USBALayer, USBAAdapter
from .bottleneck import VariationalBottleneck
from .branches import (
    DepthwiseTemporalConv,
    LowRankTemporalMix,
    ChannelAttention,
    GroupedSpatialMLP,
    GatedFusion,
)
from .losses import USBALoss, collect_usba_metrics, class_conditional_hsic
from .injection import USBAInjector, USBAInjectedModel

__all__ = [
    'USBAConfig',
    'USBALayer',
    'USBAAdapter',
    'VariationalBottleneck',
    'DepthwiseTemporalConv',
    'LowRankTemporalMix',
    'ChannelAttention',
    'GroupedSpatialMLP',
    'GatedFusion',
    'USBALoss',
    'collect_usba_metrics',
    'class_conditional_hsic',
    'USBAInjector',
    'USBAInjectedModel',
]
