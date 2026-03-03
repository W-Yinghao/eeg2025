"""
Models module for SageStream.

Contains two IIB (Information Invariant Bottleneck) implementations:

1. iib.py (NIPS version): GRL + Subject Discriminator adversarial approach
   - L_total = L_task + alpha * L_KL + beta * L_adv

2. iib_icml.py (ICML version): Dual-head + CI loss approach
   - L_total = L_inv + L_env + lambda * L_IB + beta * L_CI
"""

from .iib import (
    IIB,
    IIBConfig,
    GRL,
    VariationalEncoder,
    SubjectDiscriminator
)

from .iib_icml import (
    IIB_ICML,
    IIB_ICML_Config,
)

__all__ = [
    # NIPS version (GRL-based)
    'IIB',
    'IIBConfig',
    'GRL',
    'VariationalEncoder',
    'SubjectDiscriminator',
    # ICML version (dual-head + CI loss)
    'IIB_ICML',
    'IIB_ICML_Config',
]
