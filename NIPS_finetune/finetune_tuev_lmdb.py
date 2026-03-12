#!/usr/bin/env python3
"""
EEG Fine-tuning Script with LMDB Data Loading and WandB Logging

This script fine-tunes CBraMod or CodeBrain (SSSM) on various EEG datasets using preprocessed LMDB files.

Supported backbones:
    - CBraMod (default): --model cbramod
    - CodeBrain (SSSM): --model codebrain

Supported datasets:
    - TUEV: TUH EEG Events (6 classes) - /projects/EEG-foundation-model/diagnosis_data/tuev_preprocessed/
    - TUAB: TUH EEG Abnormal (binary) - /projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed/
    - CHB-MIT: CHB-MIT Seizure (binary) - /projects/EEG-foundation-model/diagnosis_data/CHB-MIT_preprocessed/
    - TUSZ: TUH Seizure (binary) - /projects/EEG-foundation-model/diagnosis_data/tusz_preprocessed/
    - DIAGNOSIS: Disease classification (4 classes: normal/CVD/AD/depression) - /projects/EEG-foundation-model/diagnosis_data_lmdb/
    - DEPRESSION: Depression vs Normal (binary, 64ch, 5000Hz, 2s) - /projects/EEG-foundation-model/diagnosis_data_lmdb/depression_normal_original/
    - CVD_DEPRESSION_NORMAL: 3-class EGI data (CVD/normal/depression, 129ch) - /projects/EEG-foundation-model/diagnosis_data/cvd_normal_depression_preprocessed/
    - UNIFIED_DIAGNOSIS: 3-class unified data from EGI+BP (CVD/depression/normal, 58ch) - /projects/EEG-foundation-model/diagnosis_data/unified_diagnosis_preprocessed/
    - AD_DIAGNOSIS: 4-class AD-related data (AD/CVD/depression/normal, 58ch) - /projects/EEG-foundation-model/diagnosis_data/ad_diagnosis_preprocessed/

DIAGNOSIS Label Indices:
    0: normal
    1: CVD (Cardiovascular Disease)
    2: AD (Alzheimer's Disease)
    3: depression

DEPRESSION Label Indices:
    0: depression
    1: normal

CVD_DEPRESSION_NORMAL Label Indices:
    0: CVD (Cardiovascular Disease)
    1: normal
    2: depression

UNIFIED_DIAGNOSIS Label Indices:
    0: CVD (Cardiovascular Disease)
    1: depression
    2: normal
    Sources: eeg_CVD_EGI_83, eeg_depression_EGI_21, eeg_depression_BP_122, eeg_normal_EGI_17, eeg_normal_BP_166
    Features: 58 common channels (10-20 standard naming), 200Hz, 5s segments

AD_DIAGNOSIS Label Indices:
    0: AD (Alzheimer's Disease, including MCI, SCD, HC)
    1: CVD (Cardiovascular Disease)
    2: depression
    3: normal
    Sources: eeg_AD+MCI+SCD+HC_EGI_124 (.mat), eeg_CVD_EGI_83, eeg_depression_EGI_21,
             eeg_depression_BP_122, eeg_normal_EGI_17, eeg_normal_BP_166
    Features: 58 channels, 200Hz, 5s segments, cross-subject split

Usage:
    # TUEV dataset
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --epochs 50

    # TUAB dataset
    python finetune_tuev_lmdb.py --dataset TUAB --cuda 0 --epochs 50

    # CHB-MIT dataset
    python finetune_tuev_lmdb.py --dataset CHB-MIT --cuda 0 --epochs 50

    # TUSZ dataset
    python finetune_tuev_lmdb.py --dataset TUSZ --cuda 0 --epochs 50

    # DIAGNOSIS dataset (58 channels, 4-class disease classification)
    python finetune_tuev_lmdb.py --dataset DIAGNOSIS --cuda 0 --epochs 50

    # DEPRESSION dataset (64 channels, binary classification, 5000Hz, 2s segments)
    python finetune_tuev_lmdb.py --dataset DEPRESSION --cuda 0 --epochs 50

    # CVD_DEPRESSION_NORMAL dataset (129 channels, 3-class EGI data with cross-subject split)
    python finetune_tuev_lmdb.py --dataset CVD_DEPRESSION_NORMAL --cuda 0 --epochs 50

    # UNIFIED_DIAGNOSIS dataset (58 channels, 3-class, unified EGI+BP data with cross-subject split)
    python finetune_tuev_lmdb.py --dataset UNIFIED_DIAGNOSIS --cuda 0 --epochs 50

    # AD_DIAGNOSIS dataset (58 channels, 4-class: AD/CVD/depression/normal with cross-subject split)
    python finetune_tuev_lmdb.py --dataset AD_DIAGNOSIS --cuda 0 --epochs 50

    # DIAGNOSIS: 3-class classification without AD (exclude label 2)
    python finetune_tuev_lmdb.py --dataset DIAGNOSIS --cuda 0 --exclude_labels 2

    # DIAGNOSIS: 3-class with specific labels (normal, CVD, depression)
    python finetune_tuev_lmdb.py --dataset DIAGNOSIS --cuda 0 --include_labels 0 1 3

    # DIAGNOSIS: Binary classification (normal vs AD)
    python finetune_tuev_lmdb.py --dataset DIAGNOSIS --cuda 0 --include_labels 0 2

    # DIAGNOSIS: Binary classification (normal vs depression)
    python finetune_tuev_lmdb.py --dataset DIAGNOSIS --cuda 0 --include_labels 0 3

    # DIAGNOSIS: Binary classification (CVD vs AD)
    python finetune_tuev_lmdb.py --dataset DIAGNOSIS --cuda 0 --include_labels 1 2

    # ---- CodeBrain (SSSM) backbone ----
    # TUEV with CodeBrain backbone
    python finetune_tuev_lmdb.py --model codebrain --dataset TUEV --cuda 0 --epochs 50

    # TUAB with CodeBrain backbone
    python finetune_tuev_lmdb.py --model codebrain --dataset TUAB --cuda 0 --epochs 50

    # CodeBrain with custom SSSM layers and codebook sizes
    python finetune_tuev_lmdb.py --model codebrain --dataset TUEV --cuda 0 --n_layer 8 --codebook_size_t 4096 --codebook_size_f 4096

    # CodeBrain with custom pretrained weights
    python finetune_tuev_lmdb.py --model codebrain --dataset TUEV --cuda 0 --pretrained_weights /path/to/CodeBrain.pth

    # CodeBrain linear probing
    python finetune_tuev_lmdb.py --model codebrain --dataset TUEV --cuda 0 --linear_probe

    # With wandb disabled
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --no_wandb

    # With frozen backbone (full fine-tuning with frozen backbone)
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --frozen

    # Linear probing (freeze backbone, only train classifier head with higher LR)
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --linear_probe

    # Linear probing with custom learning rate
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --linear_probe --lr 0.01

    # With t-SNE visualization every 10 epochs
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --tsne_interval 10

    # Disable t-SNE visualization
    python finetune_tuev_lmdb.py --dataset TUEV --cuda 0 --tsne_interval 0
"""

import argparse
import copy
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import lmdb
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.manifold import TSNE
except (ImportError, AttributeError):
    # Fallback: sklearn unavailable (numpy version incompatibility)
    balanced_accuracy_score = None
    cohen_kappa_score = None
    confusion_matrix = None
    f1_score = None
    roc_auc_score = None
    average_precision_score = None
    TSNE = None
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add CBraMod to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CBraMod'))
from models.cbramod import CBraMod

# Add CodeBrain to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CodeBrain'))
from Models.SSSM import SSSM

# Add BioFoundation to path (for FEMBA and LUNA models)
# Temporarily swap out CBraMod's 'models' from sys.modules so BioFoundation's 'models' can be imported
_biofound_path = os.path.join(os.path.dirname(__file__), 'BioFoundation')
sys.path.insert(0, _biofound_path)
_saved_models = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.')}
for k in _saved_models:
    del sys.modules[k]

try:
    from models.FEMBA import FEMBA
    FEMBA_AVAILABLE = True
except ImportError as e:
    FEMBA_AVAILABLE = False
    print(f"Warning: FEMBA not available ({e}). Install mamba_ssm: pip install mamba-ssm")

try:
    from models.LUNA import LUNA
    LUNA_AVAILABLE = True
except ImportError as e:
    LUNA_AVAILABLE = False
    print(f"Warning: LUNA not available ({e}). Install timm and rotary-embedding-torch.")

# Restore CBraMod's 'models' modules and remove BioFoundation from path
_biofound_modules = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.')}
for k in _biofound_modules:
    del sys.modules[k]
sys.modules.update(_saved_models)
sys.path.remove(_biofound_path)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")


# ============================================================================
# Dataset Configurations
# ============================================================================
DATASET_CONFIGS = {
    'TUEV': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/tuev_preprocessed',
        'num_classes': 6,
        'task_type': 'multiclass',
        'n_channels': 16,
        'sampling_rate': 200,
        'segment_duration': 5,  # seconds
        'patch_size': 200,
        'label_names': {
            0: 'spsw',   # spike and slow wave
            1: 'gped',   # generalized periodic epileptiform discharges
            2: 'pled',   # periodic lateralized epileptiform discharges
            3: 'eyem',   # eye movement
            4: 'artf',   # artifact
            5: 'bckg',   # background
        },
        'splits': {'train': 'train', 'val': 'train', 'test': 'eval'},
    },
    'TUAB': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed',
        'num_classes': 2,
        'task_type': 'binary',
        'n_channels': 16,
        'sampling_rate': 200,
        'segment_duration': 10,  # seconds
        'patch_size': 200,
        'label_names': {
            0: 'normal',
            1: 'abnormal',
        },
        'splits': {'train': 'train', 'val': 'val', 'test': 'test'},
    },
    'CHB-MIT': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/CHB-MIT_preprocessed',
        'num_classes': 2,
        'task_type': 'binary',
        'n_channels': 16,
        'sampling_rate': 256,
        'segment_duration': 5,  # seconds
        'patch_size': 200,  # Will resample to match
        'label_names': {
            0: 'non-seizure',
            1: 'seizure',
        },
        'splits': {'train': 'train', 'val': 'val', 'test': 'test'},
    },
    'TUSZ': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/tusz_preprocessed',
        'num_classes': 2,
        'task_type': 'binary',
        'n_channels': 22,
        'sampling_rate': 200,
        'segment_duration': 5,  # seconds
        'patch_size': 200,
        'label_names': {
            0: 'non-seizure',
            1: 'seizure',
        },
        'splits': {'train': 'train', 'val': 'dev', 'test': 'eval'},
    },
    'DIAGNOSIS': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data_lmdb——5s',
        'num_classes': 4,
        'task_type': 'multiclass',
        'n_channels': 58,
        'sampling_rate': 200,  # Resampled to 200Hz (matching patch_size=200)
        'segment_duration': 5,  # seconds (seq_len=5, matching TUEV/AD_DIAGNOSIS)
        'patch_size': 200,
        'label_names': {
            0: 'normal',
            1: 'CVD',
            2: 'AD',
            3: 'depression',
        },
        'disease_label_map': {
            'normal': 0,
            'CVD': 1,
            'AD': 2,
            'depression': 3,
        },
        'data_format': 'diagnosis',  # Special format: 'data' key and 'labels' dict
        'splits': {'train': 'eeg_segments.lmdb', 'val': 'eeg_segments.lmdb', 'test': 'eeg_segments.lmdb'},
        'single_lmdb': True,  # All splits come from same LMDB
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'cross_subject': True,  # Enable cross-subject split by default
    },
    'DEPRESSION': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/depression_normal_preprocessed_CBramod',
        'num_classes': 2,
        'task_type': 'binary',
        'n_channels': 63,  # EEG channels after excluding ECG
        'sampling_rate': 200,  # Resampled to 200 Hz (matching CBraMod)
        'segment_duration': 5,  # seconds (matching CBraMod's 5-second segments)
        'patch_size': 200,
        'label_names': {
            0: 'depression',
            1: 'normal',
        },
        'data_format': 'standard',  # Standard format: 'signal' and 'label' keys
        'splits': {'train': 'eeg_segments.lmdb', 'val': 'eeg_segments.lmdb', 'test': 'eeg_segments.lmdb'},
        'single_lmdb': True,  # All splits come from same LMDB
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'cross_subject': True,  # Enable cross-subject split by default
    },
    'CVD': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/cvd_normal_preprocessed_CBramod',
        'num_classes': 2,
        'task_type': 'binary',
        'n_channels': 129,  # EGI 128 channels + reference (after excluding stim)
        'sampling_rate': 200,  # Resampled to 200 Hz (matching CBraMod)
        'segment_duration': 5,  # seconds (matching CBraMod's 5-second segments)
        'patch_size': 200,
        'label_names': {
            0: 'CVD',
            1: 'normal',
        },
        'data_format': 'standard',  # Standard format: 'signal' and 'label' keys
        'splits': {'train': 'eeg_segments.lmdb', 'val': 'eeg_segments.lmdb', 'test': 'eeg_segments.lmdb'},
        'single_lmdb': True,  # All splits come from same LMDB
        'train_ratio': 0.7,
        'val_ratio': 0.0,  # No validation set, use train set for validation
        'test_ratio': 0.3,
        'cross_subject': True,  # Enable cross-subject split by default
        'no_val_set': True,  # Flag to indicate no separate validation set
    },
    'CVD_DEPRESSION_NORMAL': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/cvd_normal_depression_preprocessed',
        'num_classes': 3,
        'task_type': 'multiclass',
        'n_channels': 129,  # EGI 128 channels + reference (after excluding stim)
        'sampling_rate': 200,  # Resampled to 200 Hz (matching CBraMod)
        'segment_duration': 5,  # seconds (matching CBraMod's 5-second segments)
        'patch_size': 200,
        'label_names': {
            0: 'CVD',
            1: 'normal',
            2: 'depression',
        },
        'data_format': 'standard',  # Standard format: 'signal' and 'label' keys
        'splits': {'train': 'eeg_segments.lmdb', 'val': 'eeg_segments.lmdb', 'test': 'eeg_segments.lmdb'},
        'single_lmdb': True,  # All splits come from same LMDB
        'train_ratio': 0.7,
        'val_ratio': 0.0,  # No validation set, use train set for validation
        'test_ratio': 0.3,
        'cross_subject': True,  # Enable cross-subject split by default
        'no_val_set': True,  # Flag to indicate no separate validation set
    },
    # Unified diagnosis dataset combining EGI and BP data with common 58 channels
    # Sources: eeg_CVD_EGI_83, eeg_depression_EGI_21, eeg_depression_BP_122,
    #          eeg_normal_EGI_17, eeg_normal_BP_166
    'UNIFIED_DIAGNOSIS': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/unified_diagnosis_preprocessed',
        'num_classes': 3,
        'task_type': 'multiclass',
        'n_channels': 58,  # Common channels between EGI and BP systems (10-20 standard naming)
        'sampling_rate': 200,  # Resampled to 200 Hz (matching CBraMod)
        'segment_duration': 5,  # seconds (matching CBraMod's 5-second segments)
        'patch_size': 200,
        'label_names': {
            0: 'CVD',
            1: 'depression',
            2: 'normal',
        },
        'data_format': 'standard',  # Standard format: 'signal' and 'label' keys
        'splits': {'train': 'eeg_segments.lmdb', 'val': 'eeg_segments.lmdb', 'test': 'eeg_segments.lmdb'},
        'single_lmdb': True,  # All splits come from same LMDB
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'cross_subject': True,  # Enable cross-subject split by default
    },
    # AD diagnosis dataset: 4-class (AD, CVD, depression, normal)
    # Sources: eeg_AD+MCI+SCD+HC_EGI_124 (.mat), eeg_CVD_EGI_83, eeg_depression_EGI_21,
    #          eeg_depression_BP_122, eeg_normal_EGI_17, eeg_normal_BP_166
    'AD_DIAGNOSIS': {
        'data_dir': '/projects/EEG-foundation-model/diagnosis_data/ad_diagnosis_preprocessed',
        'num_classes': 4,
        'task_type': 'multiclass',
        'n_channels': 58,  # 58 common channels (10-20 standard naming)
        'sampling_rate': 200,  # Resampled to 200 Hz
        'segment_duration': 5,  # seconds
        'patch_size': 200,
        'label_names': {
            0: 'AD',
            1: 'CVD',
            2: 'depression',
            3: 'normal',
        },
        'data_format': 'standard',  # Standard format: 'signal' and 'label' keys
        'splits': {'train': 'eeg_segments.lmdb', 'val': 'eeg_segments.lmdb', 'test': 'eeg_segments.lmdb'},
        'single_lmdb': True,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'cross_subject': True,  # Enable cross-subject split by default
    },
}

DEFAULT_WEIGHTS_PATH = "/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth"
DEFAULT_CODEBRAIN_WEIGHTS_PATH = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"
DEFAULT_LUNA_WEIGHTS_PATH = "/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors"
DEFAULT_FEMBA_WEIGHTS_PATH = None  # FEMBA only has task-specific weights, no pure pretrained backbone
DEFAULT_OUTPUT_DIR = "/home/infres/yinwang/eeg2025/NIPS/checkpoints"

# Standard 10-20 bipolar montage channel locations (3D coordinates)
# for 16-channel TUH EEG data (TUEV/TUAB).
# Computed as midpoint of the two electrodes in each bipolar pair.
# Using MNE standard_1005 montage positions.
TUEG_BIPOLAR_CHANNELS = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
]

def _compute_bipolar_channel_locations():
    """Compute 3D channel locations for 16-ch bipolar montage using MNE standard_1005."""
    try:
        import mne
        # Get unique electrode names
        electrodes = list(set([part for ch in TUEG_BIPOLAR_CHANNELS for part in ch.split('-')]))
        ch_types = ['eeg'] * len(electrodes)
        info = mne.create_info(ch_names=electrodes, sfreq=200, ch_types=ch_types)
        info = info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
        positions = info.get_montage().get_positions()['ch_pos']

        locs = []
        for ch_name in TUEG_BIPOLAR_CHANNELS:
            e1, e2 = ch_name.split('-')
            loc1 = positions[e1]
            loc2 = positions[e2]
            locs.append((loc1 + loc2) / 2.0)
        return np.array(locs, dtype=np.float32)  # (16, 3)
    except Exception as e:
        print(f"Warning: Could not compute channel locations via MNE ({e}). Using dummy locations.")
        # Fallback: arrange channels uniformly on a sphere
        locs = []
        for i in range(16):
            theta = 2 * np.pi * i / 16
            locs.append([np.cos(theta) * 0.085, np.sin(theta) * 0.085, 0.0])
        return np.array(locs, dtype=np.float32)


# ============================================================================
# LMDB Dataset
# ============================================================================
# class EEGLMDBDataset(Dataset):
#     """Generic EEG Dataset reading from LMDB files."""

#     def __init__(self, lmdb_path, dataset_config, split='train', val_ratio=0.15,
#                  is_val_from_train=False):
#         """
#         Initialize EEG LMDB Dataset.

#         Args:
#             lmdb_path: Path to the LMDB directory
#             dataset_config: Configuration dict for the dataset
#             split: 'train', 'val', or 'test'
#             val_ratio: Ratio of training data to use for validation (if val_from_train)
#             is_val_from_train: Whether to split val from train data
#         """
#         self.lmdb_path = Path(lmdb_path)
#         self.config = dataset_config
#         self.split = split
#         self.val_ratio = val_ratio
#         self.is_val_from_train = is_val_from_train

#         # Target parameters for CBraMod
#         self.target_patch_size = 200
#         self.target_n_channels = 16  # Use first 16 channels if more available

#         # Open LMDB environment
#         self.env = lmdb.open(
#             str(self.lmdb_path),
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False
#         )

#         # Load metadata
#         with self.env.begin() as txn:
#             self._length = pickle.loads(txn.get('__length__'.encode()))
#             self._metadata = pickle.loads(txn.get('__metadata__'.encode()))

#         # Create index mapping for train/val split if needed
#         self._setup_indices()

#         print(f"Loaded {self.split} split with {len(self.indices)} samples from {self.lmdb_path}")
#         print(f"Label distribution in metadata: {self._metadata.get('label_counts', 'N/A')}")

#     def _setup_indices(self):
#         """Setup indices for train/val split."""
#         total_indices = list(range(self._length))

#         if self.is_val_from_train:
#             # Split this LMDB into train and val
#             np.random.seed(42)  # Fixed seed for reproducibility
#             np.random.shuffle(total_indices)

#             n_val = int(len(total_indices) * self.val_ratio)
#             if self.split == 'val':
#                 self.indices = total_indices[:n_val]
#             else:  # train
#                 self.indices = total_indices[n_val:]
#         else:
#             # Use all data from this LMDB
#             self.indices = total_indices

#     def __len__(self):
#         return len(self.indices)

#     def _resample_signal(self, signal, orig_rate, target_rate):
#         """Resample signal from orig_rate to target_rate."""
#         if orig_rate == target_rate:
#             return signal

#         from scipy import signal as scipy_signal
#         n_channels, n_samples = signal.shape
#         target_samples = int(n_samples * target_rate / orig_rate)

#         resampled = np.zeros((n_channels, target_samples), dtype=np.float32)
#         for ch in range(n_channels):
#             resampled[ch] = scipy_signal.resample(signal[ch], target_samples)

#         return resampled

#     def __getitem__(self, idx):
#         actual_idx = self.indices[idx]
#         key = f'{actual_idx:08d}'.encode()

#         with self.env.begin() as txn:
#             value = txn.get(key)
#             if value is None:
#                 raise KeyError(f"Key {actual_idx} not found")
#             sample = pickle.loads(value)

#         signal = sample['signal']
#         label = sample['label']

#         # Get original parameters
#         orig_n_channels = signal.shape[0]
#         orig_n_samples = signal.shape[1]
#         orig_rate = self.config['sampling_rate']

#         # Resample if needed (to 200 Hz for CBraMod)
#         if orig_rate != 200:
#             signal = self._resample_signal(signal, orig_rate, 200)

#         # Use only first 16 channels if more available
#         if orig_n_channels > self.target_n_channels:
#             signal = signal[:self.target_n_channels, :]
#         elif orig_n_channels < self.target_n_channels:
#             # Pad with zeros if fewer channels
#             padded = np.zeros((self.target_n_channels, signal.shape[1]), dtype=np.float32)
#             padded[:orig_n_channels, :] = signal
#             signal = padded

#         # Calculate sequence length (number of patches)
#         n_samples = signal.shape[1]
#         seq_len = n_samples // self.target_patch_size

#         # Reshape for CBraMod: (n_channels, n_samples) -> (n_channels, seq_len, patch_size)
#         signal = signal[:, :seq_len * self.target_patch_size]
#         signal = signal.reshape(self.target_n_channels, seq_len, self.target_patch_size)

#         # Normalize (divide by 100 as in original CBraMod)
#         signal = signal / 100.0

#         return signal.astype(np.float32), int(label)

#     @staticmethod
#     def collate_fn(batch):
#         """Collate function for DataLoader."""
#         x_data = np.array([x[0] for x in batch])
#         y_label = np.array([x[1] for x in batch])
#         return torch.from_numpy(x_data).float(), torch.from_numpy(y_label).long()

#     @property
#     def metadata(self):
#         return self._metadata

#     def close(self):
#         if self.env is not None:
#             self.env.close()

from pathlib import Path
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
# 将 scipy 导入放在类外部，避免在 __getitem__ 中反复导入变慢
try:
    from scipy import signal as scipy_signal
except (ImportError, AttributeError):
    scipy_signal = None

class EEGLMDBDataset(Dataset):
    """Generic EEG Dataset reading from LMDB files with multi-processing support."""

    def __init__(self, lmdb_path, dataset_config, split='train', val_ratio=0.15,
                 is_val_from_train=False, train_ratio=0.7, test_ratio=0.15,
                 include_labels=None, exclude_labels=None, cross_subject=False):
        """
        Initialize EEG LMDB Dataset.

        Args:
            lmdb_path: Path to LMDB file
            dataset_config: Dataset configuration dict
            split: 'train', 'val', or 'test'
            val_ratio: Ratio for validation split
            is_val_from_train: Whether to split val from train
            train_ratio: Ratio for train split (single_lmdb mode)
            test_ratio: Ratio for test split (single_lmdb mode)
            include_labels: List of label indices to include (None = include all)
            exclude_labels: List of label indices to exclude (None = exclude none)
            cross_subject: If True, split by subject instead of by sample
        """
        self.lmdb_path = Path(lmdb_path)
        self.config = dataset_config
        self.split = split
        self.val_ratio = val_ratio
        self.is_val_from_train = is_val_from_train
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.include_labels = include_labels
        self.exclude_labels = exclude_labels or []
        self.cross_subject = cross_subject

        # Target parameters for CBraMod
        self.target_patch_size = 200
        # Use original n_channels from config (support 58 channels for DIAGNOSIS)
        self.target_n_channels = dataset_config.get('n_channels', 16)

        # Check data format (standard or diagnosis)
        self.data_format = dataset_config.get('data_format', 'standard')
        self.disease_label_map = dataset_config.get('disease_label_map', None)
        self.single_lmdb = dataset_config.get('single_lmdb', False)

        # Label remapping for filtered labels (maps original label -> new sequential label)
        self.label_remap = None

        # 重要修改：在 __init__ 中不要打开 env，只初始化为 None
        self.env = None

        # 仅为了读取 metadata 和 length，临时打开一次 LMDB
        # 读取完毕后立刻关闭，防止被 DataLoader 的子进程继承
        tmp_env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        with tmp_env.begin() as txn:
            self._length = pickle.loads(txn.get('__length__'.encode()))
            meta_data = txn.get('__metadata__'.encode())
            if meta_data:
                self._metadata = pickle.loads(meta_data)
            else:
                self._metadata = {}

        tmp_env.close() # 必须关闭临时句柄

        # Create index mapping for train/val/test split if needed
        if cross_subject and self.single_lmdb:
            self._setup_cross_subject_indices()
        else:
            self._setup_indices()

        # Filter by labels if needed (for DIAGNOSIS dataset)
        if self.data_format == 'diagnosis' and (include_labels is not None or exclude_labels):
            self._filter_by_labels(tmp_env=None)

        print(f"Loaded {self.split} split with {len(self.indices)} samples from {self.lmdb_path}")
        print(f"Label distribution in metadata: {self._metadata.get('label_counts', 'N/A')}")

    def _setup_indices(self):
        """Setup indices for train/val/test split."""
        total_indices = list(range(self._length))

        if self.single_lmdb:
            # Split single LMDB into train/val/test
            np.random.seed(42)  # Fixed seed for reproducibility
            np.random.shuffle(total_indices)

            n_train = int(len(total_indices) * self.train_ratio)
            n_val = int(len(total_indices) * self.val_ratio)

            # Check if no validation set (val_ratio == 0)
            no_val_set = self.val_ratio == 0 or self.config.get('no_val_set', False)

            if self.split == 'train':
                self.indices = total_indices[:n_train]
            elif self.split == 'val':
                if no_val_set:
                    # Use train set as validation set when no separate val set
                    self.indices = total_indices[:n_train]
                else:
                    self.indices = total_indices[n_train:n_train + n_val]
            else:  # test
                if no_val_set:
                    # When no val set, test starts right after train
                    self.indices = total_indices[n_train:]
                else:
                    self.indices = total_indices[n_train + n_val:]
        elif self.is_val_from_train:
            # Split this LMDB into train and val
            np.random.seed(42)  # Fixed seed for reproducibility
            np.random.shuffle(total_indices)

            n_val = int(len(total_indices) * self.val_ratio)
            if self.split == 'val':
                self.indices = total_indices[:n_val]
            else:  # train
                self.indices = total_indices[n_val:]
        else:
            # Use all data from this LMDB
            self.indices = total_indices

    def _setup_cross_subject_indices(self):
        """Setup indices for cross-subject train/val/test split.

        This method splits data by subject to ensure no subject appears in multiple splits.
        This is important for fair evaluation of generalization to unseen subjects.

        Supports datasets with 2, 3, or 4 classes:
        - DEPRESSION: uses 'depression_subjects' and 'normal_subjects'
        - CVD: uses 'cvd_subjects' and 'normal_subjects'
        - CVD_DEPRESSION_NORMAL / UNIFIED_DIAGNOSIS: uses 'cvd_subjects', 'normal_subjects', 'depression_subjects'
        - AD_DIAGNOSIS: uses 'ad_subjects', 'cvd_subjects', 'depression_subjects', 'normal_subjects'
        """
        # Get subject info from metadata
        subjects = self._metadata.get('subjects', [])
        ad_subjects = self._metadata.get('ad_subjects', [])
        normal_subjects = self._metadata.get('normal_subjects', [])
        cvd_subjects = self._metadata.get('cvd_subjects', [])
        depression_subjects = self._metadata.get('depression_subjects', [])

        if not subjects:
            print("Warning: No subject info in metadata, falling back to random split")
            self._setup_indices()
            return

        # Shuffle subjects with fixed seed
        np.random.seed(42)

        no_val_set = self.val_ratio == 0 or self.config.get('no_val_set', False)

        # Determine number of classes from available subject groups
        is_4class = len(ad_subjects) > 0 and len(cvd_subjects) > 0 and len(depression_subjects) > 0 and len(normal_subjects) > 0
        is_3class = not is_4class and len(cvd_subjects) > 0 and len(depression_subjects) > 0 and len(normal_subjects) > 0

        # Helper function to split a subject list into train/val/test
        def split_subject_list(subj_list):
            subj_list = list(subj_list)
            np.random.shuffle(subj_list)
            n_train = int(len(subj_list) * self.train_ratio)
            n_val = int(len(subj_list) * self.val_ratio)
            if self.split == 'train':
                return subj_list[:n_train]
            elif self.split == 'val':
                if no_val_set:
                    return subj_list[:n_train]
                else:
                    return subj_list[n_train:n_train + n_val]
            else:  # test
                if no_val_set:
                    return subj_list[n_train:]
                else:
                    return subj_list[n_train + n_val:]

        if is_4class:
            # 4-class: AD, CVD, depression, normal
            split_subjects = set(
                split_subject_list(ad_subjects) +
                split_subject_list(cvd_subjects) +
                split_subject_list(depression_subjects) +
                split_subject_list(normal_subjects)
            )
        elif is_3class:
            # 3-class: CVD, depression, normal
            split_subjects = set(
                split_subject_list(cvd_subjects) +
                split_subject_list(depression_subjects) +
                split_subject_list(normal_subjects)
            )
        else:
            # 2-class: CVD/depression vs normal
            class0_subjects = cvd_subjects if cvd_subjects else depression_subjects
            split_subjects = set(
                split_subject_list(class0_subjects) +
                split_subject_list(normal_subjects)
            )

        print(f"  Cross-subject split: {self.split} has {len(split_subjects)} subjects")
        print(f"    Subjects: {sorted(split_subjects)}")

        # Now scan LMDB to get indices for these subjects
        tmp_env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        self.indices = []
        with tmp_env.begin(write=False) as txn:
            for idx in tqdm(range(self._length), desc=f"Building {self.split} indices", leave=False):
                key = f'{idx:08d}'.encode()
                value = txn.get(key)
                if value is None:
                    continue
                sample = pickle.loads(value)

                # Get subject_id from sample
                subject_id = sample.get('subject_id', None)
                if subject_id is None:
                    # Fallback: try to extract from source_file
                    source_file = sample.get('source_file', '')
                    # This is a simplified extraction, may need adjustment
                    subject_id = source_file

                if subject_id in split_subjects:
                    self.indices.append(idx)

        tmp_env.close()

        print(f"  {self.split} split: {len(self.indices)} samples from {len(split_subjects)} subjects")

    def _filter_by_labels(self, tmp_env=None):
        """Filter indices by label, keeping only samples with desired labels."""
        # Need to scan all data to filter by labels
        close_env = False
        if tmp_env is None:
            tmp_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            close_env = True

        # Determine which labels to keep
        if self.include_labels is not None:
            valid_labels = set(self.include_labels)
        else:
            # Get all possible labels from disease_label_map
            all_labels = set(self.disease_label_map.values())
            valid_labels = all_labels - set(self.exclude_labels)

        # Create label remapping (original label -> new sequential label)
        sorted_valid_labels = sorted(valid_labels)
        self.label_remap = {orig: new for new, orig in enumerate(sorted_valid_labels)}

        print(f"  Filtering labels: keeping {sorted_valid_labels}, remap: {self.label_remap}")

        # Filter indices
        filtered_indices = []
        with tmp_env.begin(write=False) as txn:
            for idx in tqdm(self.indices, desc=f"Filtering {self.split} by labels", leave=False):
                key = f'{idx:08d}'.encode()
                value = txn.get(key)
                if value is None:
                    continue
                sample = pickle.loads(value)

                # Get label based on data format
                if self.data_format == 'diagnosis':
                    disease_str = sample['labels']['disease']
                    label = self.disease_label_map[disease_str]
                else:
                    label = sample['label']

                if label in valid_labels:
                    filtered_indices.append(idx)

        if close_env:
            tmp_env.close()

        print(f"  Filtered {self.split}: {len(self.indices)} -> {len(filtered_indices)} samples")
        self.indices = filtered_indices

    def _init_db(self):
        """
        重要修改：专门用于子进程初始化的函数。
        当 DataLoader 的子进程启动时，会调用这个函数来建立独立的数据库连接。
        """
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

    def __len__(self):
        return len(self.indices)

    def _resample_signal(self, signal, orig_rate, target_rate):
        """Resample signal from orig_rate to target_rate."""
        if orig_rate == target_rate:
            return signal

        n_channels, n_samples = signal.shape
        target_samples = int(n_samples * target_rate / orig_rate)

        resampled = np.zeros((n_channels, target_samples), dtype=np.float32)
        for ch in range(n_channels):
            resampled[ch] = scipy_signal.resample(signal[ch], target_samples)

        return resampled

    def __getitem__(self, idx):
        # 重要修改：延迟初始化。如果当前进程还没打开 LMDB，则打开它。
        if self.env is None:
            self._init_db()

        actual_idx = self.indices[idx]
        key = f'{actual_idx:08d}'.encode()

        # 明确指明 write=False
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {actual_idx} not found")
            sample = pickle.loads(value)

        # Handle different data formats
        if self.data_format == 'diagnosis':
            # DIAGNOSIS format: 'data' key and 'labels' dict with 'disease' key
            signal = sample['data']
            disease_str = sample['labels']['disease']
            label = self.disease_label_map[disease_str]
        else:
            # Standard format: 'signal' and 'label' keys
            signal = sample['signal']
            label = sample['label']

        # Apply label remapping if filtering was used
        if self.label_remap is not None:
            label = self.label_remap[label]

        # Get original parameters
        orig_n_channels = signal.shape[0]
        orig_n_samples = signal.shape[1]
        orig_rate = self.config['sampling_rate']

        # Resample if needed (to 200 Hz for CBraMod)
        if orig_rate != 200:
            signal = self._resample_signal(signal, orig_rate, 200)

        # Handle channel count
        current_n_channels = signal.shape[0]
        if current_n_channels > self.target_n_channels:
            signal = signal[:self.target_n_channels, :]
        elif current_n_channels < self.target_n_channels:
            # Pad with zeros if fewer channels
            padded = np.zeros((self.target_n_channels, signal.shape[1]), dtype=np.float32)
            padded[:current_n_channels, :] = signal
            signal = padded

        # Calculate sequence length (number of patches)
        n_samples = signal.shape[1]
        seq_len = n_samples // self.target_patch_size

        # Handle case where segment is too short for even 1 patch
        if seq_len == 0:
            # Pad to at least 1 patch
            padded = np.zeros((self.target_n_channels, self.target_patch_size), dtype=np.float32)
            padded[:, :n_samples] = signal[:self.target_n_channels, :]
            signal = padded.reshape(self.target_n_channels, 1, self.target_patch_size)
        else:
            # Reshape for CBraMod: (n_channels, n_samples) -> (n_channels, seq_len, patch_size)
            signal = signal[:, :seq_len * self.target_patch_size]
            signal = signal.reshape(self.target_n_channels, seq_len, self.target_patch_size)

        # Normalize (divide by 100 as in original CBraMod)
        signal = signal / 100.0

        return signal.astype(np.float32), int(label)

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return torch.from_numpy(x_data).float(), torch.from_numpy(y_label).long()

    @property
    def metadata(self):
        return self._metadata

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
# ============================================================================
# Model
# ============================================================================
class EEGModel(nn.Module):
    """CBraMod model for EEG classification."""

    def __init__(self, num_classes, task_type='multiclass', pretrained_weights_path=None,
                 finetuned_ckpt_path=None,
                 classifier_type='all_patch_reps', dropout=0.1, device='cuda:0',
                 n_channels=16, seq_len=5, patch_size=200):
        super().__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.task_type = task_type
        self.num_classes = num_classes

        # Initialize backbone
        self.backbone = CBraMod(
            in_dim=patch_size,       # 200
            out_dim=patch_size,      # 200
            d_model=200,
            dim_feedforward=800,
            seq_len=30,              # Max sequence length
            n_layer=12,
            nhead=8
        )

        # Load weights: finetuned checkpoint takes priority over pretrained weights
        map_location = torch.device(device)
        if finetuned_ckpt_path and os.path.exists(finetuned_ckpt_path):
            # Load from a finetuned EEGModel checkpoint (contains backbone.* and classifier.* keys)
            # Only load backbone weights, discard classifier head
            print(f"Loading backbone from finetuned checkpoint: {finetuned_ckpt_path}")
            ckpt = torch.load(finetuned_ckpt_path, map_location=map_location)
            backbone_state = {}
            for k, v in ckpt.items():
                if k.startswith('backbone.'):
                    backbone_state[k[len('backbone.'):]] = v
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            if missing:
                print(f"  Missing keys in backbone: {missing}")
            if unexpected:
                print(f"  Unexpected keys in backbone: {unexpected}")
            print(f"  Loaded {len(backbone_state)} backbone parameters, classifier head discarded")
        elif pretrained_weights_path and os.path.exists(pretrained_weights_path):
            # Load original CBraMod pretrained weights (backbone-only state_dict)
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            self.backbone.load_state_dict(torch.load(pretrained_weights_path, map_location=map_location))
        else:
            print("No pretrained weights loaded, training from scratch")

        # Replace projection layer with identity for downstream task
        self.backbone.proj_out = nn.Identity()

        # Build classifier
        # Input dim: n_channels * seq_len * d_model = 16 * 5 * 200 = 16000
        input_dim = n_channels * seq_len * 200

        # Output dim depends on task type
        if task_type == 'binary':
            output_dim = 1
        else:
            output_dim = num_classes

        if classifier_type == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, output_dim),
            )
        elif classifier_type == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, output_dim),
            )
        elif classifier_type == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, output_dim),
            )
        elif classifier_type == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, seq_len * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, output_dim),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, x):
        # x shape: (batch, channels, seq_len, patch_size) = (B, 16, 5, 200)
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


# ============================================================================
# CodeBrain Model (SSSM backbone)
# ============================================================================
class CodeBrainModel(nn.Module):
    """CodeBrain SSSM model for EEG classification."""

    def __init__(self, num_classes, task_type='multiclass', pretrained_weights_path=None,
                 finetuned_ckpt_path=None,
                 classifier_type='all_patch_reps', dropout=0.1, device='cuda:0',
                 n_channels=16, seq_len=5, patch_size=200,
                 n_layer=8, codebook_size_t=4096, codebook_size_f=4096):
        super().__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.task_type = task_type
        self.num_classes = num_classes

        # s4_lmax depends on channel count and seq_len: channels * seq_len * ~(570/16/5)
        # Original CodeBrain uses s4_lmax=570 for 16ch * 5seq (i.e. 80 patches => factor ~7)
        # We scale proportionally: s4_lmax = n_channels * seq_len * ceil(570 / (16*5))
        # But actually s4_lmax just needs to be >= n_channels * seq_len
        s4_lmax = n_channels * seq_len
        # Round up to nearest multiple of 19 (used by PatchEmbedding positional conv kernel)
        s4_lmax = ((s4_lmax + 18) // 19) * 19

        # Initialize SSSM backbone
        self.backbone = SSSM(
            in_channels=200,
            res_channels=200,
            skip_channels=200,
            out_channels=200,
            num_res_layers=n_layer,
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=s4_lmax,
            s4_d_state=64,
            s4_dropout=dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=codebook_size_t,
            codebook_size_f=codebook_size_f,
            if_codebook=False,
        )

        # Load weights: finetuned checkpoint takes priority over pretrained weights
        map_location = torch.device(device)
        if finetuned_ckpt_path and os.path.exists(finetuned_ckpt_path):
            print(f"Loading CodeBrain backbone from finetuned checkpoint: {finetuned_ckpt_path}")
            ckpt = torch.load(finetuned_ckpt_path, map_location=map_location)
            backbone_state = {}
            for k, v in ckpt.items():
                if k.startswith('backbone.'):
                    backbone_state[k[len('backbone.'):]] = v
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            if missing:
                print(f"  Missing keys in backbone: {missing}")
            if unexpected:
                print(f"  Unexpected keys in backbone: {unexpected}")
            print(f"  Loaded {len(backbone_state)} backbone parameters, classifier head discarded")
        elif pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"Loading CodeBrain pretrained weights from {pretrained_weights_path}")
            state_dict = torch.load(pretrained_weights_path, map_location=map_location)
            # Remove 'module.' prefix from DataParallel if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
        else:
            print("No pretrained weights loaded, training CodeBrain from scratch")

        # Disable the original projection output (codebook heads)
        self.backbone.proj_out = nn.Sequential()

        # Build classifier
        # SSSM output shape: (batch, n_channels, seq_len, patch_size=200)
        # Flattened: n_channels * seq_len * 200
        input_dim = n_channels * seq_len * 200

        # Output dim depends on task type
        if task_type == 'binary':
            output_dim = 1
        else:
            output_dim = num_classes

        if classifier_type == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, output_dim),
            )
        elif classifier_type == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, output_dim),
            )
        elif classifier_type == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, output_dim),
            )
        elif classifier_type == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, seq_len * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, output_dim),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, x):
        # x shape: (batch, channels, seq_len, patch_size) = (B, 16, 5, 200)
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


# ============================================================================
# FEMBA Model (Bidirectional Mamba backbone)
# ============================================================================
class FEMBAModel(nn.Module):
    """FEMBA (Foundational Encoder Model with Bidirectional Mamba) for EEG classification.

    FEMBA operates on raw (B, C, T) EEG signals with its own PatchEmbed + BiMamba blocks.
    Input from our data pipeline is (B, C, S, P) which we reshape to (B, C, S*P) = (B, C, T).

    Architecture:
        PatchEmbed(Conv2d) -> pos_embed -> [BiMamba + residual + LayerNorm] x num_blocks -> MambaClassifier

    Key differences from CodeBrain/CBraMod:
        - Uses its own patching (Conv2d-based PatchEmbed), not the 200-sample patches
        - Bidirectional Mamba blocks instead of Transformer/SSSM
        - Internal classifier head (MambaClassifier: FC->GELU->Mamba->pool->FC)
        - Requires fake mask tensor (all zeros for classification)
        - Layerwise LR decay supported on mamba_blocks and norm_layers
    """

    def __init__(self, num_classes, task_type='multiclass', pretrained_weights_path=None,
                 finetuned_ckpt_path=None,
                 classifier_type='all_patch_reps', dropout=0.1, device='cuda:0',
                 n_channels=16, seq_len=5, patch_size=200,
                 # FEMBA-specific parameters
                 femba_embed_dim=79, femba_num_blocks=2, femba_exp=4,
                 femba_patch_size=(2, 16), femba_stride=(2, 16),
                 femba_classification_type='mcc',
                 use_builtin_classifier=True):
        super().__init__()

        if not FEMBA_AVAILABLE:
            raise ImportError("FEMBA requires mamba_ssm. Install with: pip install mamba-ssm")

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.task_type = task_type
        self.num_classes = num_classes
        self.use_builtin_classifier = use_builtin_classifier

        # Compute actual temporal length from our patched data
        self.temporal_length = seq_len * patch_size  # e.g. 5*200=1000 for TUEV, 10*200=2000 for TUAB

        # Map our task_type to FEMBA classification_type
        if task_type == 'binary':
            femba_classification_type = 'bc'
            femba_num_classes = 2
        else:
            femba_classification_type = 'mcc'
            femba_num_classes = num_classes

        if use_builtin_classifier:
            # Use FEMBA's built-in MambaClassifier
            self.backbone = FEMBA(
                seq_length=self.temporal_length,
                num_channels=n_channels,
                num_classes=femba_num_classes,
                embed_dim=femba_embed_dim,
                num_blocks=femba_num_blocks,
                exp=femba_exp,
                patch_size=femba_patch_size,
                stride=femba_stride,
                classification_type=femba_classification_type,
            )
            self.classifier = None  # Using built-in classifier
        else:
            # Use FEMBA encoder only + our own classifier head
            self.backbone = FEMBA(
                seq_length=self.temporal_length,
                num_channels=n_channels,
                num_classes=0,  # No classifier => encoder-only mode (decoder mode)
                embed_dim=femba_embed_dim,
                num_blocks=femba_num_blocks,
                exp=femba_exp,
                patch_size=femba_patch_size,
                stride=femba_stride,
                classification_type=femba_classification_type,
            )
            # Build external classifier head on top of FEMBA encoder output
            grid_size = self.backbone.patch_embed.grid_size
            encoder_out_dim = grid_size[0] * femba_embed_dim * grid_size[1]
            if task_type == 'binary':
                output_dim = 1
            else:
                output_dim = num_classes
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(encoder_out_dim, 256),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(256, output_dim),
            )

        # Load pretrained weights
        self._load_weights(pretrained_weights_path, finetuned_ckpt_path, device)

    def _load_weights(self, pretrained_weights_path, finetuned_ckpt_path, device):
        """Load pretrained weights from safetensors or pytorch checkpoint."""
        from safetensors.torch import load_file
        map_location = torch.device(device)

        if finetuned_ckpt_path and os.path.exists(finetuned_ckpt_path):
            print(f"Loading FEMBA from finetuned checkpoint: {finetuned_ckpt_path}")
            if finetuned_ckpt_path.endswith('.safetensors'):
                state_dict = load_file(finetuned_ckpt_path)
            else:
                ckpt = torch.load(finetuned_ckpt_path, map_location=map_location)
                state_dict = ckpt.get('state_dict', ckpt)
            # Try loading into the backbone (may have 'model.' prefix from PL)
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_sd[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_sd[k] = v
            missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)} (likely classifier mismatch)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        elif pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"Loading FEMBA pretrained weights from {pretrained_weights_path}")
            if pretrained_weights_path.endswith('.safetensors'):
                state_dict = load_file(pretrained_weights_path)
            else:
                state_dict = torch.load(pretrained_weights_path, map_location=map_location)
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_sd[k[6:]] = v
                else:
                    new_sd[k] = v
            missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("No pretrained weights loaded for FEMBA, training from scratch")

    def forward(self, x):
        # x shape: (B, C, S, P) = (B, 16, 5, 200) from our data pipeline
        B, C, S, P = x.shape
        # Reshape to (B, C, T) for FEMBA
        x_raw = x.reshape(B, C, S * P)  # (B, 16, 1000) or (B, 16, 2000)

        # Generate fake mask (all zeros = no masking for classification)
        mask = torch.zeros(B, C, S * P, dtype=torch.bool, device=x.device)

        if self.use_builtin_classifier:
            logits, _ = self.backbone(x_raw, mask)
            return logits
        else:
            # Encoder-only mode: backbone returns (reconstructed, original)
            encoded, _ = self.backbone(x_raw, mask)
            return self.classifier(encoded)


# ============================================================================
# LUNA Model (Query-based Cross-Attention + RoPE Transformer backbone)
# ============================================================================
class LUNAModel(nn.Module):
    """LUNA (Location-UNaware Attention) model for EEG classification.

    LUNA is topology-agnostic: it uses learned queries to compress variable-count
    channels via cross-attention, then processes temporal patches with RoPE Transformer.
    Requires 3D channel_locations for each electrode.

    Architecture:
        PatchEmbed + FreqEmbed -> CrossAttention(channels->queries) -> RoPE Transformer blocks
        -> ClassificationHead (learned aggregation query -> attention -> FFN)

    Key differences from CodeBrain/CBraMod:
        - Topology-agnostic via learned queries compressing channels
        - Uses 3D electrode coordinates for spatial encoding (NeRF positional encoding)
        - RoPE (Rotary Position Embedding) in Transformer blocks
        - Operates on raw (B, C, T) EEG + channel_locations (B, C, 3)
        - Has its own ClassificationHead using learned aggregation query + attention
    """

    def __init__(self, num_classes, task_type='multiclass', pretrained_weights_path=None,
                 finetuned_ckpt_path=None,
                 classifier_type='all_patch_reps', dropout=0.1, device='cuda:0',
                 n_channels=16, seq_len=5, patch_size=200,
                 # LUNA-specific parameters
                 luna_patch_size=40, luna_embed_dim=64, luna_depth=8,
                 luna_num_heads=2, luna_num_queries=4, luna_drop_path=0.1,
                 channel_locations=None):
        super().__init__()

        if not LUNA_AVAILABLE:
            raise ImportError("LUNA requires timm and rotary-embedding-torch. "
                              "Install with: pip install timm rotary-embedding-torch")

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.task_type = task_type
        self.num_classes = num_classes
        self.temporal_length = seq_len * patch_size

        # Map task type to LUNA num_classes
        if task_type == 'binary':
            luna_num_classes = 1  # Binary: single logit for BCEWithLogitsLoss
        else:
            luna_num_classes = num_classes

        # Initialize LUNA backbone with classification head
        self.backbone = LUNA(
            patch_size=luna_patch_size,
            embed_dim=luna_embed_dim,
            depth=luna_depth,
            num_heads=luna_num_heads,
            num_queries=luna_num_queries,
            drop_path=luna_drop_path,
            num_classes=luna_num_classes,
        )

        # Register channel locations as buffer (not a trainable parameter)
        if channel_locations is not None:
            self.register_buffer('channel_locations',
                                 torch.tensor(channel_locations, dtype=torch.float32))
        else:
            # Compute from standard 10-20 bipolar montage
            locs = _compute_bipolar_channel_locations()
            self.register_buffer('channel_locations',
                                 torch.tensor(locs, dtype=torch.float32))

        # Load pretrained weights
        self._load_weights(pretrained_weights_path, finetuned_ckpt_path, device)

    def _load_weights(self, pretrained_weights_path, finetuned_ckpt_path, device):
        """Load pretrained weights from safetensors or pytorch checkpoint."""
        from safetensors.torch import load_file
        map_location = torch.device(device)

        if finetuned_ckpt_path and os.path.exists(finetuned_ckpt_path):
            print(f"Loading LUNA from finetuned checkpoint: {finetuned_ckpt_path}")
            if finetuned_ckpt_path.endswith('.safetensors'):
                state_dict = load_file(finetuned_ckpt_path)
            else:
                ckpt = torch.load(finetuned_ckpt_path, map_location=map_location)
                state_dict = ckpt.get('state_dict', ckpt)
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_sd[k[6:]] = v
                else:
                    new_sd[k] = v
            missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        elif pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"Loading LUNA pretrained weights from {pretrained_weights_path}")
            if pretrained_weights_path.endswith('.safetensors'):
                state_dict = load_file(pretrained_weights_path)
            else:
                state_dict = torch.load(pretrained_weights_path, map_location=map_location)
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_sd[k[6:]] = v
                else:
                    new_sd[k] = v
            # Pretrained weights are encoder-only (num_classes=0), our model has classifier
            # Load with strict=False to skip classifier weights
            missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)} (expected: classifier head)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("No pretrained weights loaded for LUNA, training from scratch")

    def forward(self, x):
        # x shape: (B, C, S, P) = (B, 16, 5, 200) from our data pipeline
        B, C, S, P = x.shape
        # Reshape to (B, C, T) for LUNA
        x_raw = x.reshape(B, C, S * P)  # (B, 16, 1000) or (B, 16, 2000)

        # Generate fake mask (all zeros = no masking for classification)
        mask = torch.zeros(B, C, S * P, dtype=torch.bool, device=x.device)

        # Expand channel locations to batch size
        ch_locs = self.channel_locations.unsqueeze(0).expand(B, -1, -1)  # (B, 16, 3)

        logits, _ = self.backbone(x_raw, mask, ch_locs)
        return logits


# ============================================================================
# t-SNE Visualization
# ============================================================================
def plot_tsne(features, labels, label_names, title="t-SNE Visualization", max_samples=2000):
    """
    Generate t-SNE visualization for features.

    Args:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
        label_names: dict mapping label indices to names
        title: plot title
        max_samples: maximum number of samples to visualize (for speed)

    Returns:
        matplotlib figure
    """
    # Subsample if too many samples
    n_samples = features.shape[0]
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # Compute t-SNE (use max_iter for sklearn >= 1.2, fallback to n_iter for older versions)
    try:
        # sklearn >= 1.2 uses max_iter
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                    random_state=42, max_iter=1000)
    except TypeError:
        # sklearn < 1.2 uses n_iter
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                    random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique labels and create color map
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names.get(label, str(label))
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=label_name, alpha=0.6, s=20)

    ax.legend(loc='best')
    ax.set_title(title)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    return fig


# ============================================================================
# Trainer
# ============================================================================
class Trainer:
    """Trainer class with wandb logging support."""

    def __init__(self, params, data_loader, model, dataset_config, use_wandb=True):
        self.params = params
        self.data_loader = data_loader
        self.model = model.cuda()
        self.dataset_config = dataset_config
        self.task_type = dataset_config['task_type']
        self.label_names = dataset_config['label_names']
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.best_model_states = None

        # Loss function based on task type
        if self.task_type == 'binary':
            self.criterion = BCEWithLogitsLoss().cuda()
        else:
            self.criterion = CrossEntropyLoss(label_smoothing=params.label_smoothing).cuda()

        # Check if linear probing mode
        self.linear_probe = getattr(params, 'linear_probe', False)

        # Setup optimizer with different learning rates
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                # Freeze backbone for both frozen mode and linear_probe mode
                if params.frozen or self.linear_probe:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.linear_probe:
            # Linear probing: only train classifier with higher learning rate
            # Use a higher learning rate since we're only training the classifier
            linear_probe_lr = params.lr if params.lr > 0.001 else 0.01
            print(f"Linear probing mode: backbone frozen, classifier LR={linear_probe_lr}")
            self.optimizer = torch.optim.AdamW(
                other_params,
                lr=linear_probe_lr,
                weight_decay=params.weight_decay
            )
        elif params.multi_lr:
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': params.lr},
                {'params': other_params, 'lr': 0.001 * (params.batch_size / 256) ** 0.5}
            ], weight_decay=params.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay
            )

        # Learning rate scheduler
        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * self.data_length,
            eta_min=1e-6
        )

        # Count trainable parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    def evaluate(self, data_loader):
        """Evaluate model and return metrics."""
        self.model.eval()
        truths = []
        preds = []
        probs = []

        for x, y in tqdm(data_loader, mininterval=1, desc="Evaluating", leave=False):
            x = x.cuda()
            y = y.cuda()

            with torch.no_grad():
                logits = self.model(x)

                if self.task_type == 'binary':
                    prob = torch.sigmoid(logits).squeeze()
                    pred_y = (prob > 0.5).long()
                    prob_np = prob.cpu().numpy()
                    if prob_np.ndim == 0:  # scalar case (batch_size=1)
                        probs.append(prob_np.item())
                    else:
                        probs.extend(prob_np.tolist())
                else:
                    pred_y = torch.max(logits, dim=-1)[1]

            y_np = y.cpu().numpy()
            pred_np = pred_y.cpu().numpy()
            if y_np.ndim == 0:  # scalar case (batch_size=1)
                truths.append(y_np.item())
                preds.append(pred_np.item())
            else:
                truths.extend(y_np.tolist())
                preds.extend(pred_np.tolist())

        truths = np.array(truths)
        preds = np.array(preds)

        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted' if self.task_type == 'multiclass' else 'binary')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)

        # Additional metrics for binary classification
        if self.task_type == 'binary':
            probs = np.array(probs)
            try:
                roc_auc = roc_auc_score(truths, probs)
                pr_auc = average_precision_score(truths, probs)
            except:
                roc_auc = 0.0
                pr_auc = 0.0
            return acc, kappa, f1, cm, roc_auc, pr_auc

        return acc, kappa, f1, cm, None, None

    def extract_features(self, data_loader, max_samples=2000):
        """Extract features from the model's backbone for visualization."""
        self.model.eval()
        all_features = []
        all_labels = []
        sample_count = 0

        for x, y in tqdm(data_loader, mininterval=1, desc="Extracting features", leave=False):
            if sample_count >= max_samples:
                break

            x = x.cuda()

            with torch.no_grad():
                # Extract backbone features before classifier
                feats = self.model.backbone(x)
                # Flatten features: (batch, channels, seq_len, d_model) -> (batch, -1)
                feats_flat = feats.reshape(feats.size(0), -1)

            all_features.append(feats_flat.cpu().numpy())
            all_labels.append(y.numpy())
            sample_count += x.size(0)

        features = np.concatenate(all_features, axis=0)[:max_samples]
        labels = np.concatenate(all_labels, axis=0)[:max_samples]

        return features, labels

    def generate_tsne_plots(self, epoch):
        """Generate t-SNE plots for train, val, and test sets and log to wandb."""
        if not self.use_wandb:
            return

        max_samples = getattr(self.params, 'tsne_samples', 2000)
        print(f"  Generating t-SNE visualizations (max {max_samples} samples per split)...")
        tsne_images = {}

        for split_name, loader in [('train', self.data_loader['train']),
                                    ('val', self.data_loader['val']),
                                    ('test', self.data_loader['test'])]:
            try:
                features, labels = self.extract_features(loader, max_samples=max_samples)
                fig = plot_tsne(
                    features, labels, self.label_names,
                    title=f"t-SNE - {split_name} (Epoch {epoch})",
                    max_samples=max_samples
                )
                tsne_images[f"tsne/{split_name}"] = wandb.Image(fig)
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: Failed to generate t-SNE for {split_name}: {e}")

        if tsne_images:
            wandb.log(tsne_images, step=epoch)

    def train(self):
        """Main training loop with wandb logging."""
        acc_best = -1
        best_epoch = 0
        best_metrics = {}

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            correct = 0
            total = 0

            pbar = tqdm(self.data_loader['train'], mininterval=10, desc=f"Epoch {epoch+1}/{self.params.epochs}")
            for x, y in pbar:
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                logits = self.model(x)

                if self.task_type == 'binary':
                    loss = self.criterion(logits.squeeze(), y.float())
                    pred_y = (torch.sigmoid(logits.squeeze()) > 0.5).long()
                else:
                    loss = self.criterion(logits, y)
                    pred_y = torch.max(logits, dim=-1)[1]

                loss.backward()
                losses.append(loss.item())

                # Calculate training accuracy
                correct += (pred_y == y).sum().item()
                total += y.size(0)

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

                # Update progress bar
                pbar.set_postfix({'loss': f'{np.mean(losses[-100:]):.4f}'})

            train_loss = np.mean(losses)
            train_acc = correct / total
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            epoch_time = (timer() - start_time) / 60

            # Validation and Test evaluation
            with torch.no_grad():
                val_acc, val_kappa, val_f1, val_cm, val_roc, val_pr = self.evaluate(self.data_loader['val'])
                test_acc, test_kappa, test_f1, test_cm, test_roc, test_pr = self.evaluate(self.data_loader['test'])

            # Logging
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/balanced_acc': val_acc,
                'val/kappa': val_kappa,
                'val/f1': val_f1,
                'test/balanced_acc': test_acc,
                'test/kappa': test_kappa,
                'test/f1': test_f1,
                'learning_rate': current_lr,
                'epoch_time_min': epoch_time,
            }

            # Add AUC metrics for binary classification
            if self.task_type == 'binary':
                log_dict.update({
                    'val/roc_auc': val_roc,
                    'val/pr_auc': val_pr,
                    'test/roc_auc': test_roc,
                    'test/pr_auc': test_pr,
                })

            if self.use_wandb:
                wandb.log(log_dict)

            # Generate t-SNE visualizations at specified intervals
            if self.params.tsne_interval > 0 and (epoch + 1) % self.params.tsne_interval == 0:
                self.generate_tsne_plots(epoch + 1)

            print(f"\nEpoch {epoch + 1}/{self.params.epochs}:")
            print(f"  Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}")
            print(f"  Val Balanced Acc: {val_acc:.5f}, Kappa: {val_kappa:.5f}, F1: {val_f1:.5f}", end='')
            if self.task_type == 'binary':
                print(f", ROC-AUC: {val_roc:.5f}, PR-AUC: {val_pr:.5f}")
            else:
                print()
            print(f"  Test Balanced Acc: {test_acc:.5f}, Kappa: {test_kappa:.5f}, F1: {test_f1:.5f}", end='')
            if self.task_type == 'binary':
                print(f", ROC-AUC: {test_roc:.5f}, PR-AUC: {test_pr:.5f}")
            else:
                print()
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f} mins")
            print(f"  Val Confusion Matrix:\n{val_cm}")

            # Save best model based on balanced accuracy
            if val_acc > acc_best:
                print(f"  Balanced Acc improved from {acc_best:.5f} to {val_acc:.5f}, saving model...")
                acc_best = val_acc
                best_epoch = epoch + 1
                best_metrics = {
                    'val_acc': val_acc,
                    'val_kappa': val_kappa,
                    'val_f1': val_f1,
                }
                self.best_model_states = copy.deepcopy(self.model.state_dict())

        # Load best model for final test
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        # Final test evaluation
        print("\n" + "=" * 60)
        print("Final Test Evaluation (Best Model)")
        print("=" * 60)

        with torch.no_grad():
            test_acc, test_kappa, test_f1, test_cm, test_roc, test_pr = self.evaluate(self.data_loader['test'])

        print(f"Test Results:")
        print(f"  Balanced Accuracy: {test_acc:.5f}")
        print(f"  Kappa: {test_kappa:.5f}")
        print(f"  F1: {test_f1:.5f}")
        if self.task_type == 'binary':
            print(f"  ROC-AUC: {test_roc:.5f}")
            print(f"  PR-AUC: {test_pr:.5f}")
        print(f"Confusion Matrix:\n{test_cm}")

        # Log final test results
        if self.use_wandb:
            final_log = {
                'final_test/balanced_acc': test_acc,
                'final_test/kappa': test_kappa,
                'final_test/f1': test_f1,
                'best_epoch': best_epoch,
            }
            if self.task_type == 'binary':
                final_log['final_test/roc_auc'] = test_roc
                final_log['final_test/pr_auc'] = test_pr
            wandb.log(final_log)

            # Log confusion matrix as table
            cm_table = wandb.Table(
                columns=[f"Pred_{self.label_names.get(i, str(i))}" for i in range(test_cm.shape[1])],
                data=[[int(x) for x in row] for row in test_cm]
            )
            wandb.log({"final_test/confusion_matrix": cm_table})

        # Save model
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)

        dataset_name = self.params.dataset.lower()
        model_path = os.path.join(
            self.params.model_dir,
            f"{dataset_name}_epoch{best_epoch}_bal_acc_{test_acc:.5f}_kappa_{test_kappa:.5f}_f1_{test_f1:.5f}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")

        return test_acc, test_kappa, test_f1


# ============================================================================
# Data Loading
# ============================================================================
def load_data(params, dataset_config):
    """Load EEG data from LMDB files based on dataset configuration."""
    data_dir = Path(dataset_config['data_dir'])
    splits = dataset_config['splits']
    single_lmdb = dataset_config.get('single_lmdb', False)
    cross_subject = dataset_config.get('cross_subject', False)

    # Allow command line override for cross_subject
    if hasattr(params, 'cross_subject') and params.cross_subject is not None:
        cross_subject = params.cross_subject

    print(f"Loading {params.dataset} data from {data_dir}")
    if cross_subject:
        print(f"Using CROSS-SUBJECT split (no subject appears in multiple splits)")

    # Handle label filtering for DIAGNOSIS dataset
    include_labels = None
    exclude_labels = None
    if params.dataset == 'DIAGNOSIS':
        if params.include_labels:
            include_labels = params.include_labels
            print(f"Including only labels: {include_labels}")
        if params.exclude_labels:
            exclude_labels = params.exclude_labels
            print(f"Excluding labels: {exclude_labels}")

    if single_lmdb:
        # All splits come from the same LMDB file
        lmdb_path = data_dir / splits['train']
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB not found: {lmdb_path}")

        train_ratio = dataset_config.get('train_ratio', 0.7)
        val_ratio = dataset_config.get('val_ratio', 0.15)
        test_ratio = dataset_config.get('test_ratio', 0.15)
        no_val_set = dataset_config.get('no_val_set', False)

        if no_val_set:
            print(f"No validation set mode: train={train_ratio*100:.0f}%, test={test_ratio*100:.0f}%")
            print(f"  Validation will use the same data as training for model selection")

        train_dataset = EEGLMDBDataset(
            lmdb_path, dataset_config, split='train',
            val_ratio=val_ratio, train_ratio=train_ratio, test_ratio=test_ratio,
            include_labels=include_labels, exclude_labels=exclude_labels,
            cross_subject=cross_subject
        )
        val_dataset = EEGLMDBDataset(
            lmdb_path, dataset_config, split='val',
            val_ratio=val_ratio, train_ratio=train_ratio, test_ratio=test_ratio,
            include_labels=include_labels, exclude_labels=exclude_labels,
            cross_subject=cross_subject
        )
        test_dataset = EEGLMDBDataset(
            lmdb_path, dataset_config, split='test',
            val_ratio=val_ratio, train_ratio=train_ratio, test_ratio=test_ratio,
            include_labels=include_labels, exclude_labels=exclude_labels,
            cross_subject=cross_subject
        )
    else:
        # Determine if val needs to be split from train
        val_from_train = (splits['train'] == splits['val'])

        # Create datasets
        train_lmdb = data_dir / splits['train']
        val_lmdb = data_dir / splits['val']
        test_lmdb = data_dir / splits['test']

        if not train_lmdb.exists():
            raise FileNotFoundError(f"Train LMDB not found: {train_lmdb}")

        train_dataset = EEGLMDBDataset(
            train_lmdb, dataset_config, split='train',
            val_ratio=params.val_ratio, is_val_from_train=val_from_train,
            include_labels=include_labels, exclude_labels=exclude_labels
        )

        if val_from_train:
            val_dataset = EEGLMDBDataset(
                val_lmdb, dataset_config, split='val',
                val_ratio=params.val_ratio, is_val_from_train=True,
                include_labels=include_labels, exclude_labels=exclude_labels
            )
        else:
            if not val_lmdb.exists():
                raise FileNotFoundError(f"Val LMDB not found: {val_lmdb}")
            val_dataset = EEGLMDBDataset(val_lmdb, dataset_config, split='val',
                                          include_labels=include_labels, exclude_labels=exclude_labels)

        if not test_lmdb.exists():
            raise FileNotFoundError(f"Test LMDB not found: {test_lmdb}")
        test_dataset = EEGLMDBDataset(test_lmdb, dataset_config, split='test',
                                       include_labels=include_labels, exclude_labels=exclude_labels)

    # Determine actual number of classes based on filtering
    if include_labels is not None:
        num_classes = len(include_labels)
    elif exclude_labels:
        num_classes = dataset_config['num_classes'] - len(exclude_labels)
    else:
        num_classes = dataset_config['num_classes']

    print(f"Task type: {dataset_config['task_type']}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    data_loader = {
        'train': DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            collate_fn=EEGLMDBDataset.collate_fn,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=params.batch_size,
            collate_fn=EEGLMDBDataset.collate_fn,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=params.batch_size,
            collate_fn=EEGLMDBDataset.collate_fn,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
    }

    # Get sequence length from a sample
    sample_batch = next(iter(data_loader['train']))
    seq_len = sample_batch[0].shape[2]
    print(f"Input shape: {sample_batch[0].shape} (batch, channels, seq_len, patch_size)")

    return data_loader, num_classes, seq_len


# ============================================================================
# Main
# ============================================================================
def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='EEG Fine-tuning with CBraMod/CodeBrain and WandB')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='TUEV',
                        choices=['TUEV', 'TUAB', 'CHB-MIT', 'TUSZ', 'DIAGNOSIS', 'DEPRESSION', 'CVD', 'CVD_DEPRESSION_NORMAL', 'UNIFIED_DIAGNOSIS', 'AD_DIAGNOSIS'],
                        help='dataset to use for finetuning')

    # Basic settings
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')

    # Training settings
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay')
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio from train data')

    # Model backbone selection
    parser.add_argument('--model', type=str, default='cbramod',
                        choices=['cbramod', 'codebrain', 'femba', 'luna'],
                        help='backbone model: cbramod (CBraMod), codebrain (CodeBrain SSSM), '
                             'femba (FEMBA BiMamba), or luna (LUNA cross-attention Transformer)')

    # Model settings
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        choices=['all_patch_reps', 'all_patch_reps_twolayer',
                                 'all_patch_reps_onelayer', 'avgpooling_patch_reps'],
                        help='classifier head type')
    parser.add_argument('--multi_lr', action='store_true', default=True,
                        help='use different LRs for backbone and classifier')
    parser.add_argument('--no_multi_lr', action='store_false', dest='multi_lr',
                        help='disable multi learning rate')
    parser.add_argument('--frozen', action='store_true', default=False,
                        help='freeze backbone weights')
    parser.add_argument('--linear_probe', action='store_true', default=False,
                        help='linear probing mode: freeze backbone and only train classifier head. '
                             'Uses higher learning rate (0.01) and simpler classifier by default.')

    # CodeBrain-specific settings
    parser.add_argument('--n_layer', type=int, default=8,
                        help='number of residual layers in CodeBrain SSSM backbone')
    parser.add_argument('--codebook_size_t', type=int, default=4096,
                        help='CodeBrain temporal codebook size')
    parser.add_argument('--codebook_size_f', type=int, default=4096,
                        help='CodeBrain frequency codebook size')

    # FEMBA-specific settings
    parser.add_argument('--femba_embed_dim', type=int, default=79,
                        help='FEMBA embedding dimension')
    parser.add_argument('--femba_num_blocks', type=int, default=2,
                        help='FEMBA number of BiMamba blocks')
    parser.add_argument('--femba_exp', type=int, default=4,
                        help='FEMBA Mamba expansion factor')
    parser.add_argument('--femba_patch_size', type=int, nargs=2, default=[2, 16],
                        help='FEMBA patch size (h, w)')
    parser.add_argument('--femba_stride', type=int, nargs=2, default=[2, 16],
                        help='FEMBA stride (h, w)')
    parser.add_argument('--femba_use_builtin_classifier', action='store_true', default=True,
                        help='Use FEMBA built-in MambaClassifier (recommended)')
    parser.add_argument('--femba_external_classifier', action='store_false',
                        dest='femba_use_builtin_classifier',
                        help='Use external classifier head instead of FEMBA built-in')

    # LUNA-specific settings
    parser.add_argument('--luna_patch_size', type=int, default=40,
                        help='LUNA temporal patch size')
    parser.add_argument('--luna_embed_dim', type=int, default=64,
                        help='LUNA embedding dimension (64=base, 96=large, 128=huge)')
    parser.add_argument('--luna_depth', type=int, default=8,
                        help='LUNA Transformer depth (8=base, 10=large, 12=huge)')
    parser.add_argument('--luna_num_heads', type=int, default=2,
                        help='LUNA number of attention heads')
    parser.add_argument('--luna_num_queries', type=int, default=4,
                        help='LUNA number of learned queries (4=base, 6=large, 8=huge)')
    parser.add_argument('--luna_drop_path', type=float, default=0.1,
                        help='LUNA drop path rate')
    parser.add_argument('--luna_size', type=str, default='base',
                        choices=['base', 'large', 'huge'],
                        help='LUNA model size preset (overrides luna_embed_dim/depth/num_queries)')

    # Data settings
    parser.add_argument('--datasets_dir', type=str, default=None,
                        help='override default dataset directory')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')

    # DIAGNOSIS dataset label filtering options
    parser.add_argument('--include_labels', type=int, nargs='+', default=None,
                        help='For DIAGNOSIS: only include these label indices (0=normal, 1=CVD, 2=AD, 3=depression). '
                             'Example: --include_labels 0 1 3 for 3-class without AD')
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None,
                        help='For DIAGNOSIS: exclude these label indices. '
                             'Example: --exclude_labels 2 to exclude AD')

    # Cross-subject split options
    parser.add_argument('--cross_subject', action='store_true', default=None,
                        help='Enable cross-subject split (no subject appears in multiple splits). '
                             'Default: True for DEPRESSION dataset')
    parser.add_argument('--no_cross_subject', action='store_false', dest='cross_subject',
                        help='Disable cross-subject split (random sample split)')

    # t-SNE visualization settings
    parser.add_argument('--tsne_interval', type=int, default=1,
                        help='Generate t-SNE plots every N epochs (0 to disable)')
    parser.add_argument('--tsne_samples', type=int, default=2000,
                        help='Maximum samples per split for t-SNE visualization')

    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='path to pretrained weights (auto-selected based on --model if not specified)')
    parser.add_argument('--finetuned_ckpt', type=str, default=None,
                        help='path to a finetuned checkpoint (.pth). '
                             'Only the backbone weights will be loaded, classifier head is discarded. '
                             'Takes priority over --pretrained_weights. '
                             'Example: use AD_DIAGNOSIS finetuned model as backbone for TUEV linear probing')
    parser.add_argument('--no_pretrained', action='store_true', default=False,
                        help='train from scratch without pretrained weights')

    # Output
    parser.add_argument('--model_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='directory to save model checkpoints')

    # WandB settings
    parser.add_argument('--no_wandb', action='store_true', default=False,
                        help='disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='eeg-finetuning',
                        help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb run name (auto-generated if not provided)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity (username or team)')

    return parser.parse_args()


def main():
    params = parse_args()

    # Get dataset configuration
    if params.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {params.dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[params.dataset].copy()

    # Override data directory if specified
    if params.datasets_dir:
        dataset_config['data_dir'] = params.datasets_dir

    # Handle DIAGNOSIS label filtering - update config accordingly
    if params.dataset == 'DIAGNOSIS' and (params.include_labels or params.exclude_labels):
        original_label_names = dataset_config['label_names'].copy()

        # Determine which labels we're keeping
        if params.include_labels:
            kept_labels = set(params.include_labels)
        else:
            all_labels = set(original_label_names.keys())
            kept_labels = all_labels - set(params.exclude_labels or [])

        # Update num_classes
        new_num_classes = len(kept_labels)
        dataset_config['num_classes'] = new_num_classes

        # Update task_type based on number of classes
        if new_num_classes == 2:
            dataset_config['task_type'] = 'binary'
        else:
            dataset_config['task_type'] = 'multiclass'

        # Create new label_names with remapped indices
        sorted_labels = sorted(kept_labels)
        new_label_names = {new_idx: original_label_names[orig_idx]
                           for new_idx, orig_idx in enumerate(sorted_labels)}
        dataset_config['label_names'] = new_label_names

        print(f"DIAGNOSIS label filtering:")
        print(f"  Original labels: {original_label_names}")
        print(f"  Kept labels (original indices): {sorted_labels}")
        print(f"  New label names: {new_label_names}")
        print(f"  New num_classes: {new_num_classes}")
        print(f"  Task type: {dataset_config['task_type']}")

    # Set seed and device
    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    device = f'cuda:{params.cuda}'

    print("=" * 60)
    model_name_map = {'codebrain': 'CodeBrain (SSSM)', 'cbramod': 'CBraMod',
                       'femba': 'FEMBA (BiMamba)', 'luna': 'LUNA (Cross-Attn Transformer)'}
    model_name = model_name_map.get(params.model, params.model)
    print(f"EEG Fine-tuning with {model_name} - {params.dataset}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Task type: {dataset_config['task_type']}")
    print(f"Parameters: {params}")

    # Initialize wandb
    use_wandb = not params.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = params.wandb_run_name or f"{params.dataset}_bs{params.batch_size}_lr{params.lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            name=run_name,
            config=vars(params),
            tags=[params.dataset.lower(), params.model, 'eeg'],
        )
        print(f"WandB initialized: {wandb.run.url}")
    else:
        print("WandB logging disabled")

    # Load data
    print("\nLoading data...")
    data_loader, num_classes, seq_len = load_data(params, dataset_config)

    # Create model
    print(f"\nCreating model (backbone: {params.model})...")

    # Determine pretrained weights path
    if params.no_pretrained:
        pretrained_path = None
    elif params.pretrained_weights:
        pretrained_path = params.pretrained_weights
    else:
        # Auto-select default weights based on model type
        default_weights = {
            'codebrain': DEFAULT_CODEBRAIN_WEIGHTS_PATH,
            'cbramod': DEFAULT_WEIGHTS_PATH,
            'luna': DEFAULT_LUNA_WEIGHTS_PATH,
            'femba': DEFAULT_FEMBA_WEIGHTS_PATH,
        }
        pretrained_path = default_weights.get(params.model, DEFAULT_WEIGHTS_PATH)

    finetuned_ckpt = getattr(params, 'finetuned_ckpt', None)
    if finetuned_ckpt:
        # When using finetuned checkpoint, skip original pretrained weights
        pretrained_path = None
    n_channels = dataset_config.get('n_channels', 16)

    # Apply LUNA size presets
    if params.model == 'luna':
        luna_presets = {
            'base':  {'luna_embed_dim': 64,  'luna_depth': 8,  'luna_num_queries': 4, 'luna_num_heads': 2},
            'large': {'luna_embed_dim': 96,  'luna_depth': 10, 'luna_num_queries': 6, 'luna_num_heads': 2},
            'huge':  {'luna_embed_dim': 128, 'luna_depth': 12, 'luna_num_queries': 8, 'luna_num_heads': 2},
        }
        preset = luna_presets[params.luna_size]
        for k, v in preset.items():
            setattr(params, k, v)
        # Auto-select LUNA weights based on size
        if pretrained_path == DEFAULT_LUNA_WEIGHTS_PATH:
            luna_weights_map = {
                'base': '/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors',
                'large': '/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_large.safetensors',
                'huge': '/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_huge.safetensors',
            }
            pretrained_path = luna_weights_map.get(params.luna_size, pretrained_path)

    if params.model == 'codebrain':
        model = CodeBrainModel(
            num_classes=num_classes,
            task_type=dataset_config['task_type'],
            pretrained_weights_path=pretrained_path,
            finetuned_ckpt_path=finetuned_ckpt,
            classifier_type=params.classifier,
            dropout=params.dropout,
            device=device,
            n_channels=n_channels,
            seq_len=seq_len,
            patch_size=200,
            n_layer=params.n_layer,
            codebook_size_t=params.codebook_size_t,
            codebook_size_f=params.codebook_size_f,
        )
    elif params.model == 'femba':
        model = FEMBAModel(
            num_classes=num_classes,
            task_type=dataset_config['task_type'],
            pretrained_weights_path=pretrained_path,
            finetuned_ckpt_path=finetuned_ckpt,
            classifier_type=params.classifier,
            dropout=params.dropout,
            device=device,
            n_channels=n_channels,
            seq_len=seq_len,
            patch_size=200,
            femba_embed_dim=params.femba_embed_dim,
            femba_num_blocks=params.femba_num_blocks,
            femba_exp=params.femba_exp,
            femba_patch_size=tuple(params.femba_patch_size),
            femba_stride=tuple(params.femba_stride),
            use_builtin_classifier=params.femba_use_builtin_classifier,
        )
    elif params.model == 'luna':
        model = LUNAModel(
            num_classes=num_classes,
            task_type=dataset_config['task_type'],
            pretrained_weights_path=pretrained_path,
            finetuned_ckpt_path=finetuned_ckpt,
            classifier_type=params.classifier,
            dropout=params.dropout,
            device=device,
            n_channels=n_channels,
            seq_len=seq_len,
            patch_size=200,
            luna_patch_size=params.luna_patch_size,
            luna_embed_dim=params.luna_embed_dim,
            luna_depth=params.luna_depth,
            luna_num_heads=params.luna_num_heads,
            luna_num_queries=params.luna_num_queries,
            luna_drop_path=params.luna_drop_path,
        )
    else:
        model = EEGModel(
            num_classes=num_classes,
            task_type=dataset_config['task_type'],
            pretrained_weights_path=pretrained_path,
            finetuned_ckpt_path=finetuned_ckpt,
            classifier_type=params.classifier,
            dropout=params.dropout,
            device=device,
            n_channels=n_channels,
            seq_len=seq_len,
            patch_size=200,
        )
    print(f"Model created with {n_channels} channels, seq_len={seq_len}")

    # Log model architecture
    if use_wandb:
        wandb.watch(model, log='gradients', log_freq=100)

    # Train
    print("\nStarting training...")
    trainer = Trainer(params, data_loader, model, dataset_config, use_wandb=use_wandb)
    test_acc, test_kappa, test_f1 = trainer.train()

    # Finish wandb
    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Test Results: Balanced Acc={test_acc:.5f}, Kappa={test_kappa:.5f}, F1={test_f1:.5f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
