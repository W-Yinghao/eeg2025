"""
Unified batch protocol wrapper for DEB experiments.

Wraps the existing EEGDatasetWithSubjects to produce batches with a
standardized dict interface:

    batch = {
        "x":                   (B, C, S, P)   float32 EEG patches
        "y":                   (B,)           int64   class labels
        "subject_id":          (B,)           int64   subject IDs
        "site_id":             None           (not available in current data)
        "montage_id":          None           (not available)
        "reference_type":      None           (not available)
        "channel_coordinates": None           (not available)
        "channel_mask":        (B, C)         float32 ones (all channels valid)
    }

Fields marked None are placeholders for future metadata.  When a downstream
module receives None it should skip any gating / conditioning on that field.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DEBDataset(Dataset):
    """
    Thin wrapper that converts (signal, label, subject_id) tuples from
    EEGDatasetWithSubjects into the unified batch protocol dict.
    """

    def __init__(self, base_with_subjects):
        """
        Args:
            base_with_subjects: an EEGDatasetWithSubjects instance (from
                train_ib_disentangle.py) which yields (signal, label, subject_id).
        """
        self.ds = base_with_subjects
        self.n_channels = self.ds.base_dataset.target_n_channels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        signal, label, subject_id = self.ds[idx]
        return signal, label, subject_id, self.n_channels

    @staticmethod
    def collate_fn(batch):
        """Collate into the unified batch dict."""
        signals = np.array([b[0] for b in batch])
        labels = np.array([b[1] for b in batch])
        sids = np.array([b[2] for b in batch])
        n_ch = batch[0][3]

        B = len(batch)
        return {
            'x': torch.from_numpy(signals).float(),          # (B, C, S, P)
            'y': torch.from_numpy(labels).long(),             # (B,)
            'subject_id': torch.from_numpy(sids).long(),      # (B,)
            'site_id': None,
            'montage_id': None,
            'reference_type': None,
            'channel_coordinates': None,
            'channel_mask': torch.ones(B, n_ch, dtype=torch.float32),  # (B, C)
        }


def load_deb_data(cfg, dataset_config):
    """
    Load data through the existing infrastructure, wrap with DEBDataset.

    Uses load_data_with_subjects from train_ib_disentangle.py underneath.

    Args:
        cfg: DEB config dict (from configs/defaults.py)
        dataset_config: entry from DATASET_CONFIGS

    Returns:
        data_loaders: dict with 'train', 'val', 'test' DataLoaders
        num_classes: int
        seq_len: int
        num_subjects: int
    """
    import sys
    import os
    # Ensure repo root is importable
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from finetune_tuev_lmdb import EEGLMDBDataset
    from train_ib_disentangle import EEGDatasetWithSubjects

    from ..data.splits import SubjectSplitter

    splitter = SubjectSplitter(cfg, dataset_config)
    base_datasets = splitter.get_base_datasets()

    loaders = {}
    num_subjects = 0
    for split_name, base_ds in base_datasets.items():
        wrapped = EEGDatasetWithSubjects(base_ds)
        if split_name == 'train':
            num_subjects = wrapped.num_subjects
        deb_ds = DEBDataset(wrapped)
        loaders[split_name] = DataLoader(
            deb_ds,
            batch_size=cfg['batch_size'],
            shuffle=(split_name == 'train'),
            drop_last=(split_name == 'train'),
            num_workers=cfg.get('num_workers', 0),
            collate_fn=DEBDataset.collate_fn,
        )

    num_classes = dataset_config['num_classes']
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']
    seg_dur = dataset_config['segment_duration']
    seq_len = int(seg_dur * dataset_config['sampling_rate'] / patch_size)

    return loaders, num_classes, seq_len, num_subjects
