"""
Split logic wrapper for DEB experiments.

Reuses the existing EEGLMDBDataset split infrastructure.  Provides:
  1. Subject-level split (default, via cross_subject=True)
  2. Random split (fallback)
  3. Site-held-out (placeholder — current LMDB metadata does not include
     site information for most datasets; see README)

All split strategies are config-selectable.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from finetune_tuev_lmdb import EEGLMDBDataset


class SubjectSplitter:
    """
    Creates train/val/test EEGLMDBDataset instances respecting the
    configured split strategy.

    Strategies:
      'subject'       — cross_subject=True in EEGLMDBDataset (default)
      'random'        — cross_subject=False (sample-level shuffle)
      'site_held_out' — NOT YET IMPLEMENTED (falls back to subject split
                        with a warning)
    """

    def __init__(self, cfg: dict, dataset_config: dict):
        self.cfg = cfg
        self.dc = dataset_config
        self.strategy = cfg.get('split_strategy', 'subject')

    def get_base_datasets(self) -> dict:
        """Return {'train': ds, 'val': ds, 'test': ds} of EEGLMDBDataset."""
        if self.strategy == 'site_held_out':
            print("[WARN] site_held_out split not implemented — "
                  "falling back to subject-level split. "
                  "Current LMDB metadata does not contain site_id.")
            cross_subject = True
        elif self.strategy == 'subject':
            cross_subject = True
        else:
            cross_subject = False

        data_dir = Path(self.dc['data_dir'])
        splits = self.dc['splits']
        single_lmdb = self.dc.get('single_lmdb', False)
        train_ratio = self.dc.get('train_ratio', 0.7)
        val_ratio = self.cfg.get('val_ratio', self.dc.get('val_ratio', 0.15))
        test_ratio = self.dc.get('test_ratio', 0.15)

        include_labels = self.cfg.get('include_labels')
        exclude_labels = self.cfg.get('exclude_labels')

        datasets = {}
        for split_name in ('train', 'val', 'test'):
            if single_lmdb:
                lmdb_path = data_dir / splits['train']
            else:
                lmdb_path = data_dir / splits[split_name]

            is_val_from_train = (not single_lmdb
                                 and splits.get('val') == splits.get('train')
                                 and split_name == 'val')

            datasets[split_name] = EEGLMDBDataset(
                lmdb_path=lmdb_path,
                dataset_config=self.dc,
                split=split_name,
                val_ratio=val_ratio,
                is_val_from_train=is_val_from_train,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                include_labels=include_labels,
                exclude_labels=exclude_labels,
                cross_subject=cross_subject if single_lmdb else False,
            )

        return datasets
