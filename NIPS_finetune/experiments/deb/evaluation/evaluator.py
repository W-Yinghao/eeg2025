"""
Evaluation utilities for DEB experiments.

Primary metric: balanced accuracy (for early stopping).
Also reports: macro-F1, confusion matrix, per-class F1.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from typing import Dict, Optional, Tuple


class Evaluator:
    """Stateless evaluator — call evaluate() with model + dataloader."""

    @staticmethod
    @torch.no_grad()
    def evaluate(model, dataloader, device,
                 label_names: Optional[dict] = None) -> Dict:
        """
        Evaluate model on a dataloader.

        Args:
            model: DEBModel
            dataloader: DataLoader yielding batch dicts (DEBDataset.collate_fn)
            device: torch device
            label_names: optional {int: str} for display

        Returns:
            dict with:
              'bal_acc':       balanced accuracy (primary metric)
              'f1_macro':      macro-averaged F1
              'f1_weighted':   weighted F1
              'f1_per_class':  array of per-class F1
              'ce_loss':       cross-entropy loss (average)
              'confusion':     confusion matrix
              'preds':         all predictions
              'labels':        all labels
        """
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            out = model(x)
            logits = out['logits']

            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        f1_mac = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_wt = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_pc = f1_score(all_labels, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        result = {
            'bal_acc': bal_acc,
            'f1_macro': f1_mac,
            'f1_weighted': f1_wt,
            'f1_per_class': f1_pc,
            'ce_loss': total_loss / max(n_batches, 1),
            'confusion': cm,
            'preds': all_preds,
            'labels': all_labels,
        }

        return result

    @staticmethod
    def print_results(results: Dict, split_name: str = 'test',
                      label_names: Optional[dict] = None):
        """Pretty-print evaluation results."""
        print(f"\n  {split_name.upper()} Results:")
        print(f"    Balanced Accuracy: {results['bal_acc']:.4f}")
        print(f"    Macro F1:          {results['f1_macro']:.4f}")
        print(f"    Weighted F1:       {results['f1_weighted']:.4f}")
        print(f"    CE Loss:           {results['ce_loss']:.4f}")

        if label_names:
            print(f"    Per-class F1:")
            for i, f1 in enumerate(results['f1_per_class']):
                name = label_names.get(i, str(i))
                print(f"      {name:20s}: {f1:.4f}")

        print(f"    Confusion Matrix:\n    {results['confusion']}")
