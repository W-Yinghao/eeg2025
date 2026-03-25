"""
Explainability Evaluation Pipeline (Experiment 8).

Three evaluation types:
  1. Insertion/Deletion: measure how gate scores predict important regions
  2. Augmentation Consistency: measure gate map stability under perturbation
  3. Abnormal-Focused Analysis: detailed metrics for abnormal/disease class

All evaluations operate on a trained selector model + test dataloader.
Results are saved as JSON + optional plots.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix,
)

from ..training.augmentations import EEGAugmentor


class InsertionDeletionEvaluator:
    """
    Insertion/Deletion evaluation for gate-based evidence selection.

    Insertion: start from masked input, progressively restore high-gate regions
    Deletion: start from full input, progressively mask high-gate regions

    Measures: AUC of prediction confidence curve.
    """

    def __init__(self, n_steps: int = 10, target_class: Optional[int] = None):
        """
        Args:
            n_steps: number of insertion/deletion steps
            target_class: if set, track probability of this class only
        """
        self.n_steps = n_steps
        self.target_class = target_class

    @torch.no_grad()
    def evaluate(self, model, dataloader, device) -> Dict:
        """Run insertion/deletion evaluation."""
        model.eval()

        all_insertion_curves = []
        all_deletion_curves = []
        all_labels = []
        all_preds = []

        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            # Get gate scores
            out = model(x, return_gates=True)
            logits = out['logits']
            preds = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Use temporal gate as primary importance score
            gate_t = out.get('temporal_gate')  # (B, S, 1)
            if gate_t is None:
                continue

            gate_scores = gate_t.squeeze(-1)  # (B, S)
            B, S = gate_scores.shape

            for i in range(B):
                label = y[i].item()
                pred = preds[i].item()
                target = label if self.target_class is None else self.target_class

                score_i = gate_scores[i]  # (S,)
                sorted_idx = torch.argsort(score_i, descending=True)

                ins_curve = self._insertion_curve(
                    model, x[i:i+1], sorted_idx, target, device
                )
                del_curve = self._deletion_curve(
                    model, x[i:i+1], sorted_idx, target, device
                )

                all_insertion_curves.append(ins_curve)
                all_deletion_curves.append(del_curve)

        if not all_insertion_curves:
            return {'error': 'no gate scores available'}

        ins_curves = np.array(all_insertion_curves)
        del_curves = np.array(all_deletion_curves)
        labels = np.array(all_labels)

        result = {
            'insertion_auc': float(np.trapz(ins_curves.mean(0), dx=1.0/self.n_steps)),
            'deletion_auc': float(np.trapz(del_curves.mean(0), dx=1.0/self.n_steps)),
            'insertion_curve_mean': ins_curves.mean(0).tolist(),
            'insertion_curve_std': ins_curves.std(0).tolist(),
            'deletion_curve_mean': del_curves.mean(0).tolist(),
            'deletion_curve_std': del_curves.std(0).tolist(),
            'n_samples': len(all_insertion_curves),
        }

        # Per-class breakdown
        unique_labels = sorted(set(all_labels))
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            result[f'class_{lbl}_insertion_auc'] = float(
                np.trapz(ins_curves[mask].mean(0), dx=1.0/self.n_steps)
            )
            result[f'class_{lbl}_deletion_auc'] = float(
                np.trapz(del_curves[mask].mean(0), dx=1.0/self.n_steps)
            )
            result[f'class_{lbl}_n_samples'] = int(mask.sum())

        return result

    def _insertion_curve(self, model, x, sorted_idx, target_class, device):
        """Progressively insert high-gate regions into masked input."""
        B, C, S, P = x.shape
        curve = []
        step_size = max(1, S // self.n_steps)

        for step in range(self.n_steps + 1):
            n_reveal = min(step * step_size, S)
            mask = torch.zeros(S, dtype=torch.bool, device=device)
            if n_reveal > 0:
                mask[sorted_idx[:n_reveal]] = True

            x_masked = x.clone()
            x_masked[:, :, ~mask, :] = 0.0

            out = model(x_masked)
            prob = F.softmax(out['logits'], dim=-1)[0, target_class].item()
            curve.append(prob)

        return curve

    def _deletion_curve(self, model, x, sorted_idx, target_class, device):
        """Progressively delete high-gate regions from full input."""
        B, C, S, P = x.shape
        curve = []
        step_size = max(1, S // self.n_steps)

        for step in range(self.n_steps + 1):
            n_delete = min(step * step_size, S)
            mask = torch.ones(S, dtype=torch.bool, device=device)
            if n_delete > 0:
                mask[sorted_idx[:n_delete]] = False

            x_masked = x.clone()
            x_masked[:, :, ~mask, :] = 0.0

            out = model(x_masked)
            prob = F.softmax(out['logits'], dim=-1)[0, target_class].item()
            curve.append(prob)

        return curve


class AugmentationConsistencyEvaluator:
    """
    Evaluate gate map consistency under light augmentations.

    For each sample, apply augmentation and compare gate maps.
    Reports: mean consistency score, per-sample distribution.
    """

    def __init__(self, n_augmentations: int = 5,
                 jitter_std: float = 0.05,
                 mask_ratio: float = 0.1):
        self.n_augmentations = n_augmentations
        self.augmentor = EEGAugmentor(
            enable_time_shift=True,
            enable_amplitude_jitter=True,
            enable_time_mask=True,
            jitter_std=jitter_std,
            mask_ratio=mask_ratio,
            p_each=0.8,
        )

    @torch.no_grad()
    def evaluate(self, model, dataloader, device) -> Dict:
        """Run augmentation consistency evaluation."""
        model.eval()

        all_temporal_consistency = []
        all_frequency_consistency = []
        all_labels = []

        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            # Original gate maps
            out_orig = model(x, return_gates=True)
            gate_t_orig = out_orig.get('temporal_gate')
            gate_f_orig = out_orig.get('frequency_gate')

            if gate_t_orig is None:
                continue

            B = x.shape[0]

            for i in range(B):
                t_consistencies = []
                f_consistencies = []

                for _ in range(self.n_augmentations):
                    x_aug = self.augmentor(x[i:i+1])
                    out_aug = model(x_aug, return_gates=True)

                    gate_t_aug = out_aug.get('temporal_gate')
                    if gate_t_aug is not None and gate_t_orig is not None:
                        min_s = min(gate_t_orig.shape[1], gate_t_aug.shape[1])
                        g_o = gate_t_orig[i, :min_s].squeeze(-1)
                        g_a = gate_t_aug[0, :min_s].squeeze(-1)
                        cos = F.cosine_similarity(
                            g_o.unsqueeze(0), g_a.unsqueeze(0)
                        ).item()
                        t_consistencies.append(cos)

                    gate_f_aug = out_aug.get('frequency_gate')
                    if gate_f_aug is not None and gate_f_orig is not None:
                        g_o = gate_f_orig[i].squeeze(-1)
                        g_a = gate_f_aug[0].squeeze(-1)
                        cos = F.cosine_similarity(
                            g_o.unsqueeze(0), g_a.unsqueeze(0)
                        ).item()
                        f_consistencies.append(cos)

                if t_consistencies:
                    all_temporal_consistency.append(np.mean(t_consistencies))
                if f_consistencies:
                    all_frequency_consistency.append(np.mean(f_consistencies))
                all_labels.append(y[i].item())

        labels = np.array(all_labels)
        t_cons = np.array(all_temporal_consistency) if all_temporal_consistency else np.array([])
        f_cons = np.array(all_frequency_consistency) if all_frequency_consistency else np.array([])

        result = {
            'n_samples': len(all_labels),
            'n_augmentations': self.n_augmentations,
        }

        if len(t_cons) > 0:
            result['temporal_consistency_mean'] = float(t_cons.mean())
            result['temporal_consistency_std'] = float(t_cons.std())
            result['temporal_consistency_median'] = float(np.median(t_cons))

        if len(f_cons) > 0:
            result['frequency_consistency_mean'] = float(f_cons.mean())
            result['frequency_consistency_std'] = float(f_cons.std())
            result['frequency_consistency_median'] = float(np.median(f_cons))

        # Per-class breakdown
        unique_labels = sorted(set(all_labels))
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            if len(t_cons) > 0:
                result[f'class_{lbl}_temporal_consistency'] = float(t_cons[mask].mean())
            if len(f_cons) > 0:
                result[f'class_{lbl}_frequency_consistency'] = float(f_cons[mask].mean())

        return result


class AbnormalFocusedAnalyzer:
    """
    Abnormal-focused analysis for disease diagnosis.

    Computes:
      - abnormal F1, recall, precision
      - gate coverage on abnormal samples
      - evidence distribution statistics for abnormal class
    """

    def __init__(self, abnormal_class: int = 0, gate_threshold: float = 0.5):
        """
        Args:
            abnormal_class: index of the abnormal/disease class
            gate_threshold: threshold for "active" gate
        """
        self.abnormal_class = abnormal_class
        self.gate_threshold = gate_threshold

    @torch.no_grad()
    def evaluate(self, model, dataloader, device) -> Dict:
        """Run abnormal-focused analysis."""
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        # Gate statistics per sample
        gate_coverages = []  # fraction of gates > threshold
        gate_means = []
        evidence_lengths = []  # number of segments with high gate
        sample_labels = []

        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            out = model(x, return_gates=True)
            logits = out['logits']
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            gate_t = out.get('temporal_gate')
            if gate_t is not None:
                gate_vals = gate_t.squeeze(-1)  # (B, S)
                for i in range(gate_vals.shape[0]):
                    g = gate_vals[i]
                    coverage = (g > self.gate_threshold).float().mean().item()
                    gate_coverages.append(coverage)
                    gate_means.append(g.mean().item())
                    evidence_lengths.append((g > self.gate_threshold).sum().item())
                    sample_labels.append(y[i].item())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        sample_labels = np.array(sample_labels)

        # Overall metrics
        result = {
            'overall': {
                'bal_acc': float(balanced_accuracy_score(all_labels, all_preds)),
                'f1_macro': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
                'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            },
        }

        # Per-class metrics
        f1_pc = f1_score(all_labels, all_preds, average=None, zero_division=0)
        rec_pc = recall_score(all_labels, all_preds, average=None, zero_division=0)
        prec_pc = precision_score(all_labels, all_preds, average=None, zero_division=0)

        num_classes = len(f1_pc)
        result['per_class'] = {}
        for c in range(num_classes):
            result['per_class'][str(c)] = {
                'f1': float(f1_pc[c]),
                'recall': float(rec_pc[c]),
                'precision': float(prec_pc[c]),
                'n_samples': int((all_labels == c).sum()),
            }

        # Abnormal-focused
        abn = self.abnormal_class
        if abn < num_classes:
            result['abnormal'] = {
                'class_index': abn,
                'f1': float(f1_pc[abn]),
                'recall': float(rec_pc[abn]),
                'precision': float(prec_pc[abn]),
                'n_samples': int((all_labels == abn).sum()),
                'n_correct': int(((all_labels == abn) & (all_preds == abn)).sum()),
            }

            # Abnormal confidence
            abn_mask = all_labels == abn
            if abn_mask.sum() > 0:
                abn_probs = all_probs[abn_mask, abn]
                result['abnormal']['mean_confidence'] = float(abn_probs.mean())
                result['abnormal']['std_confidence'] = float(abn_probs.std())

        # Gate statistics
        if len(gate_coverages) > 0:
            gate_coverages = np.array(gate_coverages)
            gate_means_arr = np.array(gate_means)
            evidence_lengths_arr = np.array(evidence_lengths)

            result['gate_stats'] = {
                'overall_coverage_mean': float(gate_coverages.mean()),
                'overall_coverage_std': float(gate_coverages.std()),
                'overall_gate_mean': float(gate_means_arr.mean()),
                'overall_evidence_length_mean': float(evidence_lengths_arr.mean()),
            }

            # Per-class gate stats
            for c in range(num_classes):
                mask = sample_labels == c
                if mask.sum() == 0:
                    continue
                result['gate_stats'][f'class_{c}'] = {
                    'coverage_mean': float(gate_coverages[mask].mean()),
                    'coverage_std': float(gate_coverages[mask].std()),
                    'gate_mean': float(gate_means_arr[mask].mean()),
                    'evidence_length_mean': float(evidence_lengths_arr[mask].mean()),
                    'evidence_length_std': float(evidence_lengths_arr[mask].std()),
                }

            # Abnormal gate stats
            abn_mask = sample_labels == abn
            if abn_mask.sum() > 0:
                result['abnormal_gate'] = {
                    'coverage_mean': float(gate_coverages[abn_mask].mean()),
                    'coverage_std': float(gate_coverages[abn_mask].std()),
                    'gate_mean': float(gate_means_arr[abn_mask].mean()),
                    'evidence_length_mean': float(evidence_lengths_arr[abn_mask].mean()),
                    'evidence_length_ratio': float(
                        evidence_lengths_arr[abn_mask].mean() /
                        max(evidence_lengths_arr.mean(), 1e-7)
                    ),
                }

        return result


def run_full_explainability_eval(
    model,
    dataloader,
    device,
    output_dir: str,
    label_names: Optional[Dict] = None,
    abnormal_class: int = 0,
    n_insertion_steps: int = 10,
    n_augmentations: int = 5,
    model_name: str = 'model',
) -> Dict:
    """
    Run the complete explainability evaluation suite.

    Args:
        model: trained SelectorModel
        dataloader: test dataloader
        device: torch device
        output_dir: directory for saving results
        label_names: {int: str} class label names
        abnormal_class: index of abnormal class
        n_insertion_steps: steps for insertion/deletion
        n_augmentations: augmentation repeats for consistency
        model_name: name tag for output files

    Returns:
        Combined results dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {'model_name': model_name}

    # 1. Insertion/Deletion
    print(f"\n[Explainability] Running insertion/deletion evaluation...")
    ins_del = InsertionDeletionEvaluator(n_steps=n_insertion_steps)
    ins_del_results = ins_del.evaluate(model, dataloader, device)
    results['insertion_deletion'] = ins_del_results
    print(f"  Insertion AUC: {ins_del_results.get('insertion_auc', 'N/A'):.4f}")
    print(f"  Deletion AUC:  {ins_del_results.get('deletion_auc', 'N/A'):.4f}")

    # 2. Augmentation Consistency
    print(f"\n[Explainability] Running augmentation consistency evaluation...")
    aug_cons = AugmentationConsistencyEvaluator(n_augmentations=n_augmentations)
    aug_cons_results = aug_cons.evaluate(model, dataloader, device)
    results['augmentation_consistency'] = aug_cons_results
    if 'temporal_consistency_mean' in aug_cons_results:
        print(f"  Temporal consistency: "
              f"{aug_cons_results['temporal_consistency_mean']:.4f} "
              f"+/- {aug_cons_results.get('temporal_consistency_std', 0):.4f}")
    if 'frequency_consistency_mean' in aug_cons_results:
        print(f"  Frequency consistency: "
              f"{aug_cons_results['frequency_consistency_mean']:.4f} "
              f"+/- {aug_cons_results.get('frequency_consistency_std', 0):.4f}")

    # 3. Abnormal-Focused Analysis
    print(f"\n[Explainability] Running abnormal-focused analysis...")
    abn_analyzer = AbnormalFocusedAnalyzer(
        abnormal_class=abnormal_class
    )
    abn_results = abn_analyzer.evaluate(model, dataloader, device)
    results['abnormal_analysis'] = abn_results
    if 'abnormal' in abn_results:
        abn = abn_results['abnormal']
        print(f"  Abnormal Recall:    {abn['recall']:.4f}")
        print(f"  Abnormal F1:        {abn['f1']:.4f}")
        print(f"  Abnormal Precision: {abn['precision']:.4f}")
    if 'abnormal_gate' in abn_results:
        ag = abn_results['abnormal_gate']
        print(f"  Abnormal gate coverage: {ag['coverage_mean']:.4f}")
        print(f"  Abnormal evidence ratio: {ag.get('evidence_length_ratio', 'N/A')}")

    # Save results
    result_file = output_dir / f"explainability_{model_name}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved results to {result_file}")

    # Try to generate plots
    try:
        _plot_insertion_deletion(ins_del_results, output_dir, model_name)
    except Exception as e:
        print(f"  Skipping plots (matplotlib issue): {e}")

    return results


def _plot_insertion_deletion(results: Dict, output_dir: Path, model_name: str):
    """Generate insertion/deletion curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Insertion
    ins_mean = results.get('insertion_curve_mean', [])
    ins_std = results.get('insertion_curve_std', [])
    if ins_mean:
        x = np.linspace(0, 1, len(ins_mean))
        ax1.plot(x, ins_mean, 'b-', linewidth=2)
        if ins_std:
            ins_mean = np.array(ins_mean)
            ins_std_arr = np.array(ins_std)
            ax1.fill_between(x, ins_mean - ins_std_arr, ins_mean + ins_std_arr,
                             alpha=0.2, color='blue')
        ax1.set_xlabel('Fraction of segments revealed')
        ax1.set_ylabel('Target class probability')
        ax1.set_title(f'Insertion (AUC={results.get("insertion_auc", 0):.3f})')
        ax1.grid(True, alpha=0.3)

    # Deletion
    del_mean = results.get('deletion_curve_mean', [])
    del_std = results.get('deletion_curve_std', [])
    if del_mean:
        x = np.linspace(0, 1, len(del_mean))
        ax2.plot(x, del_mean, 'r-', linewidth=2)
        if del_std:
            del_mean = np.array(del_mean)
            del_std_arr = np.array(del_std)
            ax2.fill_between(x, del_mean - del_std_arr, del_mean + del_std_arr,
                             alpha=0.2, color='red')
        ax2.set_xlabel('Fraction of segments deleted')
        ax2.set_ylabel('Target class probability')
        ax2.set_title(f'Deletion (AUC={results.get("deletion_auc", 0):.3f})')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Insertion/Deletion — {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'insertion_deletion_{model_name}.png', dpi=150)
    plt.close()
