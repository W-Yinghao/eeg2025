#!/usr/bin/env python3
"""
Visualize gate activations for individual abnormal samples (correct vs incorrect).

Generates a report with:
  - Raw EEG signal heatmap (channels × time)
  - Temporal gate values per time window
  - Frequency gate values per channel
  - Prediction probability

Usage:
    # On Jean Zay (where checkpoints live):
    python experiments/deb/scripts/visualize_gate_cases.py \
        --checkpoint /path/to/best_*.pth \
        --dataset TUAB --model codebrain \
        --n_correct 10 --n_incorrect 10 \
        --output_dir gate_visualizations/exp7a_best

    # Exp7A best:
    python experiments/deb/scripts/visualize_gate_cases.py \
        --checkpoint checkpoints_selector/exp7a_sparse_l1e3_selector/best_TUAB_codebrain_selector_frozen_acc0.8084_s2025.pth \
        --dataset TUAB --model codebrain --seed 2025 \
        --output_dir gate_visualizations/exp7a_sparse_l1e3_s2025

    # Exp7B best:
    python experiments/deb/scripts/visualize_gate_cases.py \
        --checkpoint checkpoints_selector/exp7b_cons_l3e3_selector/best_TUAB_codebrain_selector_frozen_acc0.8092_s2025.pth \
        --dataset TUAB --model codebrain --seed 2025 \
        --output_dir gate_visualizations/exp7b_cons_l3e3_s2025
"""

import argparse
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from finetune_tuev_lmdb import DATASET_CONFIGS, setup_seed
from backbone_factory import create_backbone
from experiments.deb.data.batch_protocol import load_deb_data
from experiments.deb.models.backbone_wrapper import BackboneWrapper
from experiments.deb.models.selector_model import SelectorModel


# TUAB bipolar channel names (16 channels)
TUAB_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
]


def load_model(checkpoint_path, dataset_name, model_type, device):
    """Load trained SelectorModel from checkpoint."""
    ds_cfg = DATASET_CONFIGS[dataset_name]
    n_channels = ds_cfg.get('n_channels', 16)
    seq_len = ds_cfg.get('seq_len', 10)

    backbone, out_dim, token_dim = create_backbone(
        model_type=model_type,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=200,
        device=device,
    )
    backbone_wrapper = BackboneWrapper(backbone, model_type, n_channels, seq_len, token_dim)

    model = SelectorModel(
        backbone_wrapper=backbone_wrapper,
        mode='selector',
        num_classes=ds_cfg.get('num_classes', 2),
        cfg={
            'regime': 'frozen',
            'enable_temporal_gate': True,
            'enable_frequency_gate': True,
            'fusion': 'concat',
            'gate_hidden': 64,
            'head_dropout': 0.1,
        },
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('best_model_state', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def collect_cases(model, dataloader, device, n_correct=10, n_incorrect=10,
                  target_class=1):
    """
    Collect abnormal correct and abnormal incorrect cases with gate values.

    target_class=1 means "abnormal" in TUAB.
    """
    correct_cases = []
    incorrect_cases = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            output = model(x, return_gates=True)
            logits = output['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            gate_t = output['temporal_gate']   # (B, S, 1)
            gate_f = output['frequency_gate']  # (B, C, 1)

            for i in range(x.size(0)):
                if y[i].item() != target_class:
                    continue

                case = {
                    'signal': x[i].cpu().numpy(),         # (C, S, P)
                    'label': y[i].item(),
                    'pred': preds[i].item(),
                    'prob_abnormal': probs[i, target_class].item(),
                    'prob_normal': probs[i, 0].item(),
                    'gate_t': gate_t[i].squeeze(-1).cpu().numpy(),  # (S,)
                    'gate_f': gate_f[i].squeeze(-1).cpu().numpy(),  # (C,)
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                }

                if preds[i].item() == target_class and len(correct_cases) < n_correct:
                    correct_cases.append(case)
                elif preds[i].item() != target_class and len(incorrect_cases) < n_incorrect:
                    incorrect_cases.append(case)

                if len(correct_cases) >= n_correct and len(incorrect_cases) >= n_incorrect:
                    return correct_cases, incorrect_cases

    return correct_cases, incorrect_cases


def plot_case(case, output_path, case_type='correct', case_idx=0, channels=None):
    """
    Plot a single case: signal heatmap + temporal gate + frequency gate + prediction.
    """
    if channels is None:
        channels = TUAB_CHANNELS

    signal = case['signal']       # (C, S, P) = (16, 10, 200)
    gate_t = case['gate_t']       # (S,) = (10,)
    gate_f = case['gate_f']       # (C,) = (16,)
    prob_abn = case['prob_abnormal']
    pred = case['pred']
    label = case['label']

    C, S, P = signal.shape
    n_channels = len(channels) if len(channels) <= C else C

    # Flatten signal to (C, S*P) for heatmap
    signal_flat = signal[:n_channels].reshape(n_channels, S * P)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 2, height_ratios=[3, 1, 1, 0.3],
                           width_ratios=[4, 1], hspace=0.35, wspace=0.15)

    # --- Panel 1: EEG signal heatmap ---
    ax_signal = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(np.abs(signal_flat), 95)
    im = ax_signal.imshow(signal_flat, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax,
                          extent=[0, S, n_channels - 0.5, -0.5])
    ax_signal.set_yticks(range(n_channels))
    ax_signal.set_yticklabels(channels[:n_channels], fontsize=7)
    ax_signal.set_xlabel('Time window (each = 1 second)')
    ax_signal.set_title(f'Raw EEG Signal  |  Label: {"ABNORMAL" if label==1 else "NORMAL"}'
                        f'  |  Pred: {"ABNORMAL" if pred==1 else "NORMAL"}'
                        f'  |  P(abnormal)={prob_abn:.3f}',
                        fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_signal, fraction=0.02, pad=0.02)

    # Add temporal gate overlay (vertical colored bands)
    for s in range(S):
        alpha = float(gate_t[s]) * 0.4  # scale for visibility
        ax_signal.axvspan(s, s + 1, alpha=alpha, color='lime', zorder=0)

    # --- Panel 2: Frequency gate bar chart (right of signal) ---
    ax_freq = fig.add_subplot(gs[0, 1])
    colors_f = plt.cm.YlOrRd(gate_f[:n_channels] / max(gate_f[:n_channels].max(), 0.01))
    ax_freq.barh(range(n_channels), gate_f[:n_channels], color=colors_f, height=0.8)
    ax_freq.set_yticks(range(n_channels))
    ax_freq.set_yticklabels([])
    ax_freq.set_xlim(0, max(0.5, gate_f.max() * 1.2))
    ax_freq.set_xlabel('Gate value')
    ax_freq.set_title('Freq Gate', fontsize=10)
    ax_freq.invert_yaxis()
    # Add value labels
    for j in range(n_channels):
        ax_freq.text(gate_f[j] + 0.005, j, f'{gate_f[j]:.3f}', va='center', fontsize=6)

    # --- Panel 3: Temporal gate line plot ---
    ax_temp = fig.add_subplot(gs[1, :])
    colors_t = plt.cm.YlOrRd(gate_t / max(gate_t.max(), 0.01))
    ax_temp.bar(range(S), gate_t, color=colors_t, width=0.8, edgecolor='gray', linewidth=0.5)
    ax_temp.set_xlim(-0.5, S - 0.5)
    ax_temp.set_ylim(0, max(0.5, gate_t.max() * 1.2))
    ax_temp.set_xlabel('Time window (1 second each)')
    ax_temp.set_ylabel('Gate value')
    ax_temp.set_title(f'Temporal Gate  |  mean={gate_t.mean():.4f}  max={gate_t.max():.4f}  '
                      f'min={gate_t.min():.4f}', fontsize=10)
    for s in range(S):
        ax_temp.text(s, gate_t[s] + 0.005, f'{gate_t[s]:.3f}', ha='center', fontsize=7)

    # --- Panel 4: Per-channel signal traces (top 4 gated channels) ---
    ax_traces = fig.add_subplot(gs[2, :])
    top_ch_idx = np.argsort(gate_f[:n_channels])[-4:][::-1]  # top 4 by gate value
    time_axis = np.arange(S * P) / 200.0  # seconds
    for rank, ch in enumerate(top_ch_idx):
        trace = signal_flat[ch]
        offset = rank * vmax * 2.5
        ax_traces.plot(time_axis, trace + offset, linewidth=0.5,
                       label=f'{channels[ch]} (g={gate_f[ch]:.3f})')
    ax_traces.set_xlabel('Time (seconds)')
    ax_traces.set_ylabel('Amplitude (offset)')
    ax_traces.set_title('Top-4 gated channels (raw traces)', fontsize=10)
    ax_traces.legend(fontsize=7, loc='upper right')

    # --- Panel 5: Summary text ---
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')
    summary = (f'{case_type.upper()} case #{case_idx}  |  '
               f'Freq gate: mean={gate_f.mean():.4f} std={gate_f.std():.4f}  |  '
               f'Temp gate: mean={gate_t.mean():.4f} std={gate_t.std():.4f}  |  '
               f'Coverage(>0.5): freq={np.mean(gate_f>0.5):.2%} temp={np.mean(gate_t>0.5):.2%}')
    ax_text.text(0.5, 0.5, summary, ha='center', va='center', fontsize=9,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_summary(correct_cases, incorrect_cases, output_dir):
    """
    Plot aggregate comparison: correct vs incorrect abnormal cases.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect gate stats
    correct_gt = np.array([c['gate_t'] for c in correct_cases])   # (N, S)
    correct_gf = np.array([c['gate_f'] for c in correct_cases])   # (N, C)
    incorrect_gt = np.array([c['gate_t'] for c in incorrect_cases]) if incorrect_cases else np.zeros((0, correct_gt.shape[1]))
    incorrect_gf = np.array([c['gate_f'] for c in incorrect_cases]) if incorrect_cases else np.zeros((0, correct_gf.shape[1]))

    S = correct_gt.shape[1]
    C = correct_gf.shape[1]

    # --- Temporal gate comparison ---
    ax = axes[0, 0]
    if len(correct_gt) > 0:
        ax.errorbar(range(S), correct_gt.mean(0), yerr=correct_gt.std(0),
                     marker='o', label=f'Correct (n={len(correct_cases)})', capsize=3)
    if len(incorrect_gt) > 0:
        ax.errorbar(range(S), incorrect_gt.mean(0), yerr=incorrect_gt.std(0),
                     marker='s', label=f'Incorrect (n={len(incorrect_cases)})', capsize=3)
    ax.set_xlabel('Time window')
    ax.set_ylabel('Temporal gate value')
    ax.set_title('Temporal Gate: Correct vs Incorrect (Abnormal)')
    ax.legend()

    # --- Frequency gate comparison ---
    ax = axes[0, 1]
    x_pos = np.arange(C)
    width = 0.35
    if len(correct_gf) > 0:
        ax.bar(x_pos - width/2, correct_gf.mean(0), width,
               yerr=correct_gf.std(0), label='Correct', capsize=2, alpha=0.8)
    if len(incorrect_gf) > 0:
        ax.bar(x_pos + width/2, incorrect_gf.mean(0), width,
               yerr=incorrect_gf.std(0), label='Incorrect', capsize=2, alpha=0.8)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Frequency gate value')
    ax.set_title('Frequency Gate: Correct vs Incorrect (Abnormal)')
    ax.set_xticks(x_pos)
    ch_labels = TUAB_CHANNELS[:C] if C <= len(TUAB_CHANNELS) else [str(i) for i in range(C)]
    ax.set_xticklabels(ch_labels, rotation=45, fontsize=6)
    ax.legend()

    # --- Gate mean distributions ---
    ax = axes[1, 0]
    correct_gf_means = [c['gate_f'].mean() for c in correct_cases]
    incorrect_gf_means = [c['gate_f'].mean() for c in incorrect_cases]
    correct_gt_means = [c['gate_t'].mean() for c in correct_cases]
    incorrect_gt_means = [c['gate_t'].mean() for c in incorrect_cases]

    data_to_plot = []
    labels_to_plot = []
    if correct_gf_means:
        data_to_plot.extend([correct_gf_means, correct_gt_means])
        labels_to_plot.extend(['Correct\nFreq', 'Correct\nTemp'])
    if incorrect_gf_means:
        data_to_plot.extend([incorrect_gf_means, incorrect_gt_means])
        labels_to_plot.extend(['Incorrect\nFreq', 'Incorrect\nTemp'])

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        colors_bp = ['#4CAF50', '#8BC34A', '#F44336', '#FF9800']
        for patch, color in zip(bp['boxes'], colors_bp[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel('Mean gate value')
    ax.set_title('Gate Mean Distribution')

    # --- Prediction probability distribution ---
    ax = axes[1, 1]
    correct_probs = [c['prob_abnormal'] for c in correct_cases]
    incorrect_probs = [c['prob_abnormal'] for c in incorrect_cases]
    if correct_probs:
        ax.hist(correct_probs, bins=20, alpha=0.6, label='Correct', color='green', density=True)
    if incorrect_probs:
        ax.hist(incorrect_probs, bins=20, alpha=0.6, label='Incorrect', color='red', density=True)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('P(abnormal)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()

    plt.suptitle('Abnormal Cases: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Save numerical summary ---
    summary = {
        'n_correct': len(correct_cases),
        'n_incorrect': len(incorrect_cases),
        'correct_gate_t_mean': float(correct_gt.mean()) if len(correct_gt) else None,
        'correct_gate_f_mean': float(correct_gf.mean()) if len(correct_gf) else None,
        'incorrect_gate_t_mean': float(incorrect_gt.mean()) if len(incorrect_gt) else None,
        'incorrect_gate_f_mean': float(incorrect_gf.mean()) if len(incorrect_gf) else None,
        'correct_prob_abnormal_mean': float(np.mean(correct_probs)) if correct_probs else None,
        'incorrect_prob_abnormal_mean': float(np.mean(incorrect_probs)) if incorrect_probs else None,
        'correct_gate_f_per_channel': correct_gf.mean(0).tolist() if len(correct_gf) else None,
        'incorrect_gate_f_per_channel': incorrect_gf.mean(0).tolist() if len(incorrect_gf) else None,
        'correct_gate_t_per_window': correct_gt.mean(0).tolist() if len(correct_gt) else None,
        'incorrect_gate_t_per_window': incorrect_gt.mean(0).tolist() if len(incorrect_gt) else None,
    }
    with open(os.path.join(output_dir, 'comparison_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Visualize gate activations on individual cases')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='TUAB')
    parser.add_argument('--model', type=str, default='codebrain')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_correct', type=int, default=10)
    parser.add_argument('--n_incorrect', type=int, default=10)
    parser.add_argument('--target_class', type=int, default=1,
                        help='Class to focus on (1=abnormal for TUAB)')
    parser.add_argument('--output_dir', type=str, default='gate_visualizations')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.dataset, args.model, device)

    print(f"Loading {args.dataset} test data...")
    data_loaders = load_deb_data(
        dataset_name=args.dataset,
        model_type=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = data_loaders['test']

    print(f"Collecting {args.n_correct} correct + {args.n_incorrect} incorrect "
          f"abnormal cases...")
    correct_cases, incorrect_cases = collect_cases(
        model, test_loader, device,
        n_correct=args.n_correct,
        n_incorrect=args.n_incorrect,
        target_class=args.target_class,
    )
    print(f"  Found {len(correct_cases)} correct, {len(incorrect_cases)} incorrect")

    # Plot individual cases
    channels = TUAB_CHANNELS if args.dataset == 'TUAB' else None

    for i, case in enumerate(correct_cases):
        path = os.path.join(args.output_dir, f'correct_{i:02d}.png')
        plot_case(case, path, case_type='correct', case_idx=i, channels=channels)
        print(f"  Saved {path}")

    for i, case in enumerate(incorrect_cases):
        path = os.path.join(args.output_dir, f'incorrect_{i:02d}.png')
        plot_case(case, path, case_type='incorrect', case_idx=i, channels=channels)
        print(f"  Saved {path}")

    # Plot comparison summary
    summary = plot_comparison_summary(correct_cases, incorrect_cases, args.output_dir)
    print(f"\n=== Summary ===")
    print(f"Correct abnormal ({len(correct_cases)} cases):")
    print(f"  gate_t_mean={summary['correct_gate_t_mean']:.4f}  "
          f"gate_f_mean={summary['correct_gate_f_mean']:.4f}  "
          f"P(abn)={summary['correct_prob_abnormal_mean']:.4f}")
    if summary['incorrect_gate_t_mean'] is not None:
        print(f"Incorrect abnormal ({len(incorrect_cases)} cases):")
        print(f"  gate_t_mean={summary['incorrect_gate_t_mean']:.4f}  "
              f"gate_f_mean={summary['incorrect_gate_f_mean']:.4f}  "
              f"P(abn)={summary['incorrect_prob_abnormal_mean']:.4f}")
    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
