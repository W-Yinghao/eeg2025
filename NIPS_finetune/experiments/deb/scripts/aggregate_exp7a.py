#!/usr/bin/env python3
"""
Aggregate Exp 7A results across 3 sparse lambda configs.

Reads JSON summaries from checkpoint directories, computes mean/std
over seeds, and prints a comparison table.

Usage:
    python experiments/deb/scripts/aggregate_exp7a.py [--base_dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def find_summaries(ckpt_dir: Path):
    """Find all JSON summary files (not config/classwise/gate)."""
    results = []
    if not ckpt_dir.exists():
        return results
    for f in sorted(ckpt_dir.glob("best_*.json")):
        name = f.name
        if '_config.json' in name or '_classwise.json' in name or '_gate_stats.json' in name:
            continue
        results.append(f)
    return results


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base checkpoint directory (default: auto-detect)')
    args = parser.parse_args()

    if args.base_dir:
        base = Path(args.base_dir)
    else:
        # Auto-detect: try $WORK path, then local
        work = os.environ.get('WORK', '')
        candidates = [
            Path(work) / 'yinghao/eeg2025/NIPS_finetune/checkpoints_selector',
            Path('checkpoints_selector'),
        ]
        base = None
        for c in candidates:
            if c.exists():
                base = c
                break
        if base is None:
            print("No checkpoint directory found. Use --base_dir.")
            sys.exit(1)

    configs = {
        'l1e4': {'dir': 'exp7a_sparse_l1e4_selector', 'lambda': '1e-4'},
        'l3e4': {'dir': 'exp7a_sparse_l3e4_selector', 'lambda': '3e-4'},
        'l1e3': {'dir': 'exp7a_sparse_l1e3_selector', 'lambda': '1e-3'},
    }

    # Collect results
    all_results = {}
    for tag, info in configs.items():
        ckpt_dir = base / info['dir']
        summaries = find_summaries(ckpt_dir)
        runs = []
        for s in summaries:
            data = load_summary(s)
            test = data.get('test', {})
            gate = data.get('gate_stats', {})
            # Also try loading separate gate_stats JSON
            gate_path = s.parent / s.name.replace('.json', '_gate_stats.json')
            if gate_path.exists() and not gate:
                with open(gate_path) as gf:
                    gate = json.load(gf)
            runs.append({
                'seed': data.get('seed'),
                'bal_acc': test.get('bal_acc'),
                'f1_macro': test.get('f1_macro'),
                'abnormal_f1': test.get('abnormal_f1'),
                'abnormal_recall': test.get('abnormal_recall'),
                'gate_coverage': gate.get('gate_coverage_0.5'),
                'gate_entropy': gate.get('gate_entropy'),
                'gate_top10_ratio': gate.get('gate_top10_ratio'),
                'gate_top20_ratio': gate.get('gate_top20_ratio'),
                'abnormal_only_gate_coverage': gate.get('abnormal_only_gate_coverage'),
                'best_epoch': data.get('best_epoch'),
            })
        all_results[tag] = runs

    # Print table
    metrics = [
        ('BalAcc', 'bal_acc'),
        ('Macro-F1', 'f1_macro'),
        ('Abnormal F1', 'abnormal_f1'),
        ('Abnormal Recall', 'abnormal_recall'),
        ('Gate Coverage', 'gate_coverage'),
        ('Gate Entropy', 'gate_entropy'),
        ('Gate Top10 Ratio', 'gate_top10_ratio'),
        ('Gate Top20 Ratio', 'gate_top20_ratio'),
        ('Abn Gate Coverage', 'abnormal_only_gate_coverage'),
        ('Best Epoch', 'best_epoch'),
    ]

    print("=" * 80)
    print("  Exp 7A Aggregation: Frozen Selector + L1 Sparse")
    print("=" * 80)

    # Header
    header = f"{'Metric':<22s}"
    for tag, info in configs.items():
        n = len(all_results[tag])
        header += f" | lambda={info['lambda']:>6s} (n={n})"
    print(header)
    print("-" * len(header))

    for metric_name, metric_key in metrics:
        row = f"{metric_name:<22s}"
        for tag in configs:
            vals = [r[metric_key] for r in all_results[tag] if r[metric_key] is not None]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                row += f" | {mean:>7.4f} +/- {std:.4f}  "
            else:
                row += f" |       N/A            "
        print(row)

    print()

    # Per-seed detail
    print("Per-seed details:")
    print("-" * 80)
    for tag, info in configs.items():
        print(f"\n  lambda={info['lambda']}:")
        for r in all_results[tag]:
            print(f"    seed={r['seed']}: BalAcc={r['bal_acc']:.4f}  "
                  f"F1={r['f1_macro']:.4f}  "
                  f"AbnF1={r.get('abnormal_f1', 'N/A')}  "
                  f"GateCov={r.get('gate_coverage', 'N/A')}  "
                  f"Ep={r.get('best_epoch', '?')}")

    # Save as JSON
    output = {}
    for tag, info in configs.items():
        vals_dict = defaultdict(list)
        for r in all_results[tag]:
            for mk, mv in r.items():
                if mv is not None:
                    vals_dict[mk].append(mv)
        summary = {'lambda': info['lambda'], 'n_seeds': len(all_results[tag])}
        for mk, mvs in vals_dict.items():
            if isinstance(mvs[0], (int, float)):
                summary[f'{mk}_mean'] = float(np.mean(mvs))
                summary[f'{mk}_std'] = float(np.std(mvs))
        output[tag] = summary

    out_path = base / 'exp7a_aggregate.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved aggregate JSON: {out_path}")


if __name__ == '__main__':
    main()
