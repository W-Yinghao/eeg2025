#!/usr/bin/env python3
"""
Aggregate Exp 7B results across 3 consistency lambda configs.

Usage:
    python experiments/deb/scripts/aggregate_exp7b.py [--base_dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def find_summaries(ckpt_dir: Path):
    results = []
    if not ckpt_dir.exists():
        return results
    for f in sorted(ckpt_dir.glob("best_*.json")):
        name = f.name
        if any(x in name for x in ('_config.json', '_classwise.json',
                                     '_gate_stats.json')):
            continue
        results.append(f)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=None)
    args = parser.parse_args()

    if args.base_dir:
        base = Path(args.base_dir)
    else:
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
        'l1e3': {'dir': 'exp7b_cons_l1e3_selector', 'lambda': '1e-3'},
        'l3e3': {'dir': 'exp7b_cons_l3e3_selector', 'lambda': '3e-3'},
        'l1e2': {'dir': 'exp7b_cons_l1e2_selector', 'lambda': '1e-2'},
    }

    all_results = {}
    for tag, info in configs.items():
        ckpt_dir = base / info['dir']
        summaries = find_summaries(ckpt_dir)
        runs = []
        for s in summaries:
            data = json.loads(s.read_text())
            test = data.get('test', {})
            gate = data.get('gate_stats', {})
            gate_path = s.parent / s.name.replace('.json', '_gate_stats.json')
            if gate_path.exists() and not gate:
                gate = json.loads(gate_path.read_text())
            runs.append({
                'seed': data.get('seed'),
                'bal_acc': test.get('bal_acc'),
                'f1_macro': test.get('f1_macro'),
                'abnormal_f1': test.get('abnormal_f1'),
                'abnormal_recall': test.get('abnormal_recall'),
                'gate_coverage': gate.get('gate_coverage_0.5'),
                'gate_entropy': gate.get('gate_entropy'),
                'gate_consistency_l2': gate.get('gate_consistency_l2'),
                'abnormal_only_consistency': gate.get('abnormal_only_consistency_l2'),
                'normal_only_consistency': gate.get('normal_only_consistency_l2'),
                'abnormal_only_gate_coverage': gate.get('abnormal_only_gate_coverage'),
                'best_epoch': data.get('best_epoch'),
            })
        all_results[tag] = runs

    metrics = [
        ('BalAcc', 'bal_acc'),
        ('Macro-F1', 'f1_macro'),
        ('Abnormal F1', 'abnormal_f1'),
        ('Abnormal Recall', 'abnormal_recall'),
        ('Gate Coverage', 'gate_coverage'),
        ('Gate Entropy', 'gate_entropy'),
        ('Overall Consistency', 'gate_consistency_l2'),
        ('Abnormal Consistency', 'abnormal_only_consistency'),
        ('Normal Consistency', 'normal_only_consistency'),
        ('Abn Gate Coverage', 'abnormal_only_gate_coverage'),
        ('Best Epoch', 'best_epoch'),
    ]

    print("=" * 85)
    print("  Exp 7B Aggregation: Frozen Selector + Consistency")
    print("=" * 85)

    header = f"{'Metric':<24s}"
    for tag, info in configs.items():
        n = len(all_results[tag])
        header += f" | lambda={info['lambda']:>6s} (n={n})"
    print(header)
    print("-" * len(header))

    for metric_name, metric_key in metrics:
        row = f"{metric_name:<24s}"
        for tag in configs:
            vals = [r[metric_key] for r in all_results[tag]
                    if r.get(metric_key) is not None]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                row += f" | {mean:>7.4f} +/- {std:.4f}  "
            else:
                row += f" |       N/A            "
        print(row)

    print()
    print("Per-seed details:")
    print("-" * 85)
    for tag, info in configs.items():
        print(f"\n  lambda={info['lambda']}:")
        for r in all_results[tag]:
            cons_str = f"Cons={r.get('gate_consistency_l2', 'N/A')}"
            if isinstance(r.get('gate_consistency_l2'), float):
                cons_str = f"Cons={r['gate_consistency_l2']:.4f}"
            print(f"    seed={r['seed']}: BalAcc={r.get('bal_acc','?'):.4f}  "
                  f"F1={r.get('f1_macro','?'):.4f}  "
                  f"AbnF1={r.get('abnormal_f1','?')}  "
                  f"{cons_str}  "
                  f"Ep={r.get('best_epoch', '?')}")

    # Save JSON
    output = {}
    for tag, info in configs.items():
        vals_dict = defaultdict(list)
        for r in all_results[tag]:
            for mk, mv in r.items():
                if mv is not None:
                    vals_dict[mk].append(mv)
        summary = {'lambda': info['lambda'], 'n_seeds': len(all_results[tag])}
        for mk, mvs in vals_dict.items():
            if mvs and isinstance(mvs[0], (int, float)):
                summary[f'{mk}_mean'] = float(np.mean(mvs))
                summary[f'{mk}_std'] = float(np.std(mvs))
        output[tag] = summary

    out_path = base / 'exp7b_aggregate.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved aggregate JSON: {out_path}")


if __name__ == '__main__':
    main()
