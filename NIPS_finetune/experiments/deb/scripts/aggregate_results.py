#!/usr/bin/env python3
"""
Unified result aggregation for Experiments 6/7/8.

Reads JSON summaries from checkpoints and explainability results,
produces:
  - Markdown summary table
  - CSV summary
  - Console output

Usage:
    # Aggregate Exp 6 results
    python experiments/deb/scripts/aggregate_results.py \
        --results_dir checkpoints_selector/ --experiment exp6

    # Aggregate Exp 7 results
    python experiments/deb/scripts/aggregate_results.py \
        --results_dir checkpoints_selector/ --experiment exp7

    # Aggregate Exp 8 explainability results
    python experiments/deb/scripts/aggregate_results.py \
        --results_dir results_explainability/ --experiment exp8

    # Aggregate all
    python experiments/deb/scripts/aggregate_results.py \
        --results_dir checkpoints_selector/ --experiment all
"""

import argparse
import json
import csv
import os
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_json_results(results_dir: str) -> list:
    """Load all JSON result files from a directory."""
    results = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        try:
            with open(fpath) as f:
                data = json.load(f)
                data['_source_file'] = os.path.basename(fpath)
                results.append(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: could not load {fpath}: {e}")
    return results


def aggregate_exp6(results: list) -> dict:
    """Aggregate Experiment 6: group by (regime, mode), compute mean/std over seeds."""
    groups = defaultdict(list)

    for r in results:
        if r.get('experiment') != 'selector':
            continue
        key = (r.get('regime', '?'), r.get('mode', '?'))
        groups[key].append(r)

    summary = {}
    for (regime, mode), runs in sorted(groups.items()):
        bal_accs = [r['test']['bal_acc'] for r in runs if 'test' in r]
        f1s = [r['test']['f1_macro'] for r in runs if 'test' in r]
        abn_recalls = [r['test'].get('abnormal_recall', np.nan) for r in runs if 'test' in r]
        abn_f1s = [r['test'].get('abnormal_f1', np.nan) for r in runs if 'test' in r]
        seeds = [r.get('seed', '?') for r in runs]
        trainable_ratios = [r.get('param_summary', {}).get('trainable_ratio', np.nan) for r in runs]

        entry = {
            'regime': regime,
            'mode': mode,
            'n_seeds': len(runs),
            'seeds': seeds,
            'bal_acc_mean': float(np.nanmean(bal_accs)) if bal_accs else 0,
            'bal_acc_std': float(np.nanstd(bal_accs)) if bal_accs else 0,
            'f1_macro_mean': float(np.nanmean(f1s)) if f1s else 0,
            'f1_macro_std': float(np.nanstd(f1s)) if f1s else 0,
            'trainable_ratio': float(np.nanmean(trainable_ratios)),
        }

        # Abnormal metrics (filter NaN)
        abn_rec_valid = [x for x in abn_recalls if not np.isnan(x)]
        abn_f1_valid = [x for x in abn_f1s if not np.isnan(x)]
        if abn_rec_valid:
            entry['abnormal_recall_mean'] = float(np.mean(abn_rec_valid))
            entry['abnormal_recall_std'] = float(np.std(abn_rec_valid))
        if abn_f1_valid:
            entry['abnormal_f1_mean'] = float(np.mean(abn_f1_valid))
            entry['abnormal_f1_std'] = float(np.std(abn_f1_valid))

        summary[f"{regime}_{mode}"] = entry

    return summary


def aggregate_exp7(results: list) -> dict:
    """Aggregate Experiment 7: group by variant (sparse/consistency/both)."""
    groups = defaultdict(list)

    for r in results:
        if r.get('experiment') != 'selector':
            continue
        cfg = r.get('config', {})
        has_sparse = cfg.get('enable_sparse', False)
        has_consist = cfg.get('enable_consistency', False)

        if has_sparse and has_consist:
            variant = 'sparse+consistency'
        elif has_sparse:
            variant = 'sparse'
        elif has_consist:
            variant = 'consistency'
        else:
            variant = 'plain_selector'

        key = (r.get('regime', '?'), variant)
        groups[key].append(r)

    summary = {}
    for (regime, variant), runs in sorted(groups.items()):
        bal_accs = [r['test']['bal_acc'] for r in runs if 'test' in r]
        f1s = [r['test']['f1_macro'] for r in runs if 'test' in r]

        entry = {
            'regime': regime,
            'variant': variant,
            'n_seeds': len(runs),
            'bal_acc_mean': float(np.nanmean(bal_accs)) if bal_accs else 0,
            'bal_acc_std': float(np.nanstd(bal_accs)) if bal_accs else 0,
            'f1_macro_mean': float(np.nanmean(f1s)) if f1s else 0,
            'f1_macro_std': float(np.nanstd(f1s)) if f1s else 0,
        }

        summary[f"{regime}_{variant}"] = entry

    return summary


def aggregate_exp8(results_dir: str) -> dict:
    """Aggregate Experiment 8 explainability results."""
    combined_file = os.path.join(results_dir, 'explainability_combined.json')
    if os.path.exists(combined_file):
        with open(combined_file) as f:
            return json.load(f)

    # Otherwise load individual files
    results = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, 'explainability_*.json'))):
        if 'combined' in fpath:
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
                results[Path(fpath).stem] = data
        except Exception:
            pass
    return results


def format_markdown_exp6(summary: dict) -> str:
    """Generate markdown table for Exp 6."""
    lines = [
        "## Experiment 6: True Partial FT Boundary Search",
        "",
        "| Regime | Mode | BalAcc | F1-Macro | Abn Recall | Abn F1 | Trainable% | Seeds |",
        "|--------|------|--------|----------|------------|--------|------------|-------|",
    ]

    for key, entry in sorted(summary.items()):
        bal_acc = f"{entry['bal_acc_mean']:.4f} +/- {entry['bal_acc_std']:.4f}"
        f1 = f"{entry['f1_macro_mean']:.4f} +/- {entry['f1_macro_std']:.4f}"
        abn_rec = (f"{entry.get('abnormal_recall_mean', 0):.4f} +/- "
                   f"{entry.get('abnormal_recall_std', 0):.4f}"
                   if 'abnormal_recall_mean' in entry else 'N/A')
        abn_f1 = (f"{entry.get('abnormal_f1_mean', 0):.4f} +/- "
                  f"{entry.get('abnormal_f1_std', 0):.4f}"
                  if 'abnormal_f1_mean' in entry else 'N/A')
        ratio = f"{entry['trainable_ratio']*100:.2f}%"

        lines.append(
            f"| {entry['regime']:8s} | {entry['mode']:8s} | {bal_acc} | {f1} "
            f"| {abn_rec} | {abn_f1} | {ratio} | {entry['n_seeds']} |"
        )

    return "\n".join(lines)


def format_markdown_exp7(summary: dict) -> str:
    """Generate markdown table for Exp 7."""
    lines = [
        "## Experiment 7: Selector Interpretability Enhancement",
        "",
        "| Regime | Variant | BalAcc | F1-Macro | Seeds |",
        "|--------|---------|--------|----------|-------|",
    ]

    for key, entry in sorted(summary.items()):
        bal_acc = f"{entry['bal_acc_mean']:.4f} +/- {entry['bal_acc_std']:.4f}"
        f1 = f"{entry['f1_macro_mean']:.4f} +/- {entry['f1_macro_std']:.4f}"
        lines.append(
            f"| {entry['regime']:8s} | {entry['variant']:20s} | {bal_acc} | {f1} "
            f"| {entry['n_seeds']} |"
        )

    return "\n".join(lines)


def format_markdown_exp8(results: dict) -> str:
    """Generate markdown table for Exp 8."""
    lines = [
        "## Experiment 8: Explainability Evaluation",
        "",
        "| Model | Ins AUC | Del AUC | T-Consist | F-Consist | Abn Recall | Abn F1 | Abn Coverage |",
        "|-------|---------|---------|-----------|-----------|------------|--------|--------------|",
    ]

    for name, data in sorted(results.items()):
        ins_del = data.get('insertion_deletion', {})
        aug_con = data.get('augmentation_consistency', {})
        abn = data.get('abnormal_analysis', {}).get('abnormal', {})
        abn_gate = data.get('abnormal_analysis', {}).get('abnormal_gate', {})

        lines.append(
            f"| {name[:30]} "
            f"| {ins_del.get('insertion_auc', 0):.4f} "
            f"| {ins_del.get('deletion_auc', 0):.4f} "
            f"| {aug_con.get('temporal_consistency_mean', 0):.4f} "
            f"| {aug_con.get('frequency_consistency_mean', 0):.4f} "
            f"| {abn.get('recall', 0):.4f} "
            f"| {abn.get('f1', 0):.4f} "
            f"| {abn_gate.get('coverage_mean', 0):.4f} |"
        )

    return "\n".join(lines)


def write_csv(summary: dict, output_path: str, fieldnames: list = None):
    """Write summary dict to CSV."""
    if not summary:
        return
    if fieldnames is None:
        # Collect all keys
        all_keys = set()
        for entry in summary.values():
            all_keys.update(entry.keys())
        fieldnames = sorted(all_keys)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for entry in summary.values():
            # Convert lists to strings
            row = {}
            for k, v in entry.items():
                if isinstance(v, list):
                    row[k] = str(v)
                else:
                    row[k] = v
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--results_dir', type=str, default='checkpoints_selector')
    parser.add_argument('--explainability_dir', type=str, default='results_explainability')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['exp6', 'exp7', 'exp8', 'all'])
    parser.add_argument('--output_dir', type=str, default='results_summary')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_json_results(args.results_dir)
    print(f"Loaded {len(results)} result files from {args.results_dir}")

    md_parts = ["# Selector Experiment Results\n"]

    if args.experiment in ('exp6', 'all'):
        print("\n--- Experiment 6 ---")
        exp6 = aggregate_exp6(results)
        if exp6:
            md = format_markdown_exp6(exp6)
            md_parts.append(md)
            print(md)
            write_csv(exp6, str(output_dir / 'exp6_summary.csv'))
            with open(output_dir / 'exp6_summary.json', 'w') as f:
                json.dump(exp6, f, indent=2)

    if args.experiment in ('exp7', 'all'):
        print("\n--- Experiment 7 ---")
        exp7 = aggregate_exp7(results)
        if exp7:
            md = format_markdown_exp7(exp7)
            md_parts.append(md)
            print(md)
            write_csv(exp7, str(output_dir / 'exp7_summary.csv'))
            with open(output_dir / 'exp7_summary.json', 'w') as f:
                json.dump(exp7, f, indent=2)

    if args.experiment in ('exp8', 'all'):
        print("\n--- Experiment 8 ---")
        exp8 = aggregate_exp8(args.explainability_dir)
        if exp8:
            md = format_markdown_exp8(exp8)
            md_parts.append(md)
            print(md)
            with open(output_dir / 'exp8_summary.json', 'w') as f:
                json.dump(exp8, f, indent=2)

    # Write combined markdown
    md_content = "\n\n".join(md_parts)
    with open(output_dir / 'results_summary.md', 'w') as f:
        f.write(md_content)
    print(f"\nMarkdown summary: {output_dir / 'results_summary.md'}")
    print(f"All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()
