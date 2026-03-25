#!/usr/bin/env python3
"""Parse baseline SLURM logs and upload to wandb retrospectively."""

import re
import glob
import argparse
import wandb


def parse_log(filepath):
    """Parse a single SLURM log file, return config + epoch data + test results."""
    with open(filepath) as f:
        text = f.read()

    # Check if experiment completed
    if "All done." not in text:
        return None

    # Parse config from header: "Baseline: codebrain | TUAB | finetune=frozen | head=pool"
    header = re.search(
        r"Baseline:\s+(\w+)\s+\|\s+([\w-]+)\s+\|\s+finetune=(\w+)\s+\|\s+head=(\w+)",
        text,
    )
    if not header:
        return None

    model, dataset, finetune, head = header.groups()

    # Parse seed from filename: bl_codebrain_TUAB_frozen_s3407.out
    seed_match = re.search(r"_s(\d+)\.out$", filepath)
    seed = int(seed_match.group(1)) if seed_match else 0

    # Parse epoch lines
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"Train:\s+total=([\d.]+)\s+ce=([\d.]+)\s+acc=([\d.]+)\s+\|\s+"
        r"Val:\s+ce=([\d.]+)\s+bal_acc=([\d.]+)\s+f1=([\d.]+)\s+\|\s+([\d.]+)s"
    )
    epochs = []
    for m in epoch_pattern.finditer(text):
        epochs.append({
            "epoch": int(m.group(1)),
            "train/total_loss": float(m.group(3)),
            "train/ce_loss": float(m.group(4)),
            "train/acc": float(m.group(5)),
            "val/ce_loss": float(m.group(6)),
            "val/bal_acc": float(m.group(7)),
            "val/f1": float(m.group(8)),
            "epoch_time_s": float(m.group(9)),
        })

    # Parse early stopping
    es_match = re.search(r"Early stopping at epoch (\d+) \(best: epoch (\d+)", text)
    best_epoch = int(es_match.group(2)) if es_match else len(epochs)

    # Parse test results
    test = {}
    for key, pattern in [
        ("test/bal_acc", r"Balanced Accuracy:\s+([\d.]+)"),
        ("test/macro_f1", r"Macro F1:\s+([\d.]+)"),
        ("test/weighted_f1", r"Weighted F1:\s+([\d.]+)"),
        ("test/ce_loss", r"CE Loss:\s+([\d.]+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            test[key] = float(m.group(1))

    # Parse per-class F1
    per_class = re.findall(r"(\S+)\s+:\s+([\d.]+)", text.split("Per-class F1:")[-1]) if "Per-class F1:" in text else []
    for cls_name, f1_val in per_class:
        test[f"test/f1_{cls_name}"] = float(f1_val)

    return {
        "model": model,
        "dataset": dataset,
        "finetune": finetune,
        "head": head,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "test": test,
    }


def upload_to_wandb(parsed, project="eeg_deb"):
    """Upload a parsed log to wandb."""
    config = {
        "model": parsed["model"],
        "dataset": parsed["dataset"],
        "finetune": parsed["finetune"],
        "head_type": parsed["head"],
        "seed": parsed["seed"],
        "mode": "baseline",
        "best_epoch": parsed["best_epoch"],
    }

    run_name = f"Baseline_{parsed['model']}_{parsed['dataset']}_{parsed['finetune']}_pool_s{parsed['seed']}"

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=["baseline", parsed["dataset"], parsed["finetune"], "from_log"],
    )

    # Log epoch-level metrics
    for ep in parsed["epochs"]:
        step = ep.pop("epoch")
        wandb.log(ep, step=step)

    # Log test results as summary
    for k, v in parsed["test"].items():
        wandb.summary[k] = v
    wandb.summary["best_epoch"] = parsed["best_epoch"]

    run.finish()
    print(f"  Uploaded: {run_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="slurm_log/baselines_codebrain")
    parser.add_argument("--project", default="eeg_deb")
    parser.add_argument("--dry_run", action="store_true", help="Parse only, don't upload")
    args = parser.parse_args()

    log_files = sorted(glob.glob(f"{args.log_dir}/bl_codebrain_*.out"))
    print(f"Found {len(log_files)} log files")

    uploaded = 0
    skipped = 0
    for f in log_files:
        parsed = parse_log(f)
        if parsed is None:
            print(f"  Skipped (incomplete): {f}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] {f}: {parsed['dataset']}/{parsed['finetune']}/s{parsed['seed']} "
                  f"-> test bal_acc={parsed['test'].get('test/bal_acc', 'N/A')}")
        else:
            upload_to_wandb(parsed, project=args.project)
        uploaded += 1

    print(f"\nDone: {uploaded} uploaded, {skipped} skipped")


if __name__ == "__main__":
    main()
