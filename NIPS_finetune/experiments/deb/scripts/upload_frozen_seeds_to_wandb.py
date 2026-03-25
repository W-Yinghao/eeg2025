#!/usr/bin/env python3
"""
Parse baseline + DEB SLURM logs from deb_log/ and upload to wandb.

Supports multiple experiment sets via --exp flag:
  frozen:        frozen baseline + DEB (jobs 1259590-1259599)
  partial_full:  partial + full baseline + DEB (jobs 1259692-1259711)

Usage:
    # Dry run
    python experiments/deb/scripts/upload_frozen_seeds_to_wandb.py --exp frozen --dry_run
    python experiments/deb/scripts/upload_frozen_seeds_to_wandb.py --exp partial_full --dry_run

    # Upload
    python experiments/deb/scripts/upload_frozen_seeds_to_wandb.py --exp frozen
    python experiments/deb/scripts/upload_frozen_seeds_to_wandb.py --exp partial_full

    # Custom group
    python experiments/deb/scripts/upload_frozen_seeds_to_wandb.py --exp partial_full --group "my_group"
"""

import re
import argparse
import wandb


# ── Experiment definitions ──────────────────────────────────────────────
# Each entry: job_id -> (mode, finetune, seed)

EXPERIMENTS = {
    "frozen": {
        "group_default": "frozen_codebrain_TUAB_seeds",
        "jobs": {
            1259590: ("baseline", "frozen", 7),
            1259591: ("deb",      "frozen", 7),
            1259592: ("baseline", "frozen", 1234),
            1259593: ("deb",      "frozen", 1234),
            1259594: ("baseline", "frozen", 2025),
            1259595: ("deb",      "frozen", 2025),
            1259596: ("baseline", "frozen", 77),
            1259597: ("deb",      "frozen", 77),
            1259598: ("baseline", "frozen", 7777),
            1259599: ("deb",      "frozen", 7777),
        },
    },
    "partial_full": {
        "group_default": "partial_full_codebrain_TUAB_seeds",
        "jobs": {
            1259692: ("baseline", "partial", 7),
            1259693: ("deb",      "partial", 7),
            1259694: ("baseline", "full",    7),
            1259695: ("deb",      "full",    7),
            1259696: ("baseline", "partial", 1234),
            1259697: ("deb",      "partial", 1234),
            1259698: ("baseline", "full",    1234),
            1259699: ("deb",      "full",    1234),
            1259700: ("baseline", "partial", 2025),
            1259701: ("deb",      "partial", 2025),
            1259702: ("baseline", "full",    2025),
            1259703: ("deb",      "full",    2025),
            1259704: ("baseline", "partial", 77),
            1259705: ("deb",      "partial", 77),
            1259706: ("baseline", "full",    77),
            1259707: ("deb",      "full",    77),
            1259708: ("baseline", "partial", 7777),
            1259709: ("deb",      "partial", 7777),
            1259710: ("baseline", "full",    7777),
            1259711: ("deb",      "full",    7777),
        },
    },
}


def parse_out_log(filepath, fallback_mode=None, fallback_finetune=None, fallback_seed=None):
    """Parse a .out log file -> config dict + epoch data + test results."""
    with open(filepath) as f:
        text = f.read()

    mode = fallback_mode
    model = dataset = finetune = None
    seed = fallback_seed

    # Try parsing header: "Baseline: codebrain | TUAB | frozen | seed=7"
    bl_header = re.search(
        r"Baseline:\s+(\w+)\s+\|\s+([\w-]+)\s+\|\s+(\w+)\s+\|\s+seed=(\d+)",
        text,
    )
    # Try parsing header: "DEB: codebrain | TUAB | partial | seed=7"
    deb_header = re.search(
        r"DEB:\s+(\w+)\s+\|\s+([\w-]+)\s+\|\s+(\w+)\s+\|\s+seed=(\d+)",
        text,
    )
    # Try parsing old header: "DEB Training: codebrain on TUAB"
    deb_old_header = re.search(
        r"DEB Training:\s+(\w+)\s+on\s+([\w-]+)",
        text,
    )

    if bl_header:
        mode = "baseline"
        model, dataset, finetune, seed = bl_header.groups()
        seed = int(seed)
    elif deb_header:
        mode = "deb"
        model, dataset, finetune, seed = deb_header.groups()
        seed = int(seed)
    elif deb_old_header:
        mode = "deb"
        model, dataset = deb_old_header.groups()
        finetune = fallback_finetune or "frozen"
        seed = fallback_seed
    else:
        return None

    # Parse model summary for extra info
    params_match = re.search(r"Trainable params:\s+([\d,]+)", text)
    trainable_params = int(params_match.group(1).replace(",", "")) if params_match else None

    # Parse epoch lines - DEB format (with kl and sp)
    deb_epoch_pat = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"Train:\s+total=([\d.]+)\s+ce=([\d.]+)\s+acc=([\d.]+)\s+\|\s+"
        r"Val:\s+ce=([\d.]+)\s+bal_acc=([\d.]+)\s+f1=([\d.]+)\s+\|\s+"
        r"kl=([\d.]+)\s+sp=([\d.]+)\s+\|\s+([\d.]+)s"
    )
    # Baseline format (no kl/sp)
    baseline_epoch_pat = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"Train:\s+total=([\d.]+)\s+ce=([\d.]+)\s+acc=([\d.]+)\s+\|\s+"
        r"Val:\s+ce=([\d.]+)\s+bal_acc=([\d.]+)\s+f1=([\d.]+)\s+\|\s+([\d.]+)s"
    )

    epochs = []
    for m in deb_epoch_pat.finditer(text):
        epochs.append({
            "epoch": int(m.group(1)),
            "train/total_loss": float(m.group(3)),
            "train/ce_loss": float(m.group(4)),
            "train/bal_acc": float(m.group(5)),
            "val/ce_loss": float(m.group(6)),
            "val/bal_acc": float(m.group(7)),
            "val/f1_macro": float(m.group(8)),
            "train/kl": float(m.group(9)),
            "train/sparse": float(m.group(10)),
            "epoch_time_s": float(m.group(11)),
        })

    if not epochs:
        for m in baseline_epoch_pat.finditer(text):
            epochs.append({
                "epoch": int(m.group(1)),
                "train/total_loss": float(m.group(3)),
                "train/ce_loss": float(m.group(4)),
                "train/bal_acc": float(m.group(5)),
                "val/ce_loss": float(m.group(6)),
                "val/bal_acc": float(m.group(7)),
                "val/f1_macro": float(m.group(8)),
                "epoch_time_s": float(m.group(9)),
            })

    # Parse early stopping
    es_match = re.search(r"Early stopping at epoch (\d+) \(best: epoch (\d+)", text)
    best_epoch = int(es_match.group(2)) if es_match else (len(epochs) if epochs else 0)

    # Parse test results
    test = {}
    for key, pattern in [
        ("test/bal_acc", r"Balanced Accuracy:\s+([\d.]+)"),
        ("test/f1_macro", r"Macro F1:\s+([\d.]+)"),
        ("test/f1_weighted", r"Weighted F1:\s+([\d.]+)"),
        ("test/ce_loss", r"CE Loss:\s+([\d.]+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            test[key] = float(m.group(1))

    # Parse per-class F1
    if "Per-class F1:" in text:
        per_class_text = text.split("Per-class F1:")[-1].split("Confusion Matrix:")[0]
        per_class = re.findall(r"(\S+)\s+:\s+([\d.]+)", per_class_text)
        for cls_name, f1_val in per_class:
            test[f"test/f1_{cls_name}"] = float(f1_val)

    return {
        "mode": mode,
        "model": model,
        "dataset": dataset,
        "finetune": finetune,
        "seed": seed,
        "best_epoch": best_epoch,
        "trainable_params": trainable_params,
        "epochs": epochs,
        "test": test,
    }


def upload_to_wandb(parsed, project, group, log_file):
    """Upload a parsed log to wandb, attaching the .out file as an artifact."""
    mode = parsed["mode"]
    model = parsed["model"]
    dataset = parsed["dataset"]
    finetune = parsed["finetune"]
    seed = parsed["seed"]

    run_name = f"{mode}_{model}_{dataset}_{finetune}_s{seed}"

    config = {
        "model": model,
        "dataset": dataset,
        "finetune": finetune,
        "seed": seed,
        "mode": mode,
        "best_epoch": parsed["best_epoch"],
        "trainable_params": parsed["trainable_params"],
    }

    tags = [mode, dataset, finetune, "from_log", f"seed_{seed}"]

    run = wandb.init(
        project=project,
        group=group,
        name=run_name,
        config=config,
        tags=tags,
    )

    # Log epoch-level metrics
    for ep in parsed["epochs"]:
        step = ep["epoch"]
        metrics = {k: v for k, v in ep.items() if k != "epoch"}
        wandb.log(metrics, step=step)

    # Log test results as summary
    for k, v in parsed["test"].items():
        wandb.summary[k] = v
    wandb.summary["best_epoch"] = parsed["best_epoch"]

    # Upload .out log file
    wandb.save(log_file, policy="now")

    run.finish()
    print(f"  Uploaded: {run_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload DEB experiment logs to wandb")
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()),
                        help="Experiment set to upload")
    parser.add_argument("--log_dir", default="deb_log")
    parser.add_argument("--project", default="eeg_deb")
    parser.add_argument("--group", default=None, help="W&B group name (default: auto from exp)")
    parser.add_argument("--dry_run", action="store_true", help="Parse only, don't upload")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.exp]
    group = args.group or exp["group_default"]

    print(f"Experiment: {args.exp}  |  Group: {group}  |  Project: {args.project}")
    print("=" * 70)

    uploaded = 0
    skipped = 0

    for job_id, (expected_mode, expected_ft, seed) in sorted(exp["jobs"].items()):
        out_file = f"{args.log_dir}/{job_id}.out"
        try:
            parsed = parse_out_log(
                out_file,
                fallback_mode=expected_mode,
                fallback_finetune=expected_ft,
                fallback_seed=seed,
            )
        except FileNotFoundError:
            print(f"  Not found: {out_file}")
            skipped += 1
            continue

        if parsed is None:
            print(f"  Skipped (parse failed): {out_file}")
            skipped += 1
            continue

        # Fill seed for old DEB header format (no seed in header)
        if parsed["seed"] is None:
            parsed["seed"] = seed

        test_acc = parsed["test"].get("test/bal_acc", "N/A")
        test_f1 = parsed["test"].get("test/f1_macro", "N/A")
        n_epochs = len(parsed["epochs"])

        print(f"  [{parsed['mode']:>8}] {parsed['finetune']:<8} seed={parsed['seed']:<5} "
              f"epochs={n_epochs:<3} best={parsed['best_epoch']:<3} "
              f"test_acc={test_acc}  test_f1={test_f1}")

        if not args.dry_run:
            upload_to_wandb(parsed, project=args.project, group=group, log_file=out_file)
        uploaded += 1

    print(f"\nDone: {uploaded} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
