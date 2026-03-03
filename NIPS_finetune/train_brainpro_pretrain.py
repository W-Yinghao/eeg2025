#!/usr/bin/env python3
"""
BrainPro Pre-Training Script

Pre-trains BrainPro on multiple EEG datasets grouped by brain state
(affect, motor, others) using:
    - Region-aware masked reconstruction loss (Eq 18)
    - Brain-state decoupling loss (Eq 19)
    - Selective gradient updates (only shared + active state encoder)

Pre-training ablation configurations (Table 3):
    --no_masking         → w/o masking
    --no_reconstruction  → w/o reconstruction
    --no_decoupling      → w/o decoupling
    --random_retrieval   → w random retrieval

Hyperparameters (Table 4):
    - Batch size: 160 (32 per GPU)
    - Peak LR: 1e-4, Min LR: 1e-5
    - Cosine schedule with 2 warmup epochs
    - AdamW, β=(0.9, 0.98), weight_decay=0.05
    - Gradient clipping: 3
    - Mask ratio: 0.5
    - 30 epochs

Usage:
    # Full pre-training
    python train_brainpro_pretrain.py --data_config pretrain_config.json --epochs 30

    # Ablation: no masking
    python train_brainpro_pretrain.py --data_config pretrain_config.json --no_masking

    # Ablation: no decoupling
    python train_brainpro_pretrain.py --data_config pretrain_config.json --no_decoupling

Data Config Format (pretrain_config.json):
    {
        "affect": [
            {"path": "/path/to/affect_data.lmdb", "n_channels": 62, "sampling_rate": 200,
             "channel_names": ["FP1", "FPZ", ...]}
        ],
        "motor": [...],
        "others": [...]
    }
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(__file__))
from brainpro_model import (
    create_brainpro_pretrain,
    UNIVERSAL_CHANNELS,
    resolve_channel_indices,
    get_region_mapping,
)


# =============================================================================
# Pre-training Dataset
# =============================================================================

class PretrainEEGDataset(Dataset):
    """Generic EEG dataset for BrainPro pre-training.

    Loads EEG segments from LMDB files and returns raw (C, T) data
    with brain-state labels.

    Supports:
    - Variable channel configurations (mapped to universal template)
    - Resampling to 200Hz
    - Segmentation to fixed-length clips (10s default)
    """

    def __init__(
        self,
        lmdb_path: str,
        brain_state: str,
        n_channels: int = 60,
        sampling_rate: int = 200,
        segment_duration: float = 10.0,
        channel_names: list = None,
    ):
        self.lmdb_path = lmdb_path
        self.brain_state = brain_state
        self.n_channels = n_channels
        self.target_T = int(segment_duration * 200)  # 10s * 200Hz = 2000

        # Open LMDB
        self.env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
        with self.env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # Build key list
        self.keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                self.keys.append(key)

        # Channel mapping
        self.channel_names = channel_names
        if channel_names:
            self.channel_indices = resolve_channel_indices(channel_names)
        else:
            self.channel_indices = list(range(min(n_channels, 60)))

        print(f"  Loaded {len(self.keys)} samples from {lmdb_path} ({brain_state})")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            raw = txn.get(self.keys[idx])
        data = pickle.loads(raw)

        if isinstance(data, dict):
            signal = data.get('signal', data.get('data', data.get('eeg')))
            if signal is None:
                signal = list(data.values())[0]
        elif isinstance(data, (tuple, list)):
            signal = data[0]
        else:
            signal = data

        signal = np.array(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)

        C, T = signal.shape

        # Truncate or pad to target T
        if T > self.target_T:
            start = random.randint(0, T - self.target_T)
            signal = signal[:, start:start + self.target_T]
        elif T < self.target_T:
            padded = np.zeros((C, self.target_T), dtype=np.float32)
            padded[:, :T] = signal
            signal = padded

        # Channel handling: ensure we have consistent n_channels
        target_C = self.n_channels
        if C > target_C:
            signal = signal[:target_C, :]
        elif C < target_C:
            padded = np.zeros((target_C, self.target_T), dtype=np.float32)
            padded[:C, :] = signal
            signal = padded

        # Normalize
        signal = signal / 100.0

        return signal, self.brain_state


# =============================================================================
# Training utilities
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
    """Cosine schedule with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser(description='BrainPro Pre-Training')

    # Data
    parser.add_argument('--data_config', type=str, required=True,
                        help='JSON config file specifying pre-training datasets')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size per GPU (total = this * n_gpus)')
    parser.add_argument('--lr', type=float, default=1e-4, help='peak learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--clip_grad_norm', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)

    # Architecture
    parser.add_argument('--K_T', type=int, default=32)
    parser.add_argument('--K_C', type=int, default=32)
    parser.add_argument('--K_R', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--n_encoder_layers', type=int, default=4)
    parser.add_argument('--n_decoder_layers', type=int, default=2)
    parser.add_argument('--patch_len', type=int, default=20)
    parser.add_argument('--patch_stride', type=int, default=20)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--decoupling_margin', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Ablation flags
    parser.add_argument('--no_masking', action='store_true',
                        help='ablation: disable masking')
    parser.add_argument('--no_reconstruction', action='store_true',
                        help='ablation: disable reconstruction loss')
    parser.add_argument('--no_decoupling', action='store_true',
                        help='ablation: disable decoupling loss')
    parser.add_argument('--random_retrieval', action='store_true',
                        help='ablation: use random spatial filters')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_brainpro_pretrain')
    parser.add_argument('--save_every', type=int, default=10,
                        help='save checkpoint every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Load data config
    with open(args.data_config) as f:
        data_config = json.load(f)

    print(f"\n{'='*60}")
    print(f"BrainPro Pre-Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Ablation: masking={'ON' if not args.no_masking else 'OFF'}, "
          f"recon={'ON' if not args.no_reconstruction else 'OFF'}, "
          f"decouple={'ON' if not args.no_decoupling else 'OFF'}, "
          f"retrieval={'random' if args.random_retrieval else 'learned'}")

    # Build dataloaders per brain state
    brain_states = ['affect', 'motor', 'others']
    dataloaders = {}
    n_channels_max = 0

    for state in brain_states:
        datasets_list = []
        for ds_info in data_config.get(state, []):
            ds = PretrainEEGDataset(
                lmdb_path=ds_info['path'],
                brain_state=state,
                n_channels=ds_info.get('n_channels', 60),
                sampling_rate=ds_info.get('sampling_rate', 200),
                segment_duration=ds_info.get('segment_duration', 10.0),
                channel_names=ds_info.get('channel_names'),
            )
            datasets_list.append(ds)
            n_channels_max = max(n_channels_max, ds_info.get('n_channels', 60))

        if datasets_list:
            combined = ConcatDataset(datasets_list)
            dataloaders[state] = DataLoader(
                combined, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True,
            )
            print(f"  {state}: {len(combined)} samples")
        else:
            print(f"  {state}: no data")

    if not dataloaders:
        print("ERROR: No training data found!")
        return

    # Use universal template channels for the model
    n_channels = n_channels_max
    channel_indices = list(range(min(n_channels, 60)))

    # Create model
    model = create_brainpro_pretrain(
        n_channels=n_channels,
        random_retrieval=args.random_retrieval,
        K_T=args.K_T, K_C=args.K_C, K_R=args.K_R,
        d_model=args.d_model, nhead=args.nhead, d_ff=args.d_ff,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        patch_len=args.patch_len, patch_stride=args.patch_stride,
        mask_ratio=args.mask_ratio,
        decoupling_margin=args.decoupling_margin,
        dropout=args.dropout,
    ).to(device)

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.98), weight_decay=args.weight_decay,
    )

    # Scheduler: Cosine with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr
    )

    # WandB
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            ablation_tag = []
            if args.no_masking: ablation_tag.append('no_mask')
            if args.no_reconstruction: ablation_tag.append('no_recon')
            if args.no_decoupling: ablation_tag.append('no_decouple')
            if args.random_retrieval: ablation_tag.append('random_retrieval')
            if not ablation_tag: ablation_tag.append('full')

            run_name = args.wandb_run_name or f"brainpro_pretrain_{'_'.join(ablation_tag)}"
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name,
                config=vars(args), tags=['pretrain', 'brainpro'] + ablation_tag,
            )
        except ImportError:
            pass

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting pre-training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_dec = 0.0
        epoch_batches = 0

        model.train()

        # Iterate through brain states in round-robin
        iterators = {state: iter(dl) for state, dl in dataloaders.items()}
        active_states = list(iterators.keys())

        while active_states:
            for state in list(active_states):
                try:
                    batch = next(iterators[state])
                except StopIteration:
                    active_states.remove(state)
                    continue

                eeg_data = batch[0].to(device)

                optimizer.zero_grad()
                outputs = model(
                    eeg_data,
                    brain_state=state,
                    channel_indices=channel_indices,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    use_masking=not args.no_masking,
                    use_reconstruction=not args.no_reconstruction,
                    use_decoupling=not args.no_decoupling,
                )

                loss = outputs['loss']
                if loss.requires_grad:
                    loss.backward()
                    if args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_rec += outputs.get('loss_rec', torch.tensor(0.0)).item()
                epoch_dec += outputs.get('loss_dec', torch.tensor(0.0)).item()
                epoch_batches += 1
                global_step += 1

        scheduler.step()
        t_elapsed = time.time() - t_start

        avg_loss = epoch_loss / max(epoch_batches, 1)
        avg_rec = epoch_rec / max(epoch_batches, 1)
        avg_dec = epoch_dec / max(epoch_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} rec={avg_rec:.4f} dec={avg_dec:.4f} | "
              f"lr={current_lr:.6f} | {t_elapsed:.1f}s")

        if wandb_run:
            wandb_run.log({
                'epoch': epoch,
                'train/loss': avg_loss,
                'train/loss_rec': avg_rec,
                'train/loss_dec': avg_dec,
                'lr': current_lr,
                'time/epoch_seconds': t_elapsed,
            })

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f'brainpro_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"Pre-training complete!")
    print(f"{'='*60}")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
