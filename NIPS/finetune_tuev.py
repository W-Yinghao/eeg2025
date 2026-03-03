#!/usr/bin/env python3
"""
TUEV Fine-tuning Script for CBraMod
Based on CBraMod finetuning code

Note: The train folder at /projects/EEG-foundation-model/tuev_cbramod_test/train
does not contain labels. Labels are extracted from the eval folder filenames.
This script uses the eval folder and splits it into train/val/test.

Usage:
    python finetune_tuev.py --cuda 0 --epochs 50 --batch_size 64

Classes (6 classes):
    0: bckg (background)
    1: spsw (spike and slow wave)
    2: gped (generalized periodic epileptiform discharges)
    3: pled (periodic lateralized epileptiform discharges)
    4: artf (artifact) - if present
    5: eyem (eye movement) - if present
"""

import argparse
import copy
import os
import pickle
import random
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add CBraMod to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CBraMod'))

from models.cbramod import CBraMod


# ======================== Configuration ========================
LABEL_MAP = {
    'bckg': 0,  # background
    'spsw': 1,  # spike and slow wave
    'gped': 2,  # generalized periodic epileptiform discharges
    'pled': 3,  # periodic lateralized epileptiform discharges
    'artf': 4,  # artifact (if present)
    'eyem': 5,  # eye movement (if present)
}


# ======================== Dataset ========================
class TUEVDataset(Dataset):
    """TUEV Dataset that extracts labels from filename prefixes."""

    def __init__(self, data_dir, files, label_map=None):
        super().__init__()
        self.data_dir = data_dir
        self.files = files
        self.label_map = label_map or LABEL_MAP

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.data_dir, file)

        # Load signal data
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        signal = data_dict['signal']  # Shape: (16, 400) for 200Hz * 2sec or (16, 1000) for 200Hz * 5sec

        # Extract label from filename prefix (e.g., 'bckg_007_a__seg_90377.pkl' -> 'bckg')
        prefix = file.split('_')[0].lower()
        label = self.label_map.get(prefix, -1)

        if label == -1:
            raise ValueError(f"Unknown label prefix: {prefix} in file {file}")

        # Handle different signal lengths
        # CBraMod expects (16, 5, 200) = 16 channels, 5 time segments, 200 samples per segment
        if signal.shape[1] == 400:
            # 400 samples -> reshape to (16, 2, 200)
            signal = signal.reshape(16, 2, 200)
        elif signal.shape[1] == 1000:
            # 1000 samples -> reshape to (16, 5, 200)
            signal = signal.reshape(16, 5, 200)
        else:
            # Try to adapt to (16, N, 200)
            n_segments = signal.shape[1] // 200
            if n_segments > 0:
                signal = signal[:, :n_segments*200].reshape(16, n_segments, 200)
            else:
                raise ValueError(f"Signal length {signal.shape[1]} cannot be reshaped to patches of 200")

        # Normalize
        signal = signal / 100.0

        return signal, label

    @staticmethod
    def collate_fn(batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return torch.from_numpy(x_data).float(), torch.from_numpy(y_label).long()


# ======================== Model ========================
class TUEVModel(nn.Module):
    """CBraMod model for TUEV classification."""

    def __init__(self, num_classes, pretrained_weights_path, classifier_type='all_patch_reps',
                 dropout=0.1, device='cuda:0', seq_len=5):
        super().__init__()

        self.seq_len = seq_len

        # Initialize backbone
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        # Load pretrained weights
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            map_location = torch.device(device)
            self.backbone.load_state_dict(torch.load(pretrained_weights_path, map_location=map_location))
        else:
            print("No pretrained weights loaded, training from scratch")

        # Replace projection layer with identity
        self.backbone.proj_out = nn.Identity()

        # Build classifier based on type
        # 16 channels * seq_len segments * 200 features
        input_dim = 16 * seq_len * 200

        if classifier_type == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, num_classes),
            )
        elif classifier_type == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, num_classes),
            )
        elif classifier_type == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, num_classes),
            )
        elif classifier_type == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(input_dim, seq_len * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, num_classes),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


# ======================== Training ========================
class Trainer:
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.model = model.cuda()
        self.criterion = CrossEntropyLoss(label_smoothing=params.label_smoothing).cuda()
        self.best_model_states = None

        # Set up optimizer with different learning rates for backbone and classifier
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if params.multi_lr:
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': params.lr},
                {'params': other_params, 'lr': 0.001 * (params.batch_size / 256) ** 0.5}
            ], weight_decay=params.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay
            )

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * self.data_length,
            eta_min=1e-6
        )

        print(self.model)

    def evaluate(self, data_loader, model):
        """Evaluate model on given data loader."""
        model.eval()
        truths = []
        preds = []

        for x, y in tqdm(data_loader, mininterval=1, desc="Evaluating"):
            x = x.cuda()
            y = y.cuda()

            with torch.no_grad():
                pred = model(x)
                pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().numpy().tolist()
            preds += pred_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)

        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)

        return acc, kappa, f1, cm

    def train(self):
        """Main training loop."""
        kappa_best = 0
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10, desc=f"Epoch {epoch+1}"):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            # Validation
            with torch.no_grad():
                acc, kappa, f1, cm = self.evaluate(self.data_loader['val'], self.model)

                print(
                    f"Epoch {epoch + 1}: Training Loss: {np.mean(losses):.5f}, "
                    f"Val Acc: {acc:.5f}, Kappa: {kappa:.5f}, F1: {f1:.5f}, "
                    f"LR: {optim_state['param_groups'][0]['lr']:.6f}, "
                    f"Time: {(timer() - start_time) / 60:.2f} mins"
                )
                print(f"Confusion Matrix:\n{cm}")

                if kappa > kappa_best:
                    print(f"Kappa improved from {kappa_best:.5f} to {kappa:.5f}, saving weights...")
                    best_epoch = epoch + 1
                    kappa_best = kappa
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        # Load best model and evaluate on test set
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        print("\n" + "=" * 50)
        print("Test Evaluation")
        print("=" * 50)

        with torch.no_grad():
            acc, kappa, f1, cm = self.evaluate(self.data_loader['test'], self.model)

            print(f"Test Results - Acc: {acc:.5f}, Kappa: {kappa:.5f}, F1: {f1:.5f}")
            print(f"Confusion Matrix:\n{cm}")

            # Save model
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            model_path = os.path.join(
                self.params.model_dir,
                f"tuev_epoch{best_epoch}_acc_{acc:.5f}_kappa_{kappa:.5f}_f1_{f1:.5f}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        return acc, kappa, f1


# ======================== Data Loading ========================
def load_data(params):
    """Load TUEV data from eval folder and split into train/val/test."""

    eval_dir = os.path.join(params.datasets_dir, 'eval')

    # Get all files
    all_files = os.listdir(eval_dir)
    print(f"Total files in eval folder: {len(all_files)}")

    # Filter files that have valid label prefixes
    valid_files = []
    label_counts = {}

    for f in all_files:
        prefix = f.split('_')[0].lower()
        if prefix in LABEL_MAP:
            valid_files.append(f)
            label_counts[prefix] = label_counts.get(prefix, 0) + 1

    print(f"Valid files with labels: {len(valid_files)}")
    print(f"Label distribution: {label_counts}")

    # Get unique labels present in data
    unique_labels = sorted(set([f.split('_')[0].lower() for f in valid_files]))
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {unique_labels}")

    # Update label map to only use present classes
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Label mapping: {label_map}")

    # Shuffle and split
    random.shuffle(valid_files)

    n_total = len(valid_files)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Create datasets
    train_set = TUEVDataset(eval_dir, train_files, label_map)
    val_set = TUEVDataset(eval_dir, val_files, label_map)
    test_set = TUEVDataset(eval_dir, test_files, label_map)

    # Create data loaders
    data_loader = {
        'train': DataLoader(
            train_set,
            batch_size=params.batch_size,
            collate_fn=TUEVDataset.collate_fn,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
        'val': DataLoader(
            val_set,
            batch_size=params.batch_size,
            collate_fn=TUEVDataset.collate_fn,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
        'test': DataLoader(
            test_set,
            batch_size=params.batch_size,
            collate_fn=TUEVDataset.collate_fn,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
    }

    return data_loader, num_classes, label_map


# ======================== Main ========================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='TUEV Fine-tuning with CBraMod')

    # Basic settings
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')

    # Training settings
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay')
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    # Model settings
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        choices=['all_patch_reps', 'all_patch_reps_twolayer',
                                 'all_patch_reps_onelayer', 'avgpooling_patch_reps'],
                        help='classifier head type')
    parser.add_argument('--multi_lr', type=bool, default=True, help='use different LRs for backbone and classifier')
    parser.add_argument('--frozen', type=bool, default=False, help='freeze backbone weights')

    # Data settings
    parser.add_argument('--datasets_dir', type=str,
                        default='/projects/EEG-foundation-model/tuev_cbramod_test',
                        help='path to TUEV dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')

    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str,
                        default='/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth',
                        help='path to pretrained weights')

    # Output
    parser.add_argument('--model_dir', type=str,
                        default='/home/infres/yinwang/eeg2025/NIPS/tuev_checkpoints',
                        help='directory to save model checkpoints')

    return parser.parse_args()


def main():
    params = parse_args()
    print(f"Parameters: {params}")

    # Set seed and device
    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    device = f'cuda:{params.cuda}'

    print(f"\n{'='*50}")
    print("TUEV Fine-tuning with CBraMod")
    print(f"{'='*50}\n")

    # Load data
    print("Loading data...")
    data_loader, num_classes, label_map = load_data(params)

    # Determine sequence length from data
    sample_batch = next(iter(data_loader['train']))
    seq_len = sample_batch[0].shape[2]  # (batch, channels, seq_len, patch_size)
    print(f"Data shape: {sample_batch[0].shape}")
    print(f"Sequence length: {seq_len}")

    # Create model
    print("\nCreating model...")
    model = TUEVModel(
        num_classes=num_classes,
        pretrained_weights_path=params.pretrained_weights,
        classifier_type=params.classifier,
        dropout=params.dropout,
        device=device,
        seq_len=seq_len
    )

    # Train
    print("\nStarting training...")
    trainer = Trainer(params, data_loader, model)
    acc, kappa, f1 = trainer.train()

    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Final Test Results - Acc: {acc:.5f}, Kappa: {kappa:.5f}, F1: {f1:.5f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
