#!/usr/bin/env python3
"""
TUAB Dataset Preprocessing Script

Preprocesses the TUH EEG Abnormal (TUAB) dataset and saves to LMDB format.

Source: /projects/EEG-foundation-model/tuh_eeg_abnormal/v3.0.1/edf/
Output: /projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed/

Binary classification:
    0: Normal
    1: Abnormal

Output format: LMDB with samples containing:
    - signal: (16, 2000) - 16 bipolar channels, 10 seconds at 200Hz
    - label: int (0 or 1)
    - source_file: str
    - segment_idx: int
"""

import os
import pickle
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool

import mne
import lmdb
import numpy as np
from tqdm import tqdm

# Suppress MNE verbose output
mne.set_log_level('WARNING')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
# Input/Output paths
ROOT_DIR = Path("/projects/EEG-foundation-model/tuh_eeg_abnormal/v3.0.1/edf")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed")

# Preprocessing parameters
TARGET_SFREQ = 200  # Hz
LOW_FREQ = 0.3      # Hz
HIGH_FREQ = 75      # Hz
NOTCH_FREQ = 60     # Hz
SEGMENT_DURATION = 10  # seconds
SAMPLES_PER_SEGMENT = TARGET_SFREQ * SEGMENT_DURATION  # 2000

# Channel standard
CHANNEL_STD = "01_tcp_ar"

# Train/val split ratio
TRAIN_RATIO = 0.8

# LMDB settings
LMDB_MAP_SIZE = 50 * 1024 * 1024 * 1024  # 50 GB

# Label names
LABEL_NAMES = {
    0: 'normal',
    1: 'abnormal',
}


# ============================================================================
# Bipolar Montage Conversion
# ============================================================================
def convert_to_bipolar(raw_data, ch_names):
    """
    Convert to 16-channel bipolar montage.

    Channels:
        0-3: Left temporal chain (FP1-F7, F7-T3, T3-T5, T5-O1)
        4-7: Right temporal chain (FP2-F8, F8-T4, T4-T6, T6-O2)
        8-11: Left parasagittal chain (FP1-F3, F3-C3, C3-P3, P3-O1)
        12-15: Right parasagittal chain (FP2-F4, F4-C4, C4-P4, P4-O2)
    """
    ch2idx = {name: i for i, name in enumerate(ch_names)}

    def get_ch(name):
        return raw_data[ch2idx[name]]

    bipolar = np.zeros((16, raw_data.shape[1]), dtype=np.float32)

    # Left temporal chain
    bipolar[0] = get_ch("EEG FP1-REF") - get_ch("EEG F7-REF")
    bipolar[1] = get_ch("EEG F7-REF") - get_ch("EEG T3-REF")
    bipolar[2] = get_ch("EEG T3-REF") - get_ch("EEG T5-REF")
    bipolar[3] = get_ch("EEG T5-REF") - get_ch("EEG O1-REF")

    # Right temporal chain
    bipolar[4] = get_ch("EEG FP2-REF") - get_ch("EEG F8-REF")
    bipolar[5] = get_ch("EEG F8-REF") - get_ch("EEG T4-REF")
    bipolar[6] = get_ch("EEG T4-REF") - get_ch("EEG T6-REF")
    bipolar[7] = get_ch("EEG T6-REF") - get_ch("EEG O2-REF")

    # Left parasagittal chain
    bipolar[8] = get_ch("EEG FP1-REF") - get_ch("EEG F3-REF")
    bipolar[9] = get_ch("EEG F3-REF") - get_ch("EEG C3-REF")
    bipolar[10] = get_ch("EEG C3-REF") - get_ch("EEG P3-REF")
    bipolar[11] = get_ch("EEG P3-REF") - get_ch("EEG O1-REF")

    # Right parasagittal chain
    bipolar[12] = get_ch("EEG FP2-REF") - get_ch("EEG F4-REF")
    bipolar[13] = get_ch("EEG F4-REF") - get_ch("EEG C4-REF")
    bipolar[14] = get_ch("EEG C4-REF") - get_ch("EEG P4-REF")
    bipolar[15] = get_ch("EEG P4-REF") - get_ch("EEG O2-REF")

    return bipolar


# ============================================================================
# LMDB Writing
# ============================================================================
class LMDBWriter:
    """LMDB writer for TUAB data."""

    def __init__(self, output_dir, split_name, map_size=LMDB_MAP_SIZE):
        self.output_dir = Path(output_dir)
        self.split_name = split_name
        self.lmdb_path = self.output_dir / split_name
        self.lmdb_path.mkdir(parents=True, exist_ok=True)

        self.env = lmdb.open(
            str(self.lmdb_path),
            map_size=map_size,
            readonly=False,
            meminit=False,
            map_async=True
        )
        self.count = 0
        self.label_counts = defaultdict(int)

    def add_sample(self, signal, label, source_file, segment_idx):
        """Add a single sample to LMDB."""
        sample = {
            'signal': signal.astype(np.float32),
            'label': int(label),
            'source_file': str(source_file),
            'segment_idx': int(segment_idx),
        }

        key = f'{self.count:08d}'.encode()
        value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)

        with self.env.begin(write=True) as txn:
            txn.put(key, value)

        self.label_counts[int(label)] += 1
        self.count += 1

    def close(self):
        """Close LMDB and write metadata."""
        # Write metadata
        metadata = {
            'n_samples': self.count,
            'n_channels': 16,
            'n_samples_per_segment': SAMPLES_PER_SEGMENT,
            'sampling_rate': TARGET_SFREQ,
            'segment_duration': SEGMENT_DURATION,
            'label_names': LABEL_NAMES,
            'label_counts': dict(self.label_counts),
            'created_at': datetime.now().isoformat(),
        }

        with self.env.begin(write=True) as txn:
            txn.put('__metadata__'.encode(), pickle.dumps(metadata))
            txn.put('__length__'.encode(), pickle.dumps(self.count))

            # Store keys list for iteration
            keys = [f'{i:08d}' for i in range(self.count)]
            txn.put('__keys__'.encode(), pickle.dumps(keys))

        self.env.close()

        # Save statistics
        stats_path = self.output_dir / f'{self.split_name}_statistics.txt'
        with open(stats_path, 'w') as f:
            f.write(f"TUAB {self.split_name} Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {self.count}\n")
            f.write(f"Channels: 16 (bipolar montage)\n")
            f.write(f"Samples per segment: {SAMPLES_PER_SEGMENT}\n")
            f.write(f"Sampling rate: {TARGET_SFREQ} Hz\n")
            f.write(f"Segment duration: {SEGMENT_DURATION} seconds\n\n")
            f.write("Label distribution:\n")
            for label_id in sorted(self.label_counts.keys()):
                label_name = LABEL_NAMES.get(label_id, 'unknown')
                count = self.label_counts[label_id]
                pct = 100 * count / self.count if self.count > 0 else 0
                f.write(f"  {label_id} ({label_name}): {count} ({pct:.1f}%)\n")

        logger.info(f"Saved {self.count} samples to {self.lmdb_path}")
        logger.info(f"Statistics saved to {stats_path}")

        return self.count


# ============================================================================
# Processing Functions
# ============================================================================
def process_file(file_path, label):
    """
    Process a single EDF file.

    Returns:
        List of (signal_segment, label, source_file, segment_idx) tuples
    """
    results = []

    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose='ERROR')
        raw.resample(TARGET_SFREQ)
        raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose='ERROR')
        raw.notch_filter(NOTCH_FREQ, verbose='ERROR')

        ch_names = raw.ch_names
        raw_data = raw.get_data(units='uV')

        # Convert to bipolar montage
        bipolar_data = convert_to_bipolar(raw_data, ch_names)
        raw.close()

        # Segment into 10-second windows
        n_segments = bipolar_data.shape[1] // SAMPLES_PER_SEGMENT
        for i in range(n_segments):
            segment = bipolar_data[:, i * SAMPLES_PER_SEGMENT:(i + 1) * SAMPLES_PER_SEGMENT]
            results.append((segment, label, file_path.name, i))

    except Exception as e:
        return None, str(e)

    return results, None


def get_subject_id(filename):
    """Extract subject ID from filename."""
    return filename.split("_")[0]


def main():
    """Main preprocessing function."""
    logger.info("=" * 60)
    logger.info("TUAB Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Source: {ROOT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Target sampling rate: {TARGET_SFREQ} Hz")
    logger.info(f"Filter: {LOW_FREQ}-{HIGH_FREQ} Hz, Notch: {NOTCH_FREQ} Hz")
    logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Paths
    train_val_abnormal = ROOT_DIR / "train" / "abnormal" / CHANNEL_STD
    train_val_normal = ROOT_DIR / "train" / "normal" / CHANNEL_STD
    test_abnormal = ROOT_DIR / "eval" / "abnormal" / CHANNEL_STD
    test_normal = ROOT_DIR / "eval" / "normal" / CHANNEL_STD

    # Get subject IDs for train/val split
    # Abnormal subjects
    train_val_a_files = list(train_val_abnormal.glob("*.edf"))
    train_val_a_subs = sorted(list(set([get_subject_id(f.name) for f in train_val_a_files])))
    split_idx = int(len(train_val_a_subs) * TRAIN_RATIO)
    train_a_subs = set(train_val_a_subs[:split_idx])
    val_a_subs = set(train_val_a_subs[split_idx:])

    # Normal subjects
    train_val_n_files = list(train_val_normal.glob("*.edf"))
    train_val_n_subs = sorted(list(set([get_subject_id(f.name) for f in train_val_n_files])))
    split_idx = int(len(train_val_n_subs) * TRAIN_RATIO)
    train_n_subs = set(train_val_n_subs[:split_idx])
    val_n_subs = set(train_val_n_subs[split_idx:])

    logger.info(f"Train abnormal subjects: {len(train_a_subs)}")
    logger.info(f"Val abnormal subjects: {len(val_a_subs)}")
    logger.info(f"Train normal subjects: {len(train_n_subs)}")
    logger.info(f"Val normal subjects: {len(val_n_subs)}")

    # Organize files by split
    splits = {
        'train': [],
        'val': [],
        'test': [],
    }

    # Train files
    for f in train_val_a_files:
        sub = get_subject_id(f.name)
        if sub in train_a_subs:
            splits['train'].append((f, 1))
        elif sub in val_a_subs:
            splits['val'].append((f, 1))

    for f in train_val_n_files:
        sub = get_subject_id(f.name)
        if sub in train_n_subs:
            splits['train'].append((f, 0))
        elif sub in val_n_subs:
            splits['val'].append((f, 0))

    # Test files
    for f in test_abnormal.glob("*.edf"):
        splits['test'].append((f, 1))
    for f in test_normal.glob("*.edf"):
        splits['test'].append((f, 0))

    logger.info(f"Train files: {len(splits['train'])}")
    logger.info(f"Val files: {len(splits['val'])}")
    logger.info(f"Test files: {len(splits['test'])}")

    # Process each split
    failed_files = []
    for split_name, files in splits.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing {split_name.upper()} split ({len(files)} files)")
        logger.info("=" * 40)

        writer = LMDBWriter(OUTPUT_DIR, split_name)

        for file_path, label in tqdm(files, desc=f"Processing {split_name}"):
            results, error = process_file(file_path, label)
            if error:
                failed_files.append((str(file_path), error))
                continue

            for segment, lbl, src_file, seg_idx in results:
                writer.add_sample(segment, lbl, src_file, seg_idx)

        writer.close()

    # Log failed files
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        failed_log = OUTPUT_DIR / 'failed_files.txt'
        with open(failed_log, 'w') as f:
            for path, error in failed_files:
                f.write(f"{path}: {error}\n")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")


# ============================================================================
# LMDB Reader (for verification)
# ============================================================================
class LMDBReader:
    """LMDB reader for verification."""

    def __init__(self, lmdb_path):
        self.lmdb_path = Path(lmdb_path)
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        with self.env.begin() as txn:
            self._length = pickle.loads(txn.get('__length__'.encode()))
            self._metadata = pickle.loads(txn.get('__metadata__'.encode()))

    def __len__(self):
        return self._length

    @property
    def metadata(self):
        return self._metadata

    def __getitem__(self, idx):
        key = f'{idx:08d}'.encode()
        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {idx} not found")
            return pickle.loads(value)

    def close(self):
        self.env.close()


def verify_output():
    """Verify the output LMDB files."""
    logger.info("\n" + "=" * 60)
    logger.info("Verifying Output")
    logger.info("=" * 60)

    for split in ['train', 'val', 'test']:
        lmdb_path = OUTPUT_DIR / split
        if lmdb_path.exists():
            reader = LMDBReader(lmdb_path)
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Samples: {len(reader)}")
            logger.info(f"  Metadata: {reader.metadata}")

            # Read a sample
            if len(reader) > 0:
                sample = reader[0]
                logger.info(f"  Sample signal shape: {sample['signal'].shape}")
                logger.info(f"  Sample label: {sample['label']}")

            reader.close()


if __name__ == '__main__':
    main()
    verify_output()
