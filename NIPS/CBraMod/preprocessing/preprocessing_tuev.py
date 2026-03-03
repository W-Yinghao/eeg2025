#!/usr/bin/env python3
"""
TUEV Dataset Preprocessing Script

Preprocesses the TUH EEG Events (TUEV) dataset and saves to LMDB format.

Source: /projects/EEG-foundation-model/tuh_eeg_events/v2.0.1/edf/
Output: /projects/EEG-foundation-model/diagnosis_data/tuev_preprocessed/

Reference: https://github.com/Abhishaike/EEG_Event_Classification

TUEV Labels:
    1: SPSW (spike and slow wave)
    2: GPED (generalized periodic epileptiform discharges)
    3: PLED (periodic lateralized epileptiform discharges)
    4: EYEM (eye movement)
    5: ARTF (artifact)
    6: BCKG (background)

Output format: LMDB with samples containing:
    - signal: (16, 1000) - 16 bipolar channels, 5 seconds at 200Hz
    - label: int (0-5, mapped from 1-6)
    - offending_channel: int
    - source_file: str
    - segment_idx: int
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
ROOT_DIR = Path("/projects/EEG-foundation-model/tuh_eeg_events/v2.0.1/edf")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data/tuev_preprocessed")

# Preprocessing parameters
TARGET_SFREQ = 200  # Hz
LOW_FREQ = 0.3      # Hz
HIGH_FREQ = 75      # Hz
NOTCH_FREQ = 60     # Hz
SEGMENT_DURATION = 5  # seconds (2 seconds before and after event center)

# LMDB settings
LMDB_MAP_SIZE = 50 * 1024 * 1024 * 1024  # 50 GB

# Label mapping (original 1-6 to 0-5)
LABEL_NAMES = {
    1: 'spsw',   # spike and slow wave
    2: 'gped',   # generalized periodic epileptiform discharges
    3: 'pled',   # periodic lateralized epileptiform discharges
    4: 'eyem',   # eye movement
    5: 'artf',   # artifact
    6: 'bckg',   # background
}


# ============================================================================
# Bipolar Montage Conversion
# ============================================================================
def convert_to_bipolar(signals, raw):
    """
    Convert to 16-channel bipolar montage.

    Channels:
        0-3: Left temporal chain (FP1-F7, F7-T3, T3-T5, T5-O1)
        4-7: Right temporal chain (FP2-F8, F8-T4, T4-T6, T6-O2)
        8-11: Left parasagittal chain (FP1-F3, F3-C3, C3-P3, P3-O1)
        12-15: Right parasagittal chain (FP2-F4, F4-C4, C4-P4, P4-O2)
    """
    ch2idx = {name: i for i, name in enumerate(raw.info["ch_names"])}

    def get_ch(name):
        return signals[ch2idx[name]]

    try:
        bipolar = np.vstack([
            # Left temporal chain
            get_ch("EEG FP1-REF") - get_ch("EEG F7-REF"),   # 0
            get_ch("EEG F7-REF") - get_ch("EEG T3-REF"),    # 1
            get_ch("EEG T3-REF") - get_ch("EEG T5-REF"),    # 2
            get_ch("EEG T5-REF") - get_ch("EEG O1-REF"),    # 3
            # Right temporal chain
            get_ch("EEG FP2-REF") - get_ch("EEG F8-REF"),   # 4
            get_ch("EEG F8-REF") - get_ch("EEG T4-REF"),    # 5
            get_ch("EEG T4-REF") - get_ch("EEG T6-REF"),    # 6
            get_ch("EEG T6-REF") - get_ch("EEG O2-REF"),    # 7
            # Left parasagittal chain
            get_ch("EEG FP1-REF") - get_ch("EEG F3-REF"),   # 8
            get_ch("EEG F3-REF") - get_ch("EEG C3-REF"),    # 9
            get_ch("EEG C3-REF") - get_ch("EEG P3-REF"),    # 10
            get_ch("EEG P3-REF") - get_ch("EEG O1-REF"),    # 11
            # Right parasagittal chain
            get_ch("EEG FP2-REF") - get_ch("EEG F4-REF"),   # 12
            get_ch("EEG F4-REF") - get_ch("EEG C4-REF"),    # 13
            get_ch("EEG C4-REF") - get_ch("EEG P4-REF"),    # 14
            get_ch("EEG P4-REF") - get_ch("EEG O2-REF"),    # 15
        ])
        return bipolar.astype(np.float32)
    except KeyError as e:
        raise ValueError(f"Missing channel: {e}")


# ============================================================================
# EDF Reading
# ============================================================================
def read_edf(edf_path):
    """
    Read and preprocess EDF file.

    Returns:
        signals: (n_channels, n_samples) preprocessed signals in uV
        times: time points array
        event_data: (n_events, 4) array from .rec file
        raw: mne Raw object
    """
    edf_path = Path(edf_path)
    rec_path = edf_path.with_suffix('.rec')

    if not rec_path.exists():
        raise FileNotFoundError(f"REC file not found: {rec_path}")

    # Read EDF
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='ERROR')

    # Resample to target frequency
    raw.resample(TARGET_SFREQ, npad='auto')

    # Bandpass filter
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose='ERROR')

    # Notch filter for power line noise
    raw.notch_filter(NOTCH_FREQ, verbose='ERROR')

    # Get data
    _, times = raw[:]
    signals = raw.get_data(units='uV')

    # Read event data from .rec file
    event_data = np.genfromtxt(str(rec_path), delimiter=',')
    if event_data.ndim == 1:
        event_data = event_data.reshape(1, -1)

    return signals, times, event_data, raw


# ============================================================================
# Event Extraction
# ============================================================================
def build_events(signals, times, event_data, fs=200.0):
    """
    Extract 5-second windows around each annotated event.

    Args:
        signals: (n_channels, n_samples) bipolar signals
        times: time points array
        event_data: (n_events, 4) - [channel, start_time, end_time, label]
        fs: sampling frequency

    Returns:
        features: (n_events, 16, 1000) event windows
        offending_channels: (n_events,) channel indices
        labels: (n_events,) event labels (1-6)
    """
    n_events = event_data.shape[0]
    n_channels = signals.shape[1] if signals.ndim == 1 else signals.shape[0]
    window_samples = int(fs * SEGMENT_DURATION)  # 5 seconds = 1000 samples

    features = np.zeros((n_events, 16, window_samples), dtype=np.float32)
    offending_channels = np.zeros(n_events, dtype=np.int32)
    labels = np.zeros(n_events, dtype=np.int32)

    # Pad signals for edge cases (concatenate 3 copies)
    offset = signals.shape[1]
    padded_signals = np.concatenate([signals, signals, signals], axis=1)

    for i in range(n_events):
        chan = int(event_data[i, 0])
        event_start = event_data[i, 1]
        event_end = event_data[i, 2]
        label = int(event_data[i, 3])

        # Find sample indices
        start_idx = np.searchsorted(times, event_start)
        end_idx = np.searchsorted(times, event_end)

        # Extract window: 2 seconds before event start to 2 seconds after event end
        # (or centered if event is short)
        win_start = offset + start_idx - 2 * int(fs)
        win_end = offset + end_idx + 2 * int(fs)

        # Ensure exactly window_samples
        if win_end - win_start != window_samples:
            # Center the window
            center = (win_start + win_end) // 2
            win_start = center - window_samples // 2
            win_end = win_start + window_samples

        features[i, :, :] = padded_signals[:, win_start:win_end]
        offending_channels[i] = chan
        labels[i] = label

    return features, offending_channels, labels


# ============================================================================
# LMDB Writing
# ============================================================================
class LMDBWriter:
    """LMDB writer for TUEV data."""

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

    def add_sample(self, signal, label, offending_channel, source_file, segment_idx):
        """Add a single sample to LMDB."""
        sample = {
            'signal': signal.astype(np.float32),
            'label': int(label - 1),  # Convert 1-6 to 0-5
            'offending_channel': int(offending_channel),
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
            'n_samples_per_segment': int(TARGET_SFREQ * SEGMENT_DURATION),
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
            f.write(f"TUEV {self.split_name} Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {self.count}\n")
            f.write(f"Channels: 16 (bipolar montage)\n")
            f.write(f"Samples per segment: {int(TARGET_SFREQ * SEGMENT_DURATION)}\n")
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
def process_split(base_dir, output_dir, split_name):
    """
    Process a data split (train or eval).

    Args:
        base_dir: Path to split directory containing subject folders
        output_dir: Output directory for LMDB
        split_name: 'train' or 'eval'

    Returns:
        Number of samples processed
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        return 0

    # Find all EDF files
    edf_files = list(base_dir.rglob("*.edf"))
    edf_files.sort()

    logger.info(f"Found {len(edf_files)} EDF files in {split_name}")

    # Initialize LMDB writer
    writer = LMDBWriter(output_dir, split_name)

    # Process each file
    failed_files = []
    for edf_path in tqdm(edf_files, desc=f"Processing {split_name}"):
        try:
            # Read and preprocess
            signals, times, event_data, raw = read_edf(edf_path)

            # Convert to bipolar montage
            bipolar_signals = convert_to_bipolar(signals, raw)
            raw.close()

            # Extract events
            features, off_channels, labels = build_events(
                bipolar_signals, times, event_data, fs=TARGET_SFREQ
            )

            # Write to LMDB
            for idx, (feat, off_ch, label) in enumerate(zip(features, off_channels, labels)):
                writer.add_sample(
                    signal=feat,
                    label=label,
                    offending_channel=off_ch,
                    source_file=edf_path.name,
                    segment_idx=idx
                )

        except Exception as e:
            failed_files.append((str(edf_path), str(e)))
            logger.debug(f"Failed to process {edf_path.name}: {e}")
            continue

    # Close writer and save
    n_samples = writer.close()

    # Log failed files
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files in {split_name}")
        failed_log = output_dir / f'{split_name}_failed_files.txt'
        with open(failed_log, 'w') as f:
            for path, error in failed_files:
                f.write(f"{path}: {error}\n")

    return n_samples


def main():
    """Main preprocessing function."""
    logger.info("=" * 60)
    logger.info("TUEV Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Source: {ROOT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Target sampling rate: {TARGET_SFREQ} Hz")
    logger.info(f"Filter: {LOW_FREQ}-{HIGH_FREQ} Hz, Notch: {NOTCH_FREQ} Hz")
    logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process train split
    logger.info("\n" + "=" * 40)
    logger.info("Processing TRAIN split")
    logger.info("=" * 40)
    train_count = process_split(
        ROOT_DIR / "train",
        OUTPUT_DIR,
        "train"
    )

    # Process eval split
    logger.info("\n" + "=" * 40)
    logger.info("Processing EVAL split")
    logger.info("=" * 40)
    eval_count = process_split(
        ROOT_DIR / "eval",
        OUTPUT_DIR,
        "eval"
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Train samples: {train_count}")
    logger.info(f"Eval samples: {eval_count}")
    logger.info(f"Total samples: {train_count + eval_count}")
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

    for split in ['train', 'eval']:
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
