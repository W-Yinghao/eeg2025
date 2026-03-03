#!/usr/bin/env python3
"""
TUSZ (TUH Seizure) Dataset Preprocessing Script

Preprocesses the TUH EEG Seizure dataset and saves to LMDB format.

Source: /projects/EEG-foundation-model/tuh_eeg_seizure/v2.0.3/edf/
Output: /projects/EEG-foundation-model/diagnosis_data/tusz_preprocessed/

TUSZ Labels (Binary classification: seizure vs non-seizure):
    0: bckg (background / non-seizure)
    1: seiz (any seizure type)

Original seizure types (for reference):
    - fnsz: focal non-specific seizure
    - gnsz: generalized non-specific seizure
    - spsz: simple partial seizure
    - cpsz: complex partial seizure
    - absz: absence seizure
    - tnsz: tonic seizure
    - tcsz: tonic-clonic seizure
    - atsz: atonic seizure
    - mysz: myoclonic seizure

Output format: LMDB with samples containing:
    - signal: (22, 1000) - 22 bipolar channels, 5 seconds at 200Hz
    - label: int (0=non-seizure, 1=seizure)
    - seizure_type: str (original seizure type or 'bckg')
    - source_file: str
    - segment_idx: int
    - start_time: float
    - end_time: float
"""

import os
import sys
import pickle
import logging
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import mne
import lmdb
import numpy as np
import pandas as pd
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
ROOT_DIR = Path("/projects/EEG-foundation-model/tuh_eeg_seizure/v2.0.3/edf")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data/tusz_preprocessed")

# Preprocessing parameters
TARGET_SFREQ = 200  # Hz
LOW_FREQ = 0.3      # Hz
HIGH_FREQ = 75      # Hz
NOTCH_FREQ = 60     # Hz
SEGMENT_DURATION = 5  # seconds
SEGMENT_OVERLAP = 0   # seconds (no overlap by default)

# LMDB settings
LMDB_MAP_SIZE = 100 * 1024 * 1024 * 1024  # 100 GB

# Seizure types - all non-bckg labels are seizures
SEIZURE_TYPES = {'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'atsz', 'mysz'}

# 22-channel TCP bipolar montage used in TUSZ
TCP_BIPOLAR_CHANNELS = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',      # Left temporal chain
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',      # Right temporal chain
    'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4',       # Central chain
    'C4-T4', 'T4-A2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',      # Left parasagittal chain
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',      # Right parasagittal chain
]

# Alternative channel names mapping
CHANNEL_ALIASES = {
    'EEG FP1-F7': 'FP1-F7', 'EEG F7-T3': 'F7-T3', 'EEG T3-T5': 'T3-T5', 'EEG T5-O1': 'T5-O1',
    'EEG FP2-F8': 'FP2-F8', 'EEG F8-T4': 'F8-T4', 'EEG T4-T6': 'T4-T6', 'EEG T6-O2': 'T6-O2',
    'EEG A1-T3': 'A1-T3', 'EEG T3-C3': 'T3-C3', 'EEG C3-CZ': 'C3-CZ', 'EEG CZ-C4': 'CZ-C4',
    'EEG C4-T4': 'C4-T4', 'EEG T4-A2': 'T4-A2',
    'EEG FP1-F3': 'FP1-F3', 'EEG F3-C3': 'F3-C3', 'EEG C3-P3': 'C3-P3', 'EEG P3-O1': 'P3-O1',
    'EEG FP2-F4': 'FP2-F4', 'EEG F4-C4': 'F4-C4', 'EEG C4-P4': 'C4-P4', 'EEG P4-O2': 'P4-O2',
}


# ============================================================================
# Annotation Parsing
# ============================================================================
def parse_csv_annotations(csv_path):
    """
    Parse TUSZ CSV annotation file.

    Returns:
        List of dicts with keys: channel, start_time, stop_time, label, confidence
    """
    annotations = []

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and header
            if line.startswith('#') or line.startswith('channel') or not line:
                continue

            parts = line.split(',')
            if len(parts) >= 4:
                annotations.append({
                    'channel': parts[0],
                    'start_time': float(parts[1]),
                    'stop_time': float(parts[2]),
                    'label': parts[3],
                    'confidence': float(parts[4]) if len(parts) > 4 else 1.0
                })

    return annotations


def get_segment_labels(annotations, start_time, end_time):
    """
    Determine the label for a segment based on annotations.

    A segment is labeled as seizure if any channel has a seizure annotation
    that overlaps with the segment.

    Returns:
        (binary_label, seizure_type)
        binary_label: 0 for non-seizure, 1 for seizure
        seizure_type: 'bckg' or specific seizure type
    """
    seizure_types_found = set()

    for ann in annotations:
        # Check if annotation overlaps with segment
        if ann['stop_time'] > start_time and ann['start_time'] < end_time:
            if ann['label'] in SEIZURE_TYPES:
                seizure_types_found.add(ann['label'])

    if seizure_types_found:
        # Return the most specific seizure type (prioritize by frequency)
        priority = ['cpsz', 'fnsz', 'gnsz', 'tcsz', 'spsz', 'absz', 'tnsz', 'mysz', 'atsz']
        for st in priority:
            if st in seizure_types_found:
                return 1, st
        return 1, list(seizure_types_found)[0]
    else:
        return 0, 'bckg'


# ============================================================================
# EDF Reading
# ============================================================================
def read_edf(edf_path):
    """
    Read and preprocess EDF file.

    Returns:
        signals: (n_channels, n_samples) preprocessed signals
        sfreq: sampling frequency after resampling
        duration: total duration in seconds
        raw: mne Raw object
    """
    edf_path = Path(edf_path)

    # Read EDF
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='ERROR')

    # Get original sampling frequency
    orig_sfreq = raw.info['sfreq']

    # Resample to target frequency
    if orig_sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad='auto')

    # Bandpass filter
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose='ERROR')

    # Notch filter for power line noise
    raw.notch_filter(NOTCH_FREQ, verbose='ERROR')

    # Get data in microvolts
    signals = raw.get_data(units='uV')
    duration = raw.times[-1]

    return signals, TARGET_SFREQ, duration, raw


def extract_bipolar_channels(signals, raw):
    """
    Extract 22 bipolar channels from the raw data.

    The TUSZ data is already in bipolar format (e.g., FP1-F7).
    We need to find and order the channels correctly.

    Returns:
        bipolar_signals: (22, n_samples) or (16, n_samples) array
        n_channels: number of channels extracted
    """
    ch_names = raw.info['ch_names']

    # Normalize channel names
    ch_name_map = {}
    for i, name in enumerate(ch_names):
        # Remove 'EEG ' prefix and normalize
        normalized = name.upper().replace('EEG ', '').replace('-LE', '').replace('-REF', '').strip()
        ch_name_map[normalized] = i

        # Also try with aliases
        if name in CHANNEL_ALIASES:
            ch_name_map[CHANNEL_ALIASES[name]] = i

    # Try to extract 22 channels, fall back to 16 if needed
    extracted_channels = []
    channel_indices = []

    for target_ch in TCP_BIPOLAR_CHANNELS:
        target_normalized = target_ch.upper()

        if target_normalized in ch_name_map:
            channel_indices.append(ch_name_map[target_normalized])
            extracted_channels.append(target_ch)
        else:
            # Try alternative names
            found = False
            for orig_name, idx in ch_name_map.items():
                if target_normalized in orig_name or orig_name in target_normalized:
                    channel_indices.append(idx)
                    extracted_channels.append(target_ch)
                    found = True
                    break

            if not found:
                # Use zero padding for missing channel
                channel_indices.append(None)
                extracted_channels.append(target_ch)

    # Extract signals
    n_samples = signals.shape[1]
    n_channels = len(TCP_BIPOLAR_CHANNELS)
    bipolar_signals = np.zeros((n_channels, n_samples), dtype=np.float32)

    for i, idx in enumerate(channel_indices):
        if idx is not None:
            bipolar_signals[i] = signals[idx]

    return bipolar_signals, n_channels


# ============================================================================
# Segmentation
# ============================================================================
def segment_recording(signals, annotations, sfreq, duration, segment_duration=5, overlap=0):
    """
    Segment a recording into fixed-length windows with labels.

    Args:
        signals: (n_channels, n_samples) array
        annotations: list of annotation dicts
        sfreq: sampling frequency
        duration: total duration in seconds
        segment_duration: length of each segment in seconds
        overlap: overlap between segments in seconds

    Returns:
        segments: list of (signal, label, seizure_type, start_time, end_time)
    """
    segments = []
    step = segment_duration - overlap
    n_samples_per_segment = int(sfreq * segment_duration)

    start_time = 0
    while start_time + segment_duration <= duration:
        end_time = start_time + segment_duration
        start_sample = int(start_time * sfreq)
        end_sample = start_sample + n_samples_per_segment

        if end_sample <= signals.shape[1]:
            segment_signal = signals[:, start_sample:end_sample]

            # Get label for this segment
            binary_label, seizure_type = get_segment_labels(annotations, start_time, end_time)

            segments.append({
                'signal': segment_signal,
                'label': binary_label,
                'seizure_type': seizure_type,
                'start_time': start_time,
                'end_time': end_time,
            })

        start_time += step

    return segments


# ============================================================================
# LMDB Writing
# ============================================================================
class LMDBWriter:
    """LMDB writer for TUSZ data."""

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
        self.seizure_type_counts = defaultdict(int)

    def add_sample(self, signal, label, seizure_type, source_file, segment_idx, start_time, end_time):
        """Add a single sample to LMDB."""
        sample = {
            'signal': signal.astype(np.float32),
            'label': int(label),
            'seizure_type': str(seizure_type),
            'source_file': str(source_file),
            'segment_idx': int(segment_idx),
            'start_time': float(start_time),
            'end_time': float(end_time),
        }

        key = f'{self.count:08d}'.encode()
        value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)

        with self.env.begin(write=True) as txn:
            txn.put(key, value)

        self.label_counts[int(label)] += 1
        self.seizure_type_counts[seizure_type] += 1
        self.count += 1

    def close(self):
        """Close LMDB and write metadata."""
        # Write metadata
        metadata = {
            'n_samples': self.count,
            'n_channels': 22,
            'n_samples_per_segment': int(TARGET_SFREQ * SEGMENT_DURATION),
            'sampling_rate': TARGET_SFREQ,
            'segment_duration': SEGMENT_DURATION,
            'label_names': {0: 'non-seizure', 1: 'seizure'},
            'label_counts': dict(self.label_counts),
            'seizure_type_counts': dict(self.seizure_type_counts),
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
            f.write(f"TUSZ {self.split_name} Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {self.count}\n")
            f.write(f"Channels: 22 (TCP bipolar montage)\n")
            f.write(f"Samples per segment: {int(TARGET_SFREQ * SEGMENT_DURATION)}\n")
            f.write(f"Sampling rate: {TARGET_SFREQ} Hz\n")
            f.write(f"Segment duration: {SEGMENT_DURATION} seconds\n\n")

            f.write("Binary label distribution:\n")
            for label_id in sorted(self.label_counts.keys()):
                label_name = 'non-seizure' if label_id == 0 else 'seizure'
                count = self.label_counts[label_id]
                pct = 100 * count / self.count if self.count > 0 else 0
                f.write(f"  {label_id} ({label_name}): {count} ({pct:.1f}%)\n")

            f.write("\nSeizure type distribution:\n")
            for stype in sorted(self.seizure_type_counts.keys()):
                count = self.seizure_type_counts[stype]
                pct = 100 * count / self.count if self.count > 0 else 0
                f.write(f"  {stype}: {count} ({pct:.1f}%)\n")

        logger.info(f"Saved {self.count} samples to {self.lmdb_path}")
        logger.info(f"Statistics saved to {stats_path}")

        return self.count


# ============================================================================
# Processing Functions
# ============================================================================
def process_split(base_dir, output_dir, split_name):
    """
    Process a data split (train, dev, or eval).

    Args:
        base_dir: Path to split directory
        output_dir: Output directory for LMDB
        split_name: 'train', 'dev', or 'eval'

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
            # Find corresponding CSV annotation file
            csv_path = edf_path.with_suffix('.csv')
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV annotation not found: {csv_path}")

            # Parse annotations
            annotations = parse_csv_annotations(csv_path)

            # Read and preprocess EDF
            signals, sfreq, duration, raw = read_edf(edf_path)

            # Extract bipolar channels
            bipolar_signals, n_channels = extract_bipolar_channels(signals, raw)
            raw.close()

            # Segment recording
            segments = segment_recording(
                bipolar_signals, annotations, sfreq, duration,
                segment_duration=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP
            )

            # Write segments to LMDB
            for idx, seg in enumerate(segments):
                writer.add_sample(
                    signal=seg['signal'],
                    label=seg['label'],
                    seizure_type=seg['seizure_type'],
                    source_file=edf_path.name,
                    segment_idx=idx,
                    start_time=seg['start_time'],
                    end_time=seg['end_time'],
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
    logger.info("TUSZ (TUH Seizure) Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Source: {ROOT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Target sampling rate: {TARGET_SFREQ} Hz")
    logger.info(f"Filter: {LOW_FREQ}-{HIGH_FREQ} Hz, Notch: {NOTCH_FREQ} Hz")
    logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each split
    total_samples = 0

    for split_name in ['train', 'dev', 'eval']:
        logger.info("\n" + "=" * 40)
        logger.info(f"Processing {split_name.upper()} split")
        logger.info("=" * 40)

        split_count = process_split(
            ROOT_DIR / split_name,
            OUTPUT_DIR,
            split_name
        )
        total_samples += split_count

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total_samples}")
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

    for split in ['train', 'dev', 'eval']:
        lmdb_path = OUTPUT_DIR / split
        if lmdb_path.exists():
            try:
                reader = LMDBReader(lmdb_path)
                logger.info(f"\n{split.upper()}:")
                logger.info(f"  Samples: {len(reader)}")
                logger.info(f"  Label counts: {reader.metadata.get('label_counts', 'N/A')}")
                logger.info(f"  Seizure types: {reader.metadata.get('seizure_type_counts', 'N/A')}")

                # Read a sample
                if len(reader) > 0:
                    sample = reader[0]
                    logger.info(f"  Sample signal shape: {sample['signal'].shape}")
                    logger.info(f"  Sample label: {sample['label']} ({sample['seizure_type']})")

                reader.close()
            except Exception as e:
                logger.error(f"Failed to verify {split}: {e}")


if __name__ == '__main__':
    main()
    verify_output()
