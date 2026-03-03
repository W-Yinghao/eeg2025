#!/usr/bin/env python3
"""
CHB-MIT Dataset Preprocessing Script

Preprocesses the CHB-MIT Scalp EEG dataset and saves to LMDB format.

Source: /projects/EEG-foundation-model/CHB-MIT/
Output: /projects/EEG-foundation-model/diagnosis_data/CHB-MIT_preprocessed/

CHB-MIT Labels (Binary classification: seizure vs non-seizure):
    0: non-seizure (background)
    1: seizure

Dataset Info:
    - 23 pediatric patients with intractable seizures
    - Sampling rate: 256 Hz
    - 23 EEG channels (16 commonly used bipolar channels)

Output format: LMDB with samples containing:
    - signal: (16, 1280) - 16 bipolar channels, 5 seconds at 256Hz
    - label: int (0=non-seizure, 1=seizure)
    - patient_id: str (e.g., 'chb01')
    - source_file: str
    - segment_idx: int
    - start_time: float
    - end_time: float

Train/Val/Test split:
    - Test: chb23, chb24
    - Val: chb21, chb22
    - Train: chb01-chb20 (excluding chb12, chb13 which have different channel configs)
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
ROOT_DIR = Path("/projects/EEG-foundation-model/CHB-MIT")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data/CHB-MIT_preprocessed")

# Preprocessing parameters
ORIG_SFREQ = 256    # Original sampling rate
TARGET_SFREQ = 256  # Keep original (or set to 200 for consistency with other datasets)
LOW_FREQ = 0.3      # Hz
HIGH_FREQ = 75      # Hz
NOTCH_FREQ = 60     # Hz
SEGMENT_DURATION = 5  # seconds
SEGMENT_OVERLAP = 0   # seconds (no overlap for non-seizure, sliding window for seizure)

# LMDB settings
LMDB_MAP_SIZE = 50 * 1024 * 1024 * 1024  # 50 GB

# Patient splits
TEST_PATIENTS = ['chb23', 'chb24']
VAL_PATIENTS = ['chb21', 'chb22']
# Exclude chb12, chb13 due to different channel configurations
EXCLUDED_PATIENTS = ['chb12', 'chb13']

# 16-channel bipolar montage (standard subset)
BIPOLAR_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',      # Left temporal chain
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',      # Right temporal chain
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',      # Left parasagittal chain
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',      # Right parasagittal chain
]

# Channel name aliases for matching
CHANNEL_ALIASES = {
    'FP1-F7': ['FP1-F7', 'EEG FP1-F7'],
    'F7-T7': ['F7-T7', 'EEG F7-T7', 'F7-T3'],
    'T7-P7': ['T7-P7', 'EEG T7-P7', 'T3-T5'],
    'P7-O1': ['P7-O1', 'EEG P7-O1', 'T5-O1'],
    'FP2-F8': ['FP2-F8', 'EEG FP2-F8'],
    'F8-T8': ['F8-T8', 'EEG F8-T8', 'F8-T4'],
    'T8-P8': ['T8-P8', 'EEG T8-P8', 'T4-T6'],
    'P8-O2': ['P8-O2', 'EEG P8-O2', 'T6-O2'],
    'FP1-F3': ['FP1-F3', 'EEG FP1-F3'],
    'F3-C3': ['F3-C3', 'EEG F3-C3'],
    'C3-P3': ['C3-P3', 'EEG C3-P3'],
    'P3-O1': ['P3-O1', 'EEG P3-O1'],
    'FP2-F4': ['FP2-F4', 'EEG FP2-F4'],
    'F4-C4': ['F4-C4', 'EEG F4-C4'],
    'C4-P4': ['C4-P4', 'EEG C4-P4'],
    'P4-O2': ['P4-O2', 'EEG P4-O2'],
}


# ============================================================================
# Summary File Parsing
# ============================================================================
def parse_summary_file(summary_path):
    """
    Parse CHB-MIT summary file to extract seizure annotations.

    Returns:
        dict mapping filename to list of (start_time, end_time) tuples
    """
    annotations = {}

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Find file name
        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()
            annotations[current_file] = []

        # Find number of seizures
        elif line.startswith('Number of Seizures in File:'):
            num_seizures = int(line.split(':')[1].strip())

            # Parse seizure times
            for _ in range(num_seizures):
                i += 1
                while i < len(lines):
                    start_line = lines[i].strip()
                    if 'Seizure' in start_line and 'Start' in start_line:
                        # Extract start time
                        start_time = int(re.search(r'(\d+)\s*seconds', start_line).group(1))

                        # Next line should be end time
                        i += 1
                        end_line = lines[i].strip()
                        end_time = int(re.search(r'(\d+)\s*seconds', end_line).group(1))

                        if current_file:
                            annotations[current_file].append((start_time, end_time))
                        break
                    i += 1

        i += 1

    return annotations


# ============================================================================
# EDF Reading
# ============================================================================
def read_edf(edf_path):
    """
    Read and preprocess EDF file.

    Returns:
        signals: (n_channels, n_samples) preprocessed signals
        sfreq: sampling frequency
        duration: total duration in seconds
        ch_names: list of channel names
    """
    edf_path = Path(edf_path)

    # Read EDF
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='ERROR')

    # Resample if needed
    if raw.info['sfreq'] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad='auto')

    # Bandpass filter
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose='ERROR')

    # Notch filter for power line noise
    raw.notch_filter(NOTCH_FREQ, verbose='ERROR')

    # Get data in microvolts
    signals = raw.get_data(units='uV')
    duration = raw.times[-1]
    ch_names = raw.info['ch_names']

    return signals, TARGET_SFREQ, duration, ch_names


def extract_bipolar_channels(signals, ch_names):
    """
    Extract 16 bipolar channels from the raw data.

    Returns:
        bipolar_signals: (16, n_samples) array
        n_valid_channels: number of valid channels found
    """
    # Normalize channel names
    ch_name_map = {}
    for i, name in enumerate(ch_names):
        # Normalize: uppercase, remove spaces
        normalized = name.upper().replace(' ', '').replace('EEG', '').strip()
        ch_name_map[normalized] = i
        ch_name_map[name] = i

    # Extract channels
    n_samples = signals.shape[1]
    bipolar_signals = np.zeros((len(BIPOLAR_CHANNELS), n_samples), dtype=np.float32)
    n_valid = 0

    for i, target_ch in enumerate(BIPOLAR_CHANNELS):
        found = False

        # Try all aliases
        for alias in CHANNEL_ALIASES.get(target_ch, [target_ch]):
            normalized_alias = alias.upper().replace(' ', '').replace('EEG', '').strip()

            if normalized_alias in ch_name_map:
                bipolar_signals[i] = signals[ch_name_map[normalized_alias]]
                found = True
                n_valid += 1
                break

            if alias in ch_name_map:
                bipolar_signals[i] = signals[ch_name_map[alias]]
                found = True
                n_valid += 1
                break

        if not found:
            # Try fuzzy matching
            for ch_name, idx in ch_name_map.items():
                if target_ch.replace('-', '') in ch_name.replace('-', '').upper():
                    bipolar_signals[i] = signals[idx]
                    n_valid += 1
                    break

    return bipolar_signals, n_valid


# ============================================================================
# Segmentation
# ============================================================================
def segment_recording(signals, seizure_times, sfreq, duration, segment_duration=5):
    """
    Segment a recording into fixed-length windows with labels.

    For seizure segments, use overlapping windows to increase samples.

    Args:
        signals: (n_channels, n_samples) array
        seizure_times: list of (start_time, end_time) tuples in seconds
        sfreq: sampling frequency
        duration: total duration in seconds
        segment_duration: length of each segment in seconds

    Returns:
        segments: list of dicts with signal, label, start_time, end_time
    """
    segments = []
    n_samples_per_segment = int(sfreq * segment_duration)

    # Non-overlapping segments for the entire recording
    start_time = 0
    while start_time + segment_duration <= duration:
        end_time = start_time + segment_duration
        start_sample = int(start_time * sfreq)
        end_sample = start_sample + n_samples_per_segment

        if end_sample <= signals.shape[1]:
            segment_signal = signals[:, start_sample:end_sample]

            # Check if segment overlaps with any seizure
            label = 0
            for sz_start, sz_end in seizure_times:
                if (start_time < sz_end and end_time > sz_start):
                    label = 1
                    break

            segments.append({
                'signal': segment_signal,
                'label': label,
                'start_time': start_time,
                'end_time': end_time,
            })

        start_time += segment_duration

    # Additional overlapping segments for seizure periods (data augmentation)
    for sz_start, sz_end in seizure_times:
        # Use sliding window with 50% overlap around seizure
        window_start = max(0, sz_start - segment_duration)
        window_end = min(duration, sz_end + segment_duration)

        step = segment_duration / 2  # 50% overlap
        t = window_start
        while t + segment_duration <= window_end:
            start_sample = int(t * sfreq)
            end_sample = start_sample + n_samples_per_segment

            if end_sample <= signals.shape[1]:
                segment_signal = signals[:, start_sample:end_sample]

                # Only add if it overlaps with seizure
                if t < sz_end and (t + segment_duration) > sz_start:
                    segments.append({
                        'signal': segment_signal,
                        'label': 1,
                        'start_time': t,
                        'end_time': t + segment_duration,
                        'augmented': True,
                    })

            t += step

    return segments


# ============================================================================
# LMDB Writing
# ============================================================================
class LMDBWriter:
    """LMDB writer for CHB-MIT data."""

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
        self.patient_counts = defaultdict(int)

    def add_sample(self, signal, label, patient_id, source_file, segment_idx, start_time, end_time):
        """Add a single sample to LMDB."""
        sample = {
            'signal': signal.astype(np.float32),
            'label': int(label),
            'patient_id': str(patient_id),
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
        self.patient_counts[patient_id] += 1
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
            'label_names': {0: 'non-seizure', 1: 'seizure'},
            'label_counts': dict(self.label_counts),
            'patient_counts': dict(self.patient_counts),
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
            f.write(f"CHB-MIT {self.split_name} Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {self.count}\n")
            f.write(f"Channels: 16 (bipolar montage)\n")
            f.write(f"Samples per segment: {int(TARGET_SFREQ * SEGMENT_DURATION)}\n")
            f.write(f"Sampling rate: {TARGET_SFREQ} Hz\n")
            f.write(f"Segment duration: {SEGMENT_DURATION} seconds\n\n")

            f.write("Label distribution:\n")
            for label_id in sorted(self.label_counts.keys()):
                label_name = 'non-seizure' if label_id == 0 else 'seizure'
                count = self.label_counts[label_id]
                pct = 100 * count / self.count if self.count > 0 else 0
                f.write(f"  {label_id} ({label_name}): {count} ({pct:.1f}%)\n")

            f.write("\nPatient distribution:\n")
            for patient in sorted(self.patient_counts.keys()):
                count = self.patient_counts[patient]
                pct = 100 * count / self.count if self.count > 0 else 0
                f.write(f"  {patient}: {count} ({pct:.1f}%)\n")

        logger.info(f"Saved {self.count} samples to {self.lmdb_path}")
        logger.info(f"Statistics saved to {stats_path}")

        return self.count


# ============================================================================
# Processing Functions
# ============================================================================
def get_patient_split(patient_id):
    """Determine which split a patient belongs to."""
    if patient_id in TEST_PATIENTS:
        return 'test'
    elif patient_id in VAL_PATIENTS:
        return 'val'
    elif patient_id in EXCLUDED_PATIENTS:
        return None
    else:
        return 'train'


def process_patient(patient_dir, writers):
    """
    Process all EDF files for a patient.

    Args:
        patient_dir: Path to patient directory
        writers: dict of LMDBWriter objects for each split

    Returns:
        Number of segments processed
    """
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name

    # Determine split
    split = get_patient_split(patient_id)
    if split is None:
        logger.info(f"Skipping {patient_id} (excluded)")
        return 0

    # Parse summary file
    summary_path = patient_dir / f"{patient_id}-summary.txt"
    if not summary_path.exists():
        logger.warning(f"Summary file not found: {summary_path}")
        return 0

    try:
        annotations = parse_summary_file(summary_path)
    except Exception as e:
        logger.warning(f"Failed to parse summary for {patient_id}: {e}")
        return 0

    # Process each EDF file
    n_segments = 0
    edf_files = sorted(patient_dir.glob("*.edf"))

    for edf_path in edf_files:
        try:
            # Get seizure times for this file
            seizure_times = annotations.get(edf_path.name, [])

            # Read and preprocess
            signals, sfreq, duration, ch_names = read_edf(edf_path)

            # Extract bipolar channels
            bipolar_signals, n_valid = extract_bipolar_channels(signals, ch_names)

            if n_valid < 12:  # Require at least 12 valid channels
                logger.debug(f"Skipping {edf_path.name}: only {n_valid} valid channels")
                continue

            # Segment recording
            segments = segment_recording(
                bipolar_signals, seizure_times, sfreq, duration,
                segment_duration=SEGMENT_DURATION
            )

            # Write to LMDB
            for idx, seg in enumerate(segments):
                writers[split].add_sample(
                    signal=seg['signal'],
                    label=seg['label'],
                    patient_id=patient_id,
                    source_file=edf_path.name,
                    segment_idx=idx,
                    start_time=seg['start_time'],
                    end_time=seg['end_time'],
                )
                n_segments += 1

        except Exception as e:
            logger.debug(f"Failed to process {edf_path.name}: {e}")
            continue

    return n_segments


def main():
    """Main preprocessing function."""
    logger.info("=" * 60)
    logger.info("CHB-MIT Dataset Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Source: {ROOT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Sampling rate: {TARGET_SFREQ} Hz")
    logger.info(f"Filter: {LOW_FREQ}-{HIGH_FREQ} Hz, Notch: {NOTCH_FREQ} Hz")
    logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize writers for each split
    writers = {
        'train': LMDBWriter(OUTPUT_DIR, 'train'),
        'val': LMDBWriter(OUTPUT_DIR, 'val'),
        'test': LMDBWriter(OUTPUT_DIR, 'test'),
    }

    # Find all patient directories
    patient_dirs = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')])

    logger.info(f"Found {len(patient_dirs)} patient directories")

    # Process each patient
    total_segments = 0
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        n_segments = process_patient(patient_dir, writers)
        total_segments += n_segments
        logger.info(f"  {patient_dir.name}: {n_segments} segments")

    # Close writers
    for split, writer in writers.items():
        writer.close()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total segments: {total_segments}")
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
            try:
                reader = LMDBReader(lmdb_path)
                logger.info(f"\n{split.upper()}:")
                logger.info(f"  Samples: {len(reader)}")
                logger.info(f"  Label counts: {reader.metadata.get('label_counts', 'N/A')}")
                logger.info(f"  Patients: {list(reader.metadata.get('patient_counts', {}).keys())}")

                # Read a sample
                if len(reader) > 0:
                    sample = reader[0]
                    logger.info(f"  Sample signal shape: {sample['signal'].shape}")
                    logger.info(f"  Sample label: {sample['label']}")
                    logger.info(f"  Sample patient: {sample['patient_id']}")

                reader.close()
            except Exception as e:
                logger.error(f"Failed to verify {split}: {e}")


if __name__ == '__main__':
    main()
    verify_output()
