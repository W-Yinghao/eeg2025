#!/usr/bin/env python3
"""
CVD vs Normal EEG Dataset Preprocessing Script

Preprocesses EEG data from eeg_CVD_EGI_83 and eeg_normal_EGI_17 folders
and saves to a single LMDB file.

Source:
    - /projects/EEG-foundation-model/diagnosis_data/eeg_CVD_EGI_83/
    - /projects/EEG-foundation-model/diagnosis_data/eeg_normal_EGI_17/

Output:
    /projects/EEG-foundation-model/diagnosis_data/cvd_normal_preprocessed_CBramod/eeg_segments.lmdb

Reference: preprocessing_depression_normal.py

Data format:
    - EGI .mff format (folders)
    - ~140 channels (EEG + stim)
    - Original sampling rate: 1000 Hz

Preprocessing pipeline:
    1. Read EGI .mff files
    2. Pick EEG channels (exclude stim channels)
    3. Resample to target frequency (200 Hz)
    4. Bandpass filter (0.3-75 Hz)
    5. Notch filter (50 Hz for Chinese data)
    6. Apply average reference
    7. Segment into fixed-length epochs
    8. Save to LMDB

Labels:
    0: CVD (Cardiovascular Disease)
    1: normal

Usage:
    python preprocessing_cvd_normal.py
    python preprocessing_cvd_normal.py --target_sfreq 200 --segment_duration 5
    python preprocessing_cvd_normal.py --num_workers 8
"""

import os
import sys
import pickle
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import lmdb
import numpy as np
from tqdm import tqdm

# Suppress MNE verbose output
mne.set_log_level('ERROR')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Subject ID Extraction
# ============================================================================
def extract_subject_id(foldername, label_name):
    """
    Extract subject ID from EGI .mff folder name.

    Naming patterns:
        CVD:    gma_<subject>_<number>_object_<date>_<time>.mff -> subject
        Normal: nc_<subject>_object_<number>_<date>_<time>.mff -> subject

    Examples:
        gma_bajianping_1_object_20240819_023910.mff -> bajianping
        nc_caichunhong_object_1_20220713_095504.mff -> caichunhong

    Args:
        foldername: The folder name (without path)
        label_name: 'CVD' or 'normal'

    Returns:
        subject_id: Extracted subject identifier
    """
    # Remove .mff extension
    name = foldername.replace('.mff', '')
    parts = name.split('_')

    if label_name == 'normal':
        # Normal files: nc_<subject>_object_...
        # Subject is the 2nd part
        if len(parts) >= 2:
            return parts[1]

    else:  # CVD
        # CVD files: gma_<subject>_<number>_object_...
        # Subject is the 2nd part
        if len(parts) >= 2:
            return parts[1]

    # Fallback: use full foldername as subject
    return name


# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT = Path("/projects/EEG-foundation-model/diagnosis_data")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data/cvd_normal_preprocessed_CBramod")

DATASETS = {
    'CVD': {
        'folder': DATA_ROOT / "eeg_CVD_EGI_83",
        'label': 0,
        'label_name': 'CVD',
    },
    'normal': {
        'folder': DATA_ROOT / "eeg_normal_EGI_17",
        'label': 1,
        'label_name': 'normal',
    },
}

# Preprocessing parameters (matching TUEV preprocessing)
TARGET_SFREQ = 200      # Hz (matching CBraMod's expected input)
LOW_FREQ = 0.3          # Hz
HIGH_FREQ = 75          # Hz
NOTCH_FREQ = 50         # Hz (50 Hz for Chinese data)
SEGMENT_DURATION = 5    # seconds (matching CBraMod's 5-second segments)

# LMDB settings
LMDB_MAP_SIZE = 100 * 1024 * 1024 * 1024  # 100 GB

# Label mapping
LABEL_NAMES = {
    0: 'CVD',
    1: 'normal',
}


# ============================================================================
# EEG Reading and Preprocessing
# ============================================================================
def read_egi_mff(file_path):
    """
    Read EGI .mff format file.

    Args:
        file_path: Path to .mff folder

    Returns:
        raw: MNE Raw object
    """
    return mne.io.read_raw_egi(str(file_path), preload=True, verbose='ERROR')


def preprocess_raw(raw, target_sfreq=TARGET_SFREQ, low_freq=LOW_FREQ,
                   high_freq=HIGH_FREQ, notch_freq=NOTCH_FREQ):
    """
    Preprocess raw EEG data.

    Steps:
        1. Pick EEG channels only (exclude stim channels)
        2. Resample to target frequency
        3. Bandpass filter
        4. Notch filter
        5. Apply average reference

    Args:
        raw: MNE Raw object
        target_sfreq: Target sampling frequency
        low_freq: Low cutoff for bandpass filter
        high_freq: High cutoff for bandpass filter
        notch_freq: Notch filter frequency

    Returns:
        raw: Preprocessed MNE Raw object
        orig_sfreq: Original sampling frequency
    """
    # Make a copy to avoid modifying original
    raw = raw.copy()

    # Get original sampling rate for logging
    orig_sfreq = raw.info['sfreq']

    # Pick only EEG channels (exclude stim and other non-EEG channels)
    raw.pick_types(eeg=True, stim=False, exclude='bads')

    # Resample to target frequency
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad='auto')

    # Bandpass filter
    raw.filter(l_freq=low_freq, h_freq=high_freq, verbose='ERROR')

    # Notch filter for power line noise
    raw.notch_filter(notch_freq, verbose='ERROR')

    # Apply average reference
    raw.set_eeg_reference('average', projection=False)

    return raw, orig_sfreq


def segment_data(raw, segment_duration=SEGMENT_DURATION):
    """
    Segment continuous EEG data into fixed-length epochs.

    Args:
        raw: Preprocessed MNE Raw object
        segment_duration: Duration of each segment in seconds

    Returns:
        segments: list of np.ndarray, each with shape (n_channels, n_samples)
    """
    sfreq = raw.info['sfreq']
    n_samples_per_segment = int(segment_duration * sfreq)
    total_samples = len(raw.times)

    # Get data in microvolts
    data = raw.get_data(units='uV')  # shape: (n_channels, n_samples)

    segments = []
    start = 0
    while start + n_samples_per_segment <= total_samples:
        segment = data[:, start:start + n_samples_per_segment]
        segments.append(segment.astype(np.float32))
        start += n_samples_per_segment

    return segments


# ============================================================================
# Single File Processing (for multiprocessing)
# ============================================================================
def process_single_file(args):
    """
    Process a single EEG file. Designed for multiprocessing.

    Args:
        args: tuple of (file_path, label, label_name, target_sfreq, segment_duration,
                        low_freq, high_freq, notch_freq)

    Returns:
        result: dict with 'samples', 'success', 'error', 'file' keys
    """
    (file_path, label, label_name, target_sfreq, segment_duration,
     low_freq, high_freq, notch_freq) = args

    # Extract subject ID from folder name
    subject_id = extract_subject_id(file_path.name, label_name)

    result = {
        'file': file_path.name,
        'label': label,
        'label_name': label_name,
        'subject_id': subject_id,
        'samples': [],
        'success': False,
        'error': None,
        'orig_sfreq': None,
        'n_channels': None,
    }

    try:
        # Read EGI .mff file
        raw = read_egi_mff(file_path)

        # Preprocess
        raw, orig_sfreq = preprocess_raw(
            raw,
            target_sfreq=target_sfreq,
            low_freq=low_freq,
            high_freq=high_freq,
            notch_freq=notch_freq
        )

        # Get info after preprocessing
        n_channels = len(raw.ch_names)
        ch_names = raw.ch_names

        # Segment the data
        segments = segment_data(raw, segment_duration)

        # Create samples
        samples = []
        for seg_idx, segment in enumerate(segments):
            sample = {
                'signal': segment,  # shape: (n_channels, n_samples)
                'label': label,
                'subject_id': subject_id,  # Add subject ID for cross-subject split
                'source_file': file_path.name,
                'segment_idx': seg_idx,
            }
            samples.append(sample)

        result['samples'] = samples
        result['success'] = True
        result['n_segments'] = len(samples)
        result['orig_sfreq'] = orig_sfreq
        result['n_channels'] = n_channels

        # Close raw to free memory
        raw.close()

    except Exception as e:
        result['error'] = str(e)
        result['success'] = False

    return result


# ============================================================================
# LMDB Writer
# ============================================================================
class LMDBWriter:
    """LMDB writer for CVD/normal data with streaming support."""

    def __init__(self, output_dir, map_size=LMDB_MAP_SIZE, batch_size=500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lmdb_path = self.output_dir / "eeg_segments.lmdb"
        self.batch_size = batch_size

        self.env = lmdb.open(
            str(self.lmdb_path),
            map_size=map_size,
            readonly=False,
            meminit=False,
            map_async=True
        )

        self.count = 0
        self.label_counts = defaultdict(int)
        self.batch_buffer = []

        # Track preprocessing info
        self.n_channels = None
        self.orig_sfreq = None

        # Track subjects for cross-subject split
        self.subjects = set()
        self.subject_label_map = {}  # subject_id -> label
        self.subject_sample_counts = defaultdict(int)  # subject_id -> count

    def add_samples(self, samples, n_channels=None, orig_sfreq=None):
        """Add samples to the buffer and flush if needed."""
        if n_channels is not None and self.n_channels is None:
            self.n_channels = n_channels
        if orig_sfreq is not None and self.orig_sfreq is None:
            self.orig_sfreq = orig_sfreq

        for sample in samples:
            self.batch_buffer.append(sample)
            self.label_counts[sample['label']] += 1

            # Track subject info
            subject_id = sample.get('subject_id', 'unknown')
            self.subjects.add(subject_id)
            self.subject_label_map[subject_id] = sample['label']
            self.subject_sample_counts[subject_id] += 1

            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self):
        """Write buffered samples to LMDB."""
        if not self.batch_buffer:
            return

        with self.env.begin(write=True) as txn:
            for sample in self.batch_buffer:
                key = f'{self.count:08d}'.encode()
                value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key, value)
                self.count += 1

        self.batch_buffer = []

    def close(self, target_sfreq, segment_duration, low_freq, high_freq, notch_freq):
        """Close LMDB and write metadata."""
        # Flush any remaining samples
        self._flush_batch()

        # Get sample shape from first sample
        n_samples_per_segment = int(target_sfreq * segment_duration)

        # Organize subjects by label for cross-subject split
        cvd_subjects = [s for s, l in self.subject_label_map.items() if l == 0]
        normal_subjects = [s for s, l in self.subject_label_map.items() if l == 1]

        # Write metadata
        metadata = {
            'n_samples': self.count,
            'n_channels': self.n_channels,
            'n_samples_per_segment': n_samples_per_segment,
            'sampling_rate': target_sfreq,
            'original_sampling_rate': self.orig_sfreq,
            'segment_duration': segment_duration,
            'label_names': LABEL_NAMES,
            'label_counts': dict(self.label_counts),
            'preprocessing': {
                'low_freq': low_freq,
                'high_freq': high_freq,
                'notch_freq': notch_freq,
                'reference': 'average',
            },
            'created_at': datetime.now().isoformat(),
            # Subject information for cross-subject split
            'subjects': list(self.subjects),
            'n_subjects': len(self.subjects),
            'subject_label_map': self.subject_label_map,
            'subject_sample_counts': dict(self.subject_sample_counts),
            'cvd_subjects': cvd_subjects,
            'normal_subjects': normal_subjects,
            # For compatibility with cross_subject split in finetune script
            'depression_subjects': cvd_subjects,  # Reuse the same key for class 0
        }

        with self.env.begin(write=True) as txn:
            txn.put('__metadata__'.encode(), pickle.dumps(metadata))
            txn.put('__length__'.encode(), pickle.dumps(self.count))

            # Store keys list for iteration
            keys = [f'{i:08d}' for i in range(self.count)]
            txn.put('__keys__'.encode(), pickle.dumps(keys))

        self.env.sync()
        self.env.close()

        # Save statistics
        stats_path = self.output_dir / 'statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("CVD vs Normal EEG Preprocessing Statistics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total samples: {self.count}\n")
            f.write(f"Channels: {self.n_channels}\n")
            f.write(f"Samples per segment: {n_samples_per_segment}\n")
            f.write(f"Target sampling rate: {target_sfreq} Hz\n")
            f.write(f"Original sampling rate: {self.orig_sfreq} Hz\n")
            f.write(f"Segment duration: {segment_duration} seconds\n")
            f.write(f"Bandpass filter: {low_freq}-{high_freq} Hz\n")
            f.write(f"Notch filter: {notch_freq} Hz\n")
            f.write(f"Reference: average\n\n")
            f.write("Label distribution:\n")
            for label_id in sorted(self.label_counts.keys()):
                label_name = LABEL_NAMES.get(label_id, 'unknown')
                count = self.label_counts[label_id]
                pct = 100 * count / self.count if self.count > 0 else 0
                f.write(f"  {label_id} ({label_name}): {count} ({pct:.1f}%)\n")
            f.write(f"\nSubject Information (for cross-subject split):\n")
            f.write(f"  Total subjects: {len(self.subjects)}\n")
            f.write(f"  CVD subjects: {len(cvd_subjects)}\n")
            f.write(f"  Normal subjects: {len(normal_subjects)}\n")
            f.write(f"\n  CVD subjects: {sorted(cvd_subjects)}\n")
            f.write(f"  Normal subjects: {sorted(normal_subjects)}\n")
            f.write(f"\n  Samples per subject:\n")
            for subj in sorted(self.subject_sample_counts.keys()):
                label = self.subject_label_map.get(subj, -1)
                label_name = LABEL_NAMES.get(label, 'unknown')
                f.write(f"    {subj} ({label_name}): {self.subject_sample_counts[subj]}\n")

        logger.info(f"Saved {self.count} samples to {self.lmdb_path}")
        logger.info(f"Total subjects: {len(self.subjects)} (CVD: {len(cvd_subjects)}, normal: {len(normal_subjects)})")
        logger.info(f"Statistics saved to {stats_path}")

        return self.count


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


# ============================================================================
# Main Processing
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CVD vs Normal EEG data (following TUEV preprocessing pipeline)'
    )
    parser.add_argument('--target_sfreq', type=float, default=TARGET_SFREQ,
                        help=f'Target sampling frequency in Hz (default: {TARGET_SFREQ})')
    parser.add_argument('--segment_duration', type=float, default=SEGMENT_DURATION,
                        help=f'Segment duration in seconds (default: {SEGMENT_DURATION})')
    parser.add_argument('--low_freq', type=float, default=LOW_FREQ,
                        help=f'Low cutoff frequency for bandpass filter (default: {LOW_FREQ})')
    parser.add_argument('--high_freq', type=float, default=HIGH_FREQ,
                        help=f'High cutoff frequency for bandpass filter (default: {HIGH_FREQ})')
    parser.add_argument('--notch_freq', type=float, default=NOTCH_FREQ,
                        help=f'Notch filter frequency (default: {NOTCH_FREQ})')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for LMDB')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes (default: 4)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for LMDB writing (default: 500)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CVD vs Normal EEG Preprocessing")
    logger.info("(Following TUEV preprocessing pipeline)")
    logger.info("=" * 60)
    logger.info(f"Target sampling rate: {args.target_sfreq} Hz")
    logger.info(f"Bandpass filter: {args.low_freq}-{args.high_freq} Hz")
    logger.info(f"Notch filter: {args.notch_freq} Hz")
    logger.info(f"Segment duration: {args.segment_duration} seconds")
    logger.info(f"Reference: average")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of workers: {args.num_workers}")

    # Collect all file tasks
    all_tasks = []
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info['folder']
        label = dataset_info['label']
        label_name = dataset_info['label_name']

        if not folder.exists():
            logger.error(f"Folder not found: {folder}")
            continue

        # Find all .mff folders
        mff_folders = sorted([f for f in folder.iterdir() if f.is_dir() and f.suffix == '.mff'])
        logger.info(f"Found {len(mff_folders)} .mff folders in {dataset_name}")

        for file_path in mff_folders:
            all_tasks.append((
                file_path, label, label_name,
                args.target_sfreq, args.segment_duration,
                args.low_freq, args.high_freq, args.notch_freq
            ))

    logger.info(f"Total files to process: {len(all_tasks)}")

    if len(all_tasks) == 0:
        logger.error("No files to process!")
        return

    # Shuffle tasks for better label distribution
    np.random.seed(42)
    np.random.shuffle(all_tasks)

    # Initialize LMDB writer
    output_path = Path(args.output_dir)
    writer = LMDBWriter(output_path, batch_size=args.batch_size)

    # Statistics
    file_stats = defaultdict(lambda: {'files': 0, 'segments': 0, 'errors': 0})
    error_files = []

    # Process files with multiprocessing
    logger.info(f"\nProcessing with {args.num_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, task): task for task in all_tasks}

        # Process results as they complete
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                label_name = result['label_name']

                if result['success']:
                    # Stream samples to LMDB
                    writer.add_samples(
                        result['samples'],
                        n_channels=result['n_channels'],
                        orig_sfreq=result['orig_sfreq']
                    )
                    file_stats[label_name]['files'] += 1
                    file_stats[label_name]['segments'] += len(result['samples'])

                    # Clear samples to free memory
                    result['samples'] = None
                else:
                    file_stats[label_name]['errors'] += 1
                    error_files.append((result['file'], result['error']))
                    logger.debug(f"Failed: {result['file']}: {result['error']}")

                pbar.update(1)

    # Finalize LMDB
    total_samples = writer.close(
        args.target_sfreq, args.segment_duration,
        args.low_freq, args.high_freq, args.notch_freq
    )

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Processing Statistics")
    logger.info("=" * 60)
    for label_name, stats in file_stats.items():
        logger.info(f"{label_name}:")
        logger.info(f"  Files processed: {stats['files']}")
        logger.info(f"  Segments created: {stats['segments']}")
        logger.info(f"  Errors: {stats['errors']}")

    if error_files:
        logger.warning(f"\nErrors ({len(error_files)} files):")
        # Save error log
        error_log_path = output_path / 'failed_files.txt'
        with open(error_log_path, 'w') as f:
            for fname, err in error_files:
                f.write(f"{fname}: {err}\n")
        logger.info(f"Error log saved to {error_log_path}")

        # Print first few errors
        for fname, err in error_files[:5]:
            logger.warning(f"  {fname}: {err}")
        if len(error_files) > 5:
            logger.warning(f"  ... and {len(error_files) - 5} more errors")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Output: {output_path / 'eeg_segments.lmdb'}")


def verify_output(output_dir=OUTPUT_DIR):
    """Verify the output LMDB file."""
    logger.info("\n" + "=" * 60)
    logger.info("Verifying Output")
    logger.info("=" * 60)

    lmdb_path = Path(output_dir) / "eeg_segments.lmdb"
    if lmdb_path.exists():
        reader = LMDBReader(lmdb_path)
        logger.info(f"LMDB path: {lmdb_path}")
        logger.info(f"Total samples: {len(reader)}")
        logger.info(f"Metadata: {reader.metadata}")

        # Read a few samples
        if len(reader) > 0:
            for i in [0, len(reader) // 2, len(reader) - 1]:
                sample = reader[i]
                logger.info(f"\nSample {i}:")
                logger.info(f"  Signal shape: {sample['signal'].shape}")
                logger.info(f"  Label: {sample['label']} ({LABEL_NAMES[sample['label']]})")
                logger.info(f"  Subject ID: {sample.get('subject_id', 'N/A')}")
                logger.info(f"  Source file: {sample['source_file']}")
                logger.info(f"  Segment idx: {sample['segment_idx']}")

        reader.close()
    else:
        logger.error(f"LMDB not found: {lmdb_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        # Just verify existing output
        output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
        verify_output(output_dir)
    else:
        main()
        verify_output()
