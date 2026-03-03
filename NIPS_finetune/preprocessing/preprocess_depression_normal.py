#!/usr/bin/env python3
"""
Preprocess EEG data for Depression vs Normal classification.

This script:
1. Reads EEG files from eeg_normal_BP_166 and eeg_depression_BP_122 folders
2. Applies only re-referencing (average reference) - no downsampling or channel selection
3. Segments data into 2-second epochs
4. Saves all segments to a single LMDB file

Usage:
    python preprocess_depression_normal.py
    python preprocess_depression_normal.py --num_workers 8
    python preprocess_depression_normal.py --segment_duration 3.0

Output:
    /projects/EEG-foundation-model/diagnosis_data_lmdb/depression_normal_original/eeg_segments.lmdb
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import lmdb
import numpy as np
import mne
from tqdm import tqdm

# Add the read_eeg module
sys.path.insert(0, os.path.dirname(__file__))
from read_eeg import read_eeg

# Suppress MNE verbose output
mne.set_log_level('ERROR')


# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT = Path("/projects/EEG-foundation-model/diagnosis_data")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data_lmdb/depression_normal_original")

DATASETS = {
    'normal': {
        'folder': DATA_ROOT / "eeg_normal_BP_166",
        'label': 'normal',
    },
    'depression': {
        'folder': DATA_ROOT / "eeg_depression_BP_122",
        'label': 'depression',
    },
}

SEGMENT_DURATION = 2.0  # seconds


# ============================================================================
# Preprocessing Functions
# ============================================================================
def preprocess_raw(raw):
    """
    Apply minimal preprocessing: only re-referencing (average reference).
    No downsampling, no channel selection.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data
    """
    # Make a copy to avoid modifying original
    raw = raw.copy()

    # Pick only EEG channels (exclude non-EEG like ECG, EOG, etc. if present)
    raw.pick_types(eeg=True, exclude='bads')

    # Apply average reference
    raw.set_eeg_reference('average', projection=False)

    return raw


def segment_data(raw, segment_duration=2.0):
    """
    Segment continuous EEG data into fixed-length epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw EEG data
    segment_duration : float
        Duration of each segment in seconds

    Returns
    -------
    segments : list of np.ndarray
        List of segments, each with shape (n_channels, n_samples)
    """
    sfreq = raw.info['sfreq']
    n_samples_per_segment = int(segment_duration * sfreq)
    total_samples = len(raw.times)

    # Get data
    data = raw.get_data()  # shape: (n_channels, n_samples)

    segments = []
    start = 0
    while start + n_samples_per_segment <= total_samples:
        segment = data[:, start:start + n_samples_per_segment]
        segments.append(segment.astype(np.float32))
        start += n_samples_per_segment

    return segments


def process_single_file(args):
    """
    Process a single EEG file. This function is designed for multiprocessing.

    Parameters
    ----------
    args : tuple
        (file_path, label, segment_duration)

    Returns
    -------
    result : dict
        Dictionary with 'samples', 'success', 'error', 'file' keys
    """
    file_path, label, segment_duration = args

    result = {
        'file': file_path.name,
        'label': label,
        'samples': [],
        'success': False,
        'error': None,
    }

    try:
        # Read EEG file
        raw = read_eeg(file_path)

        # Preprocess (only re-referencing)
        raw = preprocess_raw(raw)

        # Get info after preprocessing
        n_channels = len(raw.ch_names)
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names

        # Segment the data
        segments = segment_data(raw, segment_duration)

        # Create samples
        samples = []
        for seg_idx, segment in enumerate(segments):
            sample = {
                'data': segment,  # shape: (n_channels, n_samples)
                'labels': {
                    'disease': label,
                },
                'metadata': {
                    'file': file_path.name,
                    'segment_idx': seg_idx,
                    'n_channels': n_channels,
                    'sfreq': sfreq,
                    'ch_names': list(ch_names),
                    'segment_duration': segment_duration,
                }
            }
            samples.append(sample)

        result['samples'] = samples
        result['success'] = True
        result['n_segments'] = len(samples)

    except Exception as e:
        result['error'] = str(e)
        result['success'] = False

    return result


# ============================================================================
# LMDB Writer
# ============================================================================
def save_to_lmdb_streaming(output_path, segment_duration, map_size=100 * 1024**3):
    """
    Create a streaming LMDB writer that writes samples incrementally.

    Parameters
    ----------
    output_path : Path
        Output LMDB directory path
    segment_duration : float
        Segment duration for metadata
    map_size : int
        Maximum size of the database in bytes (default: 100GB)

    Returns
    -------
    writer : LMDBStreamWriter
        Streaming writer object
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    lmdb_path = output_path / "eeg_segments.lmdb"

    return LMDBStreamWriter(lmdb_path, segment_duration, map_size)


class LMDBStreamWriter:
    """Streaming LMDB writer to avoid OOM by writing in batches."""

    def __init__(self, lmdb_path, segment_duration, map_size=100 * 1024**3, batch_size=1000):
        self.lmdb_path = Path(lmdb_path)
        self.segment_duration = segment_duration
        self.map_size = map_size
        self.batch_size = batch_size

        self.env = lmdb.open(
            str(self.lmdb_path),
            map_size=map_size,
            readonly=False,
            meminit=False,
            map_async=True,
        )

        self.current_idx = 0
        self.label_counts = defaultdict(int)
        self.batch_buffer = []

    def add_samples(self, samples):
        """Add samples to the buffer and flush if needed."""
        for sample in samples:
            self.batch_buffer.append(sample)
            self.label_counts[sample['labels']['disease']] += 1

            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self):
        """Write buffered samples to LMDB."""
        if not self.batch_buffer:
            return

        with self.env.begin(write=True) as txn:
            for sample in self.batch_buffer:
                key = f'{self.current_idx:08d}'.encode()
                value = pickle.dumps(sample)
                txn.put(key, value)
                self.current_idx += 1

        self.batch_buffer = []

    def finalize(self):
        """Flush remaining samples and write metadata."""
        # Flush any remaining samples
        self._flush_batch()

        # Write metadata
        metadata = {
            'total_samples': self.current_idx,
            'label_counts': dict(self.label_counts),
            'labels': list(self.label_counts.keys()),
            'segment_duration': self.segment_duration,
        }

        with self.env.begin(write=True) as txn:
            txn.put('__length__'.encode(), pickle.dumps(self.current_idx))
            txn.put('__metadata__'.encode(), pickle.dumps(metadata))

        self.env.close()

        print(f"\nLMDB saved to: {self.lmdb_path}")
        print(f"Total samples: {self.current_idx}")
        print(f"Label distribution: {dict(self.label_counts)}")

        return self.lmdb_path


def save_to_lmdb(all_samples, output_path, map_size=100 * 1024**3, batch_size=500):
    """
    Save all samples to LMDB file with batch writing to avoid OOM.

    Parameters
    ----------
    all_samples : list of dict
        All samples to save
    output_path : Path
        Output LMDB directory path
    map_size : int
        Maximum size of the database in bytes (default: 100GB)
    batch_size : int
        Number of samples to write per transaction (default: 500)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    lmdb_path = output_path / "eeg_segments.lmdb"

    print(f"\nSaving {len(all_samples)} samples to {lmdb_path}")
    print(f"Using batch size: {batch_size}")

    # Count labels
    label_counts = defaultdict(int)
    for sample in all_samples:
        label_counts[sample['labels']['disease']] += 1

    print(f"Label distribution: {dict(label_counts)}")

    # Create LMDB environment
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    # Write samples in batches
    total_samples = len(all_samples)
    pbar = tqdm(total=total_samples, desc="Writing to LMDB")

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch = all_samples[batch_start:batch_end]

        with env.begin(write=True) as txn:
            for i, sample in enumerate(batch):
                idx = batch_start + i
                key = f'{idx:08d}'.encode()
                value = pickle.dumps(sample)
                txn.put(key, value)

        pbar.update(len(batch))

        # Clear batch from memory
        del batch

    pbar.close()

    # Write metadata in separate transaction
    metadata = {
        'total_samples': total_samples,
        'label_counts': dict(label_counts),
        'labels': list(label_counts.keys()),
        'segment_duration': SEGMENT_DURATION,
    }

    with env.begin(write=True) as txn:
        txn.put('__length__'.encode(), pickle.dumps(total_samples))
        txn.put('__metadata__'.encode(), pickle.dumps(metadata))

    env.sync()
    env.close()

    print(f"LMDB saved to: {lmdb_path}")

    return lmdb_path


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Preprocess Depression vs Normal EEG data')
    parser.add_argument('--segment_duration', type=float, default=2.0,
                        help='Segment duration in seconds (default: 2.0)')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for LMDB')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count - 2)')
    parser.add_argument('--streaming', action='store_true', default=True,
                        help='Use streaming mode to avoid OOM (default: True)')
    parser.add_argument('--no_streaming', action='store_false', dest='streaming',
                        help='Disable streaming mode (load all samples in memory)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for LMDB writing (default: 500)')
    args = parser.parse_args()

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = max(1, mp.cpu_count() - 2)

    print("=" * 60)
    print("Depression vs Normal EEG Preprocessing")
    print("=" * 60)
    print(f"Segment duration: {args.segment_duration} seconds")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Streaming mode: {args.streaming}")
    print(f"Batch size: {args.batch_size}")
    print("Preprocessing: Re-referencing (average) only")
    print("No downsampling, no channel selection")
    print("=" * 60)

    # Collect all file tasks
    all_tasks = []
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info['folder']
        label = dataset_info['label']

        # Find all .vhdr files
        vhdr_files = sorted(folder.glob("*.vhdr"))
        print(f"\nFound {len(vhdr_files)} files in {dataset_name}")

        for file_path in vhdr_files:
            all_tasks.append((file_path, label, args.segment_duration))

    print(f"\nTotal files to process: {len(all_tasks)}")

    # Shuffle tasks for better label distribution during streaming
    np.random.seed(42)
    np.random.shuffle(all_tasks)

    file_stats = defaultdict(lambda: {'files': 0, 'segments': 0, 'errors': 0})
    error_files = []
    sample_info = None

    if args.streaming:
        # Streaming mode: write directly to LMDB as we process
        print(f"\nProcessing with {args.num_workers} workers (streaming mode)...")

        output_path = Path(args.output_dir)
        lmdb_writer = save_to_lmdb_streaming(output_path, args.segment_duration)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, task): task for task in all_tasks}

            # Process results as they complete and stream to LMDB
            with tqdm(total=len(futures), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    label = result['label']

                    if result['success']:
                        # Stream samples directly to LMDB
                        lmdb_writer.add_samples(result['samples'])
                        file_stats[label]['files'] += 1
                        file_stats[label]['segments'] += len(result['samples'])

                        # Save first sample info for display
                        if sample_info is None and result['samples']:
                            sample_info = result['samples'][0]

                        # Clear samples from result to free memory
                        result['samples'] = None
                    else:
                        file_stats[label]['errors'] += 1
                        error_files.append((result['file'], result['error']))

                    pbar.update(1)

        # Finalize LMDB
        lmdb_writer.finalize()

    else:
        # Non-streaming mode: collect all samples then write
        print(f"\nProcessing with {args.num_workers} workers (non-streaming mode)...")

        all_samples = []

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, task): task for task in all_tasks}

            # Process results as they complete
            with tqdm(total=len(futures), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    label = result['label']

                    if result['success']:
                        all_samples.extend(result['samples'])
                        file_stats[label]['files'] += 1
                        file_stats[label]['segments'] += len(result['samples'])
                    else:
                        file_stats[label]['errors'] += 1
                        error_files.append((result['file'], result['error']))

                    pbar.update(1)

        print(f"\nTotal segments: {len(all_samples)}")

        # Shuffle samples
        print("Shuffling samples...")
        np.random.shuffle(all_samples)

        # Save first sample info for display
        if all_samples:
            sample_info = all_samples[0]

        # Save to LMDB
        output_path = Path(args.output_dir)
        save_to_lmdb(all_samples, output_path, batch_size=args.batch_size)

        # Clear memory
        del all_samples

    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Statistics")
    print("=" * 60)
    for label, stats in file_stats.items():
        print(f"{label}:")
        print(f"  Files processed: {stats['files']}")
        print(f"  Segments created: {stats['segments']}")
        print(f"  Errors: {stats['errors']}")

    if error_files:
        print(f"\nErrors ({len(error_files)} files):")
        for fname, err in error_files[:10]:  # Show first 10 errors
            print(f"  {fname}: {err}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more errors")

    # Print sample info
    if sample_info is not None:
        print("\n" + "=" * 60)
        print("Sample Info")
        print("=" * 60)
        print(f"Data shape: {sample_info['data'].shape}")
        print(f"Label: {sample_info['labels']['disease']}")
        print(f"Sampling rate: {sample_info['metadata']['sfreq']} Hz")
        print(f"Number of channels: {sample_info['metadata']['n_channels']}")
        print(f"Segment duration: {sample_info['metadata']['segment_duration']} s")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
