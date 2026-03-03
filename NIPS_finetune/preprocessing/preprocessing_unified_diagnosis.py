#!/usr/bin/env python3
"""
Unified EEG Dataset Preprocessing Script for Multi-class Diagnosis

Preprocesses EEG data from multiple sources (EGI and BrainVision formats)
and unifies them to a common channel set and sampling frequency.

Source datasets:
    - eeg_CVD_EGI_83:        EGI .mff format, 129 EEG channels, 1000 Hz
    - eeg_depression_EGI_21: EGI .mff format, 129 EEG channels, 500 Hz
    - eeg_normal_EGI_17:     EGI .mff format, 129 EEG channels, 500 Hz
    - eeg_depression_BP_122: BrainVision format, 64 channels, 5000 Hz
    - eeg_normal_BP_166:     BrainVision format, 64 channels, 5000 Hz

Output:
    /projects/EEG-foundation-model/diagnosis_data/unified_diagnosis_preprocessed/eeg_segments.lmdb

Channel unification strategy:
    - EGI uses E1-E128 naming, BP uses 10-20 standard naming
    - We map EGI channels to 10-20 equivalents using spatial position matching
    - Final output uses 58 common channels in 10-20 naming

Preprocessing pipeline:
    1. Read EEG files (EGI .mff or BrainVision .vhdr)
    2. Pick common channels (58 channels)
    3. Rename EGI channels to 10-20 names
    4. Resample to target frequency (200 Hz)
    5. Bandpass filter (0.3-75 Hz)
    6. Notch filter (50 Hz for Chinese data)
    7. Apply average reference
    8. Segment into fixed-length epochs
    9. Save to LMDB

Labels (3-class):
    0: CVD (Cardiovascular Disease)
    1: depression
    2: normal

Usage:
    python preprocessing_unified_diagnosis.py
    python preprocessing_unified_diagnosis.py --target_sfreq 200 --segment_duration 5
    python preprocessing_unified_diagnosis.py --num_workers 8
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
# Channel Mapping: EGI GSN-HydroCel-129 to 10-20 Standard
# ============================================================================
# This mapping was created by finding the nearest EGI electrode for each
# 10-20 standard position using MNE's montage coordinates

EGI_TO_1020_MAPPING = {
    'E23': 'AF3',
    'E3': 'AF4',
    'E26': 'AF7',
    'E2': 'AF8',
    'E37': 'C1',
    'E87': 'C2',
    'E42': 'C3',
    'E93': 'C4',
    'E46': 'C5',
    'E102': 'C6',
    'E54': 'CP1',
    'E86': 'CP2',
    'E52': 'CP3',
    'E92': 'CP4',
    'E51': 'CP5',
    'E97': 'CP6',
    'E55': 'CPz',
    'Cz': 'Cz',  # Reference channel
    'E20': 'F1',
    'E118': 'F2',
    'E24': 'F3',
    'E124': 'F4',
    'E27': 'F5',
    'E123': 'F6',
    'E33': 'F7',
    'E122': 'F8',
    'E13': 'FC1',
    'E112': 'FC2',
    'E35': 'FC3',
    'E110': 'FC4',
    'E34': 'FC5',
    'E116': 'FC6',
    'E39': 'FT7',
    'E115': 'FT8',
    'E25': 'Fp1',
    'E8': 'Fp2',
    'E5': 'Fz',
    'E70': 'O1',
    'E83': 'O2',
    'E75': 'Oz',
    'E67': 'P1',
    'E78': 'P2',
    'E60': 'P3',
    'E85': 'P4',
    'E59': 'P5',
    'E91': 'P6',
    'E58': 'P7',
    'E96': 'P8',
    'E66': 'PO3',
    'E84': 'PO4',
    'E65': 'PO7',
    'E90': 'PO8',
    'E72': 'POz',
    'E62': 'Pz',
    'E45': 'T7',
    'E108': 'T8',
    'E50': 'TP7',
    'E101': 'TP8',
}

# Reverse mapping: 10-20 name -> EGI name
MAPPING_1020_TO_EGI = {v: k for k, v in EGI_TO_1020_MAPPING.items()}

# Common channels in 10-20 naming (58 channels)
COMMON_CHANNELS_1020 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'FT7', 'FT8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CPz', 'CP2', 'CP6', 'TP7', 'TP8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'AF7', 'AF3', 'AF4', 'AF8',
    'F5', 'F1', 'F2', 'F6',
    'FC3', 'FC4',
    'C5', 'C1', 'C2', 'C6',
    'CP3', 'CP4',
    'P5', 'P1', 'P2', 'P6',
]

# EGI channels to pick (corresponding to COMMON_CHANNELS_1020)
COMMON_CHANNELS_EGI = [MAPPING_1020_TO_EGI.get(ch, ch) for ch in COMMON_CHANNELS_1020]

# BP channels to pick (same as 1020, excluding ECG)
COMMON_CHANNELS_BP = COMMON_CHANNELS_1020.copy()


# ============================================================================
# Subject ID Extraction
# ============================================================================
def extract_subject_id(filename, label_name, system):
    """
    Extract subject ID from filename.

    Naming patterns:
        EGI CVD:        gma_<subject>_<number>_object_<date>_<time>.mff -> subject
        EGI Normal:     nc_<subject>_object_<number>_<date>_<time>.mff -> subject
        EGI Depression: <number><name>-<type>_<date>_<time>.mff -> number+name
        BP Depression:  ccs_before_<subject>_<date>_<state>.vhdr -> subject
                        njh_after_<subject>_<date>_<state>.vhdr -> subject
                        hc_<subject>_<date>_<state>.vhdr -> subject
        BP Normal:      hc_<subject>_<date>_<state>.vhdr -> subject
                        jkdz_<subject>_<date>_<state>.vhdr -> subject

    Args:
        filename: The file name (without path)
        label_name: 'CVD', 'normal', or 'depression'
        system: 'EGI' or 'BP'

    Returns:
        subject_id: Extracted subject identifier
    """
    if system == 'EGI':
        # Remove .mff extension
        name = filename.replace('.mff', '')
        parts = name.split('_')

        if label_name == 'normal':
            # Normal files: nc_<subject>_object_...
            if len(parts) >= 2:
                return parts[1]
        elif label_name == 'CVD':
            # CVD files: gma_<subject>_<number>_object_...
            if len(parts) >= 2:
                return parts[1]
        elif label_name == 'depression':
            # Depression files: <number><name>-<type>_<date>_<time>.mff
            if '-' in name:
                return name.split('-')[0]
            elif len(parts) >= 1:
                return parts[0]

    elif system == 'BP':
        # Remove .vhdr extension
        name = filename.replace('.vhdr', '')
        parts = name.split('_')

        # Common pattern: <prefix>_<subject>_<date>_<state>
        # or: <prefix>_<action>_<subject>_<date>_<state>
        date_pattern = re.compile(r'^\d{8}$')

        for i, part in enumerate(parts):
            if date_pattern.match(part):
                # Found date, subject is just before it
                if i > 0:
                    candidate = parts[i - 1]
                    # Skip keywords that are not subject IDs
                    if candidate not in ['close', 'open', 'before', 'after', 'rest']:
                        return candidate
                    elif i > 1:
                        return parts[i - 2]
                break

        # Fallback: second part is often subject
        if len(parts) >= 2:
            return parts[1]

    # Fallback: use full filename
    return filename


# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT = Path("/projects/EEG-foundation-model/diagnosis_data")
OUTPUT_DIR = Path("/projects/EEG-foundation-model/diagnosis_data/unified_diagnosis_preprocessed")

DATASETS = {
    'CVD_EGI': {
        'folder': DATA_ROOT / "eeg_CVD_EGI_83",
        'label': 0,
        'label_name': 'CVD',
        'system': 'EGI',
        'file_pattern': '.mff',
    },
    'depression_EGI': {
        'folder': DATA_ROOT / "eeg_depression_EGI_21",
        'label': 1,
        'label_name': 'depression',
        'system': 'EGI',
        'file_pattern': '.mff',
    },
    'depression_BP': {
        'folder': DATA_ROOT / "eeg_depression_BP_122",
        'label': 1,
        'label_name': 'depression',
        'system': 'BP',
        'file_pattern': '.vhdr',
    },
    'normal_EGI': {
        'folder': DATA_ROOT / "eeg_normal_EGI_17",
        'label': 2,
        'label_name': 'normal',
        'system': 'EGI',
        'file_pattern': '.mff',
    },
    'normal_BP': {
        'folder': DATA_ROOT / "eeg_normal_BP_166",
        'label': 2,
        'label_name': 'normal',
        'system': 'BP',
        'file_pattern': '.vhdr',
    },
}

# Preprocessing parameters
TARGET_SFREQ = 200      # Hz (matching CBraMod's expected input)
LOW_FREQ = 0.3          # Hz
HIGH_FREQ = 75          # Hz
NOTCH_FREQ = 50         # Hz (50 Hz for Chinese data)
SEGMENT_DURATION = 5    # seconds

# LMDB settings
LMDB_MAP_SIZE = 200 * 1024 * 1024 * 1024  # 200 GB

# Label mapping
LABEL_NAMES = {
    0: 'CVD',
    1: 'depression',
    2: 'normal',
}

# Number of common channels
N_COMMON_CHANNELS = len(COMMON_CHANNELS_1020)  # 58 channels


# ============================================================================
# EEG Reading Functions
# ============================================================================
def read_egi_mff(file_path):
    """Read EGI .mff format file."""
    return mne.io.read_raw_egi(str(file_path), preload=True, verbose='ERROR')


def read_brainvision(file_path):
    """Read BrainVision .vhdr format file."""
    return mne.io.read_raw_brainvision(str(file_path), preload=True, verbose='ERROR')


# ============================================================================
# Preprocessing Functions
# ============================================================================
def preprocess_egi(raw, target_sfreq=TARGET_SFREQ, low_freq=LOW_FREQ,
                   high_freq=HIGH_FREQ, notch_freq=NOTCH_FREQ):
    """
    Preprocess EGI EEG data.

    Steps:
        1. Pick common EGI channels
        2. Rename to 10-20 standard names
        3. Resample to target frequency
        4. Bandpass filter
        5. Notch filter
        6. Apply average reference
    """
    raw = raw.copy()
    orig_sfreq = raw.info['sfreq']

    # Pick only EEG channels first
    raw.pick_types(eeg=True, stim=False, exclude='bads')

    # Find available common channels
    available_chs = raw.ch_names
    pick_chs = [ch for ch in COMMON_CHANNELS_EGI if ch in available_chs]

    if len(pick_chs) < N_COMMON_CHANNELS:
        missing = set(COMMON_CHANNELS_EGI) - set(pick_chs)
        logger.debug(f"Missing EGI channels: {missing}")

    # Pick common channels
    raw.pick_channels(pick_chs)

    # Rename EGI channels to 10-20 names
    rename_dict = {ch: EGI_TO_1020_MAPPING[ch] for ch in pick_chs if ch in EGI_TO_1020_MAPPING}
    raw.rename_channels(rename_dict)

    # Reorder channels to standard order
    ch_order = [ch for ch in COMMON_CHANNELS_1020 if ch in raw.ch_names]
    raw.reorder_channels(ch_order)

    # Resample
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad='auto')

    # Bandpass filter
    raw.filter(l_freq=low_freq, h_freq=high_freq, verbose='ERROR')

    # Notch filter
    raw.notch_filter(notch_freq, verbose='ERROR')

    # Average reference
    raw.set_eeg_reference('average', projection=False)

    return raw, orig_sfreq


def preprocess_bp(raw, target_sfreq=TARGET_SFREQ, low_freq=LOW_FREQ,
                  high_freq=HIGH_FREQ, notch_freq=NOTCH_FREQ):
    """
    Preprocess BrainVision EEG data.

    Steps:
        1. Exclude ECG channel
        2. Pick common channels
        3. Resample to target frequency
        4. Bandpass filter
        5. Notch filter
        6. Apply average reference
    """
    raw = raw.copy()
    orig_sfreq = raw.info['sfreq']

    # Exclude ECG channel
    if 'ECG' in raw.ch_names:
        raw.drop_channels(['ECG'])

    # Find available common channels
    available_chs = raw.ch_names
    pick_chs = [ch for ch in COMMON_CHANNELS_BP if ch in available_chs]

    if len(pick_chs) < N_COMMON_CHANNELS:
        missing = set(COMMON_CHANNELS_BP) - set(pick_chs)
        logger.debug(f"Missing BP channels: {missing}")

    # Pick common channels
    raw.pick_channels(pick_chs)

    # Reorder channels to standard order
    ch_order = [ch for ch in COMMON_CHANNELS_1020 if ch in raw.ch_names]
    raw.reorder_channels(ch_order)

    # Resample (BP is 5000 Hz, need to downsample significantly)
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad='auto')

    # Bandpass filter
    raw.filter(l_freq=low_freq, h_freq=high_freq, verbose='ERROR')

    # Notch filter
    raw.notch_filter(notch_freq, verbose='ERROR')

    # Average reference
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
    data = raw.get_data(units='uV')

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
    """
    (file_path, label, label_name, system, target_sfreq, segment_duration,
     low_freq, high_freq, notch_freq) = args

    # Extract subject ID
    subject_id = extract_subject_id(file_path.name, label_name, system)

    result = {
        'file': file_path.name,
        'label': label,
        'label_name': label_name,
        'system': system,
        'subject_id': subject_id,
        'samples': [],
        'success': False,
        'error': None,
        'orig_sfreq': None,
        'n_channels': None,
    }

    try:
        # Read file based on system
        if system == 'EGI':
            raw = read_egi_mff(file_path)
            raw, orig_sfreq = preprocess_egi(
                raw, target_sfreq, low_freq, high_freq, notch_freq
            )
        else:  # BP
            raw = read_brainvision(file_path)
            raw, orig_sfreq = preprocess_bp(
                raw, target_sfreq, low_freq, high_freq, notch_freq
            )

        # Get info
        n_channels = len(raw.ch_names)
        ch_names = raw.ch_names

        # Segment the data
        segments = segment_data(raw, segment_duration)

        # Create samples
        samples = []
        for seg_idx, segment in enumerate(segments):
            sample = {
                'signal': segment,
                'label': label,
                'subject_id': subject_id,
                'source_file': file_path.name,
                'segment_idx': seg_idx,
                'system': system,
            }
            samples.append(sample)

        result['samples'] = samples
        result['success'] = True
        result['n_segments'] = len(samples)
        result['orig_sfreq'] = orig_sfreq
        result['n_channels'] = n_channels
        result['ch_names'] = ch_names

        raw.close()

    except Exception as e:
        result['error'] = str(e)
        result['success'] = False

    return result


# ============================================================================
# LMDB Writer
# ============================================================================
class LMDBWriter:
    """LMDB writer for unified diagnosis data."""

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

        self.n_channels = None
        self.orig_sfreq = None
        self.ch_names = None

        # Track subjects
        self.subjects = set()
        self.subject_label_map = {}
        self.subject_sample_counts = defaultdict(int)
        self.subject_system_map = {}  # Track which system each subject is from

    def add_samples(self, samples, n_channels=None, orig_sfreq=None, ch_names=None):
        """Add samples to the buffer and flush if needed."""
        if n_channels is not None and self.n_channels is None:
            self.n_channels = n_channels
        if orig_sfreq is not None and self.orig_sfreq is None:
            self.orig_sfreq = orig_sfreq
        if ch_names is not None and self.ch_names is None:
            self.ch_names = ch_names

        for sample in samples:
            self.batch_buffer.append(sample)
            self.label_counts[sample['label']] += 1

            subject_id = sample.get('subject_id', 'unknown')
            self.subjects.add(subject_id)
            self.subject_label_map[subject_id] = sample['label']
            self.subject_sample_counts[subject_id] += 1
            self.subject_system_map[subject_id] = sample.get('system', 'unknown')

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
        self._flush_batch()

        n_samples_per_segment = int(target_sfreq * segment_duration)

        # Organize subjects by label
        cvd_subjects = [s for s, l in self.subject_label_map.items() if l == 0]
        depression_subjects = [s for s, l in self.subject_label_map.items() if l == 1]
        normal_subjects = [s for s, l in self.subject_label_map.items() if l == 2]

        metadata = {
            'n_samples': self.count,
            'n_channels': self.n_channels,
            'n_samples_per_segment': n_samples_per_segment,
            'sampling_rate': target_sfreq,
            'original_sampling_rate': 'varies',  # Different sources have different rates
            'segment_duration': segment_duration,
            'label_names': LABEL_NAMES,
            'label_counts': dict(self.label_counts),
            'n_classes': 3,
            'task_type': 'multiclass',
            'channel_names': self.ch_names,
            'preprocessing': {
                'low_freq': low_freq,
                'high_freq': high_freq,
                'notch_freq': notch_freq,
                'reference': 'average',
            },
            'created_at': datetime.now().isoformat(),
            # Subject information
            'subjects': list(self.subjects),
            'n_subjects': len(self.subjects),
            'subject_label_map': self.subject_label_map,
            'subject_sample_counts': dict(self.subject_sample_counts),
            'subject_system_map': self.subject_system_map,
            'cvd_subjects': cvd_subjects,
            'depression_subjects': depression_subjects,
            'normal_subjects': normal_subjects,
        }

        with self.env.begin(write=True) as txn:
            txn.put('__metadata__'.encode(), pickle.dumps(metadata))
            txn.put('__length__'.encode(), pickle.dumps(self.count))
            keys = [f'{i:08d}' for i in range(self.count)]
            txn.put('__keys__'.encode(), pickle.dumps(keys))

        self.env.sync()
        self.env.close()

        # Save statistics
        stats_path = self.output_dir / 'statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("Unified Diagnosis EEG Preprocessing Statistics\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total samples: {self.count}\n")
            f.write(f"Channels: {self.n_channels}\n")
            f.write(f"Channel names: {self.ch_names}\n")
            f.write(f"Samples per segment: {n_samples_per_segment}\n")
            f.write(f"Target sampling rate: {target_sfreq} Hz\n")
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
            f.write(f"  Depression subjects: {len(depression_subjects)}\n")
            f.write(f"  Normal subjects: {len(normal_subjects)}\n")

            f.write(f"\n  CVD subjects: {sorted(cvd_subjects)}\n")
            f.write(f"  Depression subjects: {sorted(depression_subjects)}\n")
            f.write(f"  Normal subjects: {sorted(normal_subjects)}\n")

            f.write(f"\n  Samples per subject:\n")
            for subj in sorted(self.subject_sample_counts.keys()):
                label = self.subject_label_map.get(subj, -1)
                label_name = LABEL_NAMES.get(label, 'unknown')
                system = self.subject_system_map.get(subj, 'unknown')
                f.write(f"    {subj} ({label_name}, {system}): {self.subject_sample_counts[subj]}\n")

        logger.info(f"Saved {self.count} samples to {self.lmdb_path}")
        logger.info(f"Total subjects: {len(self.subjects)} (CVD: {len(cvd_subjects)}, "
                   f"depression: {len(depression_subjects)}, normal: {len(normal_subjects)})")
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
        description='Unified preprocessing for multi-source EEG diagnosis data'
    )
    parser.add_argument('--target_sfreq', type=float, default=TARGET_SFREQ,
                        help=f'Target sampling frequency (default: {TARGET_SFREQ})')
    parser.add_argument('--segment_duration', type=float, default=SEGMENT_DURATION,
                        help=f'Segment duration in seconds (default: {SEGMENT_DURATION})')
    parser.add_argument('--low_freq', type=float, default=LOW_FREQ,
                        help=f'Low cutoff for bandpass filter (default: {LOW_FREQ})')
    parser.add_argument('--high_freq', type=float, default=HIGH_FREQ,
                        help=f'High cutoff for bandpass filter (default: {HIGH_FREQ})')
    parser.add_argument('--notch_freq', type=float, default=NOTCH_FREQ,
                        help=f'Notch filter frequency (default: {NOTCH_FREQ})')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for LMDB')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes (default: 4)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for LMDB writing (default: 500)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Unified EEG Preprocessing for Diagnosis Classification")
    logger.info("(3-class: CVD, depression, normal)")
    logger.info("=" * 70)
    logger.info(f"Common channels: {N_COMMON_CHANNELS}")
    logger.info(f"Target sampling rate: {args.target_sfreq} Hz")
    logger.info(f"Bandpass filter: {args.low_freq}-{args.high_freq} Hz")
    logger.info(f"Notch filter: {args.notch_freq} Hz")
    logger.info(f"Segment duration: {args.segment_duration} seconds")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of workers: {args.num_workers}")

    # Collect all file tasks
    all_tasks = []
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info['folder']
        label = dataset_info['label']
        label_name = dataset_info['label_name']
        system = dataset_info['system']
        file_pattern = dataset_info['file_pattern']

        if not folder.exists():
            logger.error(f"Folder not found: {folder}")
            continue

        # Find files based on pattern
        if file_pattern == '.mff':
            files = sorted([f for f in folder.iterdir() if f.is_dir() and f.suffix == '.mff'])
        else:  # .vhdr
            files = sorted(folder.glob('*.vhdr'))

        logger.info(f"Found {len(files)} files in {dataset_name} ({system})")

        for file_path in files:
            all_tasks.append((
                file_path, label, label_name, system,
                args.target_sfreq, args.segment_duration,
                args.low_freq, args.high_freq, args.notch_freq
            ))

    logger.info(f"Total files to process: {len(all_tasks)}")

    if len(all_tasks) == 0:
        logger.error("No files to process!")
        return

    # Shuffle tasks
    np.random.seed(42)
    np.random.shuffle(all_tasks)

    # Initialize LMDB writer
    output_path = Path(args.output_dir)
    writer = LMDBWriter(output_path, batch_size=args.batch_size)

    # Statistics
    file_stats = defaultdict(lambda: {'files': 0, 'segments': 0, 'errors': 0})
    error_files = []

    # Process files
    logger.info(f"\nProcessing with {args.num_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in all_tasks}

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                dataset_key = f"{result['label_name']}_{result['system']}"

                if result['success']:
                    writer.add_samples(
                        result['samples'],
                        n_channels=result['n_channels'],
                        orig_sfreq=result['orig_sfreq'],
                        ch_names=result.get('ch_names')
                    )
                    file_stats[dataset_key]['files'] += 1
                    file_stats[dataset_key]['segments'] += len(result['samples'])
                    result['samples'] = None
                else:
                    file_stats[dataset_key]['errors'] += 1
                    error_files.append((result['file'], result['error']))
                    logger.debug(f"Failed: {result['file']}: {result['error']}")

                pbar.update(1)

    # Finalize
    total_samples = writer.close(
        args.target_sfreq, args.segment_duration,
        args.low_freq, args.high_freq, args.notch_freq
    )

    # Print statistics
    logger.info("\n" + "=" * 70)
    logger.info("Processing Statistics")
    logger.info("=" * 70)
    for dataset_key, stats in sorted(file_stats.items()):
        logger.info(f"{dataset_key}:")
        logger.info(f"  Files processed: {stats['files']}")
        logger.info(f"  Segments created: {stats['segments']}")
        logger.info(f"  Errors: {stats['errors']}")

    if error_files:
        logger.warning(f"\nErrors ({len(error_files)} files):")
        error_log_path = output_path / 'failed_files.txt'
        with open(error_log_path, 'w') as f:
            for fname, err in error_files:
                f.write(f"{fname}: {err}\n")
        logger.info(f"Error log saved to {error_log_path}")

        for fname, err in error_files[:5]:
            logger.warning(f"  {fname}: {err}")
        if len(error_files) > 5:
            logger.warning(f"  ... and {len(error_files) - 5} more errors")

    logger.info("\n" + "=" * 70)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 70)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Output: {output_path / 'eeg_segments.lmdb'}")


def verify_output(output_dir=OUTPUT_DIR):
    """Verify the output LMDB file."""
    logger.info("\n" + "=" * 70)
    logger.info("Verifying Output")
    logger.info("=" * 70)

    lmdb_path = Path(output_dir) / "eeg_segments.lmdb"
    if lmdb_path.exists():
        reader = LMDBReader(lmdb_path)
        logger.info(f"LMDB path: {lmdb_path}")
        logger.info(f"Total samples: {len(reader)}")

        meta = reader.metadata
        logger.info(f"Channels: {meta['n_channels']}")
        logger.info(f"Sampling rate: {meta['sampling_rate']} Hz")
        logger.info(f"Label counts: {meta['label_counts']}")
        logger.info(f"Subjects: {meta['n_subjects']}")

        # Read a few samples
        if len(reader) > 0:
            for i in [0, len(reader) // 2, len(reader) - 1]:
                sample = reader[i]
                logger.info(f"\nSample {i}:")
                logger.info(f"  Signal shape: {sample['signal'].shape}")
                logger.info(f"  Label: {sample['label']} ({LABEL_NAMES[sample['label']]})")
                logger.info(f"  Subject ID: {sample.get('subject_id', 'N/A')}")
                logger.info(f"  System: {sample.get('system', 'N/A')}")
                logger.info(f"  Source file: {sample['source_file']}")

        reader.close()
    else:
        logger.error(f"LMDB not found: {lmdb_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
        verify_output(output_dir)
    else:
        main()
        verify_output()
