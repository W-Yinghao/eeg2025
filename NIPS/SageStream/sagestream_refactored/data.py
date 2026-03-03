"""
Data loading and preprocessing module for SageStream.

This module provides:
- APAVADataset: PyTorch Dataset for APAVA data
- TUABDataset: PyTorch Dataset for TUAB data (LMDB or pickle format)
- Data splitting utilities (k-fold, random split)
- DataLoader creation utilities
"""

from typing import Optional, List, Tuple, Dict, Any
import hashlib
import pickle
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torcheeg import transforms

# Try to import lmdb for TUAB LMDB format support
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False


class APAVADataset(Dataset):
    """
    APAVA Dataset for EEG classification.

    This dataset loads EEG features and labels from the APAVA dataset,
    supporting subject-based splitting and caching.

    Args:
        root_path: Root directory of the dataset
        subset: 'train', 'test', or 'all'
        online_transform: Transform to apply to EEG data
        label_transform: Transform to apply to labels
        use_cache: Whether to use cached data
        cache_dir: Directory for caching
        train_ids: List of subject IDs for training
        test_ids: List of subject IDs for testing
    """

    def __init__(
        self,
        root_path: str,
        subset: str = 'all',
        online_transform: Optional[transforms.Compose] = None,
        label_transform: Optional[Any] = None,
        use_cache: bool = True,
        cache_dir: str = './cache',
        train_ids: Optional[List[int]] = None,
        test_ids: Optional[List[int]] = None
    ):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        self.online_transform = online_transform
        self.label_transform = label_transform
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Load label info to get subject IDs
        data_list = np.load(self.label_path)
        all_ids = list(data_list[:, 1].astype(int))

        self.train_ids = train_ids
        self.test_ids = test_ids

        # Set subset
        if subset == 'train':
            self.ids = self.train_ids
            self.subset_name = 'train'
        elif subset == 'test':
            self.ids = self.test_ids
            self.subset_name = 'test'
        else:
            self.ids = all_ids
            self.subset_name = 'all'

        # Generate cache key
        ids_str = '_'.join(map(str, sorted(self.ids)))
        cache_key = f"{self.subset_name}_{ids_str}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        self.cache_file = os.path.join(self.cache_dir, f'apava_{self.subset_name}_{cache_hash}.pkl')
        self.cache_key = cache_key
        self.cache_hash = cache_hash

        # Load data
        self.X, self.y, self.subject_ids = self._load_apava()
        self.data = self.X
        self.labels = self.y

    def _load_apava(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load APAVA data from files or cache."""
        # Try loading from cache
        if self.use_cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['X'], cached_data['y'], cached_data['subject_ids']

        # Load from files
        feature_list = []
        label_list = []
        subject_id_list = []
        filenames = sorted(os.listdir(self.data_path))

        subject_label = np.load(self.label_path)

        for j, filename in enumerate(filenames):
            if j + 1 in self.ids:
                trial_label = subject_label[j]
                path = os.path.join(self.data_path, filename)
                subject_feature = np.load(path)

                for trial_feature in subject_feature:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
                    subject_id_list.append(j + 1)

        X = np.array(feature_list)
        y = np.array(label_list)
        subject_ids = np.array(subject_id_list)

        # Shuffle with fixed seed for reproducibility
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        subject_ids = subject_ids[indices]

        # Transpose to [samples, channels, time] if needed
        # Check if data has 3 dimensions (samples, time, channels) or 2 dimensions
        if X.ndim == 3:
            # Assume shape is (samples, time, channels) -> transpose to (samples, channels, time)
            X = X.transpose(0, 2, 1)
        elif X.ndim == 2:
            # Data is 2D (samples, features), no transpose needed
            # This might be the case if features are already flattened
            pass
        else:
            raise ValueError(f"Unexpected data dimensions: {X.ndim}, shape: {X.shape}")

        # Save to cache
        if self.use_cache:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            cache_data = {
                'X': X,
                'y': y[:, 0],
                'subject_ids': subject_ids,
                'cache_key': self.cache_key,
                'subset_name': self.subset_name,
                'ids': sorted(self.ids)
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        return X, y[:, 0], subject_ids

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single sample."""
        eeg_data = self.X[index].astype(np.float32)
        label = self.y[index].astype(np.int64)
        subject_id = self.subject_ids[index]

        if self.online_transform:
            sample = self.online_transform(eeg=eeg_data)
            eeg_data = sample['eeg']

        if self.label_transform:
            label = self.label_transform(label)

        return eeg_data, label, subject_id

    def __len__(self) -> int:
        return len(self.y)


class PTBDataset(Dataset):
    """
    PTB Dataset for ECG/EEG classification.

    Same directory structure as APAVA:
        - Label/label.npy: (198, 2) with [class_label, subject_id]
        - Feature/feature_XXX.npy: (num_samples, 300, 15) per subject

    Args:
        root_path: Root directory of the dataset
        subset: 'train', 'test', or 'all'
        online_transform: Transform to apply to data
        label_transform: Transform to apply to labels
        use_cache: Whether to use cached data
        cache_dir: Directory for caching
        train_ids: List of subject IDs for training
        test_ids: List of subject IDs for testing
    """

    def __init__(
        self,
        root_path: str,
        subset: str = 'all',
        online_transform: Optional[transforms.Compose] = None,
        label_transform: Optional[Any] = None,
        use_cache: bool = True,
        cache_dir: str = './cache',
        train_ids: Optional[List[int]] = None,
        test_ids: Optional[List[int]] = None
    ):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        self.online_transform = online_transform
        self.label_transform = label_transform
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Load label info to get subject IDs
        data_list = np.load(self.label_path)
        all_ids = list(data_list[:, 1].astype(int))

        self.train_ids = train_ids
        self.test_ids = test_ids

        # Set subset
        if subset == 'train':
            self.ids = self.train_ids
            self.subset_name = 'train'
        elif subset == 'test':
            self.ids = self.test_ids
            self.subset_name = 'test'
        else:
            self.ids = all_ids
            self.subset_name = 'all'

        # Generate cache key
        ids_str = '_'.join(map(str, sorted(self.ids)))
        cache_key = f"{self.subset_name}_{ids_str}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        self.cache_file = os.path.join(self.cache_dir, f'ptb_{self.subset_name}_{cache_hash}.pkl')
        self.cache_key = cache_key
        self.cache_hash = cache_hash

        # Load data
        self.X, self.y, self.subject_ids = self._load_data()
        self.data = self.X
        self.labels = self.y

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load PTB data from files or cache."""
        # Try loading from cache
        if self.use_cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['X'], cached_data['y'], cached_data['subject_ids']

        # Load from files
        feature_list = []
        label_list = []
        subject_id_list = []
        filenames = sorted(os.listdir(self.data_path))

        subject_label = np.load(self.label_path)

        for j, filename in enumerate(filenames):
            if j + 1 in self.ids:
                trial_label = subject_label[j]
                path = os.path.join(self.data_path, filename)
                subject_feature = np.load(path)

                for trial_feature in subject_feature:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
                    subject_id_list.append(j + 1)

        X = np.array(feature_list)
        y = np.array(label_list)
        subject_ids = np.array(subject_id_list)

        # Shuffle with fixed seed for reproducibility
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        subject_ids = subject_ids[indices]

        # Transpose to [samples, channels, time] if needed
        if X.ndim == 3:
            # Shape is (samples, time, channels) -> transpose to (samples, channels, time)
            X = X.transpose(0, 2, 1)
        elif X.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected data dimensions: {X.ndim}, shape: {X.shape}")

        # Save to cache
        if self.use_cache:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            cache_data = {
                'X': X,
                'y': y[:, 0],
                'subject_ids': subject_ids,
                'cache_key': self.cache_key,
                'subset_name': self.subset_name,
                'ids': sorted(self.ids)
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        return X, y[:, 0], subject_ids

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single sample."""
        eeg_data = self.X[index].astype(np.float32)
        label = self.y[index].astype(np.int64)
        subject_id = self.subject_ids[index]

        if self.online_transform:
            sample = self.online_transform(eeg=eeg_data)
            eeg_data = sample['eeg']

        if self.label_transform:
            label = self.label_transform(label)

        return eeg_data, label, subject_id

    def __len__(self) -> int:
        return len(self.y)


class TUABDataset(Dataset):
    """
    TUAB (TUH EEG Abnormal) Dataset for EEG classification.

    Supports both LMDB format and pickle file format.

    LMDB format (from preprocessing_tuab.py):
        - signal: (16, 2000) - 16 bipolar channels, 10 seconds at 200Hz
        - label: int (0=normal, 1=abnormal)
        - source_file: str
        - segment_idx: int

    Pickle format (alternative):
        - X: (16, 2000) signal data
        - y: int label

    Args:
        data_path: Path to the data directory (LMDB dir or pickle file dir)
        mode: 'train', 'val', or 'test'
        data_format: 'lmdb' or 'pickle'
        target_seq_len: Target sequence length (will resample if needed)
        normalize: Whether to normalize data (divide by 100 for uV scaling)
    """

    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        data_format: str = 'lmdb',
        target_seq_len: int = 256,
        normalize: bool = True,
        online_transform: Optional[transforms.Compose] = None
    ):
        self.data_path = data_path
        self.mode = mode
        self.data_format = data_format
        self.target_seq_len = target_seq_len
        self.normalize = normalize
        self.online_transform = online_transform

        if data_format == 'lmdb':
            self._init_lmdb()
        else:
            self._init_pickle()

    def _init_lmdb(self):
        """Initialize LMDB dataset."""
        if not LMDB_AVAILABLE:
            raise ImportError("lmdb is required for LMDB format. Install with: pip install lmdb")

        # Patch numpy for pickle compatibility with older numpy versions
        import sys
        try:
            import numpy._core.numeric
        except ImportError:
            import numpy.core.numeric
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.numeric'] = numpy.core.numeric

        lmdb_path = os.path.join(self.data_path, self.mode)
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

        self.env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        with self.env.begin() as txn:
            self._length = pickle.loads(txn.get('__length__'.encode()))
            self._metadata = pickle.loads(txn.get('__metadata__'.encode()))

        # Extract subject IDs from source files for cross-subject splits
        self.subject_ids = []
        self._extract_subject_info()

        print(f"Loaded TUAB {self.mode} split: {self._length} samples")
        print(f"  Label distribution: {self._metadata.get('label_counts', {})}")

    def _extract_subject_info(self):
        """Extract subject IDs from source files for cross-subject experiments.

        Uses a consistent hash mapping from subject name -> integer ID.
        Subject name is extracted from the source_file field at __getitem__ time
        to avoid slow full-scan of large LMDB databases at init.
        """
        # We use lazy per-sample extraction in __getitem__ instead of
        # scanning all 297K+ entries at init time.
        # Initialize subject_name_to_id mapping with a small sample.
        self._subject_name_to_id = {}
        self._next_subject_id = 0

        # Pre-populate with a small sample to estimate unique subjects
        sample_size = min(self._length, 2000)
        with self.env.begin() as txn:
            for i in range(sample_size):
                key = f'{i:08d}'.encode()
                value = txn.get(key)
                if value is None:
                    continue
                try:
                    sample = pickle.loads(value)
                    source = sample.get('source_file', '')
                    match = re.search(r'([a-z]+)_s\d+_t\d+', source)
                    if match:
                        subject_name = match.group(1)
                        if subject_name not in self._subject_name_to_id:
                            self._subject_name_to_id[subject_name] = self._next_subject_id
                            self._next_subject_id += 1
                except Exception:
                    pass

        # Fill subject_ids list for all samples (use 0 as placeholder,
        # actual IDs are resolved in __getitem__)
        self.subject_ids = list(range(self._length))

        print(f"  Unique subjects sampled: {len(self._subject_name_to_id)} (from first {sample_size} samples)")

    def _init_pickle(self):
        """Initialize pickle file dataset."""
        self.data_files = sorted([
            os.path.join(self.data_path, self.mode, f)
            for f in os.listdir(os.path.join(self.data_path, self.mode))
            if f.endswith('.pkl') or f.endswith('.pickle')
        ])
        self._length = len(self.data_files)
        self.subject_ids = list(range(self._length))
        print(f"Loaded TUAB {self.mode} split: {self._length} samples (pickle format)")

    def _get_sample_lmdb(self, idx: int) -> Dict:
        """Get a sample from LMDB."""
        key = f'{idx:08d}'.encode()
        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {idx} not found")
            # Try to load with different pickle protocols for compatibility
            try:
                return pickle.loads(value)
            except ModuleNotFoundError:
                # Handle numpy version incompatibility
                import io
                import sys
                # Patch numpy._core.numeric to numpy.core.numeric for old pickles
                if 'numpy._core.numeric' not in sys.modules:
                    sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric', None)
                return pickle.loads(value)

    def _resolve_subject_id(self, sample: Dict, index: int) -> int:
        """Resolve subject ID from sample source_file, with lazy caching."""
        if self.data_format != 'lmdb':
            return index

        source = sample.get('source_file', '')
        match = re.search(r'([a-z]+)_s\d+_t\d+', source)
        if match:
            subject_name = match.group(1)
            if subject_name not in self._subject_name_to_id:
                self._subject_name_to_id[subject_name] = self._next_subject_id
                self._next_subject_id += 1
            return self._subject_name_to_id[subject_name]
        return index

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single sample."""
        if self.data_format == 'lmdb':
            sample = self._get_sample_lmdb(index)
            signal = sample['signal'].astype(np.float32)
            label = int(sample['label'])
            subject_id = self._resolve_subject_id(sample, index)
        else:
            with open(self.data_files[index], 'rb') as f:
                data_dict = pickle.load(f)
            signal = data_dict['X'].astype(np.float32)
            label = int(data_dict['y'])
            subject_id = index

        # Normalize (data is in uV, divide by 100 for stability)
        if self.normalize:
            signal = signal / 100.0

        # Resample if needed (from 2000 to target_seq_len)
        if signal.shape[1] != self.target_seq_len:
            from scipy import signal as scipy_signal
            signal = scipy_signal.resample(signal, self.target_seq_len, axis=-1)

        signal = signal.astype(np.float32)

        if self.online_transform:
            sample_dict = self.online_transform(eeg=signal)
            signal = sample_dict['eeg']

        return signal, label, subject_id

    def __len__(self) -> int:
        return self._length

    def close(self):
        """Close LMDB environment."""
        if self.data_format == 'lmdb' and hasattr(self, 'env'):
            self.env.close()


class SEEDVDataset(Dataset):
    """
    SEED-V Dataset for EEG emotion classification.

    Uses LMDB format from CBraMod preprocessing.

    Data format:
        - sample: (62, 1, 200) - 62 channels, 1 second at 200Hz
        - label: int (0-4, 5 emotion classes)

    Args:
        data_path: Path to the LMDB data directory
        mode: 'train', 'val', or 'test'
        target_seq_len: Target sequence length (default 200, matches preprocessed data)
        online_transform: Transform to apply to EEG data
    """

    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        target_seq_len: int = 200,
        online_transform: Optional[transforms.Compose] = None
    ):
        self.data_path = data_path
        self.mode = mode
        self.target_seq_len = target_seq_len
        self.online_transform = online_transform

        if not LMDB_AVAILABLE:
            raise ImportError("lmdb is required for SEED-V dataset. Install with: pip install lmdb")

        self._init_lmdb()

    def _init_lmdb(self):
        """Initialize LMDB dataset."""
        # Patch numpy for pickle compatibility with older numpy versions
        import sys
        try:
            import numpy._core.numeric
        except ImportError:
            import numpy.core.numeric
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.numeric'] = numpy.core.numeric

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"LMDB path not found: {self.data_path}")

        self.env = lmdb.open(
            str(self.data_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # Load keys index
        with self.env.begin() as txn:
            keys_raw = txn.get('__keys__'.encode())
            if keys_raw is None:
                raise KeyError(
                    f"SEED-V LMDB at {self.data_path} does not contain '__keys__' entry. "
                    f"Make sure you are using the correct preprocessed LMDB directory "
                    f"(e.g., 'processed_kiet' instead of 'processed_cbramod')."
                )
            keys_dict = pickle.loads(keys_raw)

        if not isinstance(keys_dict, dict) or self.mode not in keys_dict:
            available = list(keys_dict.keys()) if isinstance(keys_dict, dict) else type(keys_dict).__name__
            raise KeyError(
                f"SEED-V LMDB '__keys__' does not contain mode '{self.mode}'. "
                f"Available: {available}"
            )

        # Get keys for this mode
        self.keys = keys_dict[self.mode]
        self._length = len(self.keys)

        # Extract subject IDs from keys
        # Key format: '{subject_id}_{session}_{date}.cnt-{trial_idx}-{sample_idx}'
        # e.g., '10_1_20180507.cnt-0-0' -> subject_id = 10
        self.subject_ids = []
        self._extract_subject_info()

        print(f"Loaded SEED-V {self.mode} split: {self._length} samples")

    def _extract_subject_info(self):
        """Extract subject IDs from sample keys.

        Key format: '{subject_id}_{session}_{date}.cnt-{trial_idx}-{sample_idx}'
        e.g., '10_1_20180507.cnt-0-0' -> subject_id = 10 (1-based integer)
        """
        for key in self.keys:
            try:
                # First part before '_' is the integer subject ID
                subject_id_str = key.split('_')[0]
                subject_id = int(subject_id_str)
                self.subject_ids.append(subject_id)
            except (ValueError, IndexError):
                self.subject_ids.append(0)

    def _get_sample(self, idx: int) -> Dict:
        """Get a sample from LMDB."""
        key = self.keys[idx].encode()
        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {self.keys[idx]} not found")
            try:
                return pickle.loads(value)
            except ModuleNotFoundError:
                import sys
                if 'numpy._core.numeric' not in sys.modules:
                    sys.modules['numpy._core.numeric'] = sys.modules.get('numpy.core.numeric', None)
                return pickle.loads(value)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single sample."""
        sample = self._get_sample(index)

        # Data shape: (62, 1, 200) -> squeeze to (62, 200)
        signal = sample['sample'].astype(np.float32)
        if signal.ndim == 3:
            signal = signal.squeeze(1)  # Remove the middle dimension

        label = int(sample['label'])

        # Resample if needed
        if signal.shape[1] != self.target_seq_len:
            from scipy import signal as scipy_signal
            signal = scipy_signal.resample(signal, self.target_seq_len, axis=-1)

        signal = signal.astype(np.float32)

        if self.online_transform:
            sample_dict = self.online_transform(eeg=signal)
            signal = sample_dict['eeg']

        subject_id = self.subject_ids[index] if index < len(self.subject_ids) else 0

        return signal, label, subject_id

    def __len__(self) -> int:
        return self._length

    def close(self):
        """Close LMDB environment."""
        if hasattr(self, 'env'):
            self.env.close()


def get_random_subject_split(
    all_subject_ids: List[int],
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split subjects randomly into train and test sets.

    Args:
        all_subject_ids: List of all subject IDs
        test_ratio: Ratio of test subjects
        random_state: Random seed

    Returns:
        Tuple of (train_ids, test_ids)
    """
    import random
    random.seed(random_state)

    subject_ids = all_subject_ids.copy()
    random.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    n_test = max(1, int(n_subjects * test_ratio))
    n_train = n_subjects - n_test

    if n_train <= 0:
        n_train = 1
        n_test = n_subjects - n_train

    test_ids = subject_ids[:n_test]
    train_ids = subject_ids[n_test:]

    return train_ids, test_ids


def get_k_fold_subject_splits(
    all_subject_ids: List[int],
    k: int = 5,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate k-fold cross-validation splits by subject.

    Args:
        all_subject_ids: List of all subject IDs
        k: Number of folds
        random_state: Random seed

    Returns:
        List of (train_ids, test_ids) tuples for each fold
    """
    import random
    random.seed(random_state)

    subject_ids = all_subject_ids.copy()
    random.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    fold_size = n_subjects // k
    remainder = n_subjects % k

    folds = []
    start_idx = 0

    for fold_idx in range(k):
        current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + current_fold_size

        test_ids = subject_ids[start_idx:end_idx]
        train_ids = subject_ids[:start_idx] + subject_ids[end_idx:]

        folds.append((train_ids, test_ids))
        start_idx = end_idx

    return folds


def split_dataset_by_subject(
    dataset: Dataset,
    train_ratio: float = 0.75,
    random_state: int = 42
) -> Tuple[Subset, Subset]:
    """
    Split a dataset by subject into train and validation sets.

    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training subjects
        random_state: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    all_subject_ids = np.array(dataset.subject_ids)
    unique_subjects = np.unique(all_subject_ids)

    np.random.seed(random_state)
    np.random.shuffle(unique_subjects)

    n_train_subjects = int(len(unique_subjects) * train_ratio)
    train_subjects = unique_subjects[:n_train_subjects]
    val_subjects = unique_subjects[n_train_subjects:]

    train_indices = []
    val_indices = []

    for i in range(len(dataset)):
        subject_id = all_subject_ids[i]
        if subject_id in train_subjects:
            train_indices.append(i)
        else:
            val_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


class DataLoaderFactory:
    """
    Factory class for creating DataLoaders.

    This class provides a unified interface for creating DataLoaders
    with consistent settings across different experiments and datasets.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "APAVA",
        cache_dir: str = './cache',
        use_cache: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_seq_len: int = 256,
        **kwargs  # Additional dataset-specific options
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_seq_len = target_seq_len
        self.kwargs = kwargs

    def _create_apava_dataset(
        self,
        subset: str,
        train_ids: List[int],
        test_ids: List[int]
    ) -> APAVADataset:
        """Create an APAVA dataset with given split."""
        return APAVADataset(
            root_path=self.dataset_path,
            subset=subset,
            online_transform=transforms.Compose([transforms.ToTensor()]),
            label_transform=None,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            train_ids=train_ids,
            test_ids=test_ids
        )

    def _create_ptb_dataset(
        self,
        subset: str,
        train_ids: List[int],
        test_ids: List[int]
    ) -> PTBDataset:
        """Create a PTB dataset with given split."""
        return PTBDataset(
            root_path=self.dataset_path,
            subset=subset,
            online_transform=transforms.Compose([transforms.ToTensor()]),
            label_transform=None,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            train_ids=train_ids,
            test_ids=test_ids
        )

    def _create_tuab_dataset(
        self,
        mode: str
    ) -> TUABDataset:
        """Create a TUAB dataset for given mode."""
        return TUABDataset(
            data_path=self.dataset_path,
            mode=mode,
            data_format=self.kwargs.get('tuab_data_format', 'lmdb'),
            target_seq_len=self.target_seq_len,
            normalize=self.kwargs.get('tuab_normalize', True),
            online_transform=transforms.Compose([transforms.ToTensor()])
        )

    def _create_seedv_dataset(
        self,
        mode: str
    ) -> SEEDVDataset:
        """Create a SEED-V dataset for given mode."""
        return SEEDVDataset(
            data_path=self.dataset_path,
            mode=mode,
            target_seq_len=self.target_seq_len,
            online_transform=transforms.Compose([transforms.ToTensor()])
        )

    def _create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False
    ) -> DataLoader:
        """Create a DataLoader from a dataset."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory
        )

    def get_k_fold_loaders(
        self,
        k: int = 5,
        random_state: int = 42,
        val_split_ratio: float = 0.25
    ) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get k-fold cross-validation DataLoaders.

        Args:
            k: Number of folds
            random_state: Random seed
            val_split_ratio: Ratio of validation split from training data

        Returns:
            List of (train_loader, val_loader, test_loader) tuples
        """
        if self.dataset_name == "TUAB":
            # TUAB uses pre-defined train/val/test splits
            return self._get_tuab_loaders()

        if self.dataset_name == "SEEDV":
            # SEED-V uses pre-defined train/val/test splits
            return self._get_seedv_loaders()

        # APAVA / PTB: subject-based k-fold cross-validation
        label_path = os.path.join(self.dataset_path, "Label/label.npy")
        data_list = np.load(label_path)
        all_subject_ids = list(set(data_list[:, 1].astype(int)))

        fold_splits = get_k_fold_subject_splits(all_subject_ids, k=k, random_state=random_state)

        # Select dataset factory based on name
        if self.dataset_name == "PTB":
            create_dataset = self._create_ptb_dataset
        else:
            create_dataset = self._create_apava_dataset

        fold_loaders = []

        for fold_idx, (train_ids, test_ids) in enumerate(fold_splits):
            print(f"\nFold {fold_idx + 1}/{k}:")
            print(f"  Train subjects: {sorted(train_ids)} (n={len(train_ids)})")
            print(f"  Test subjects: {sorted(test_ids)} (n={len(test_ids)})")

            # Create full train dataset
            train_dataset = create_dataset('train', train_ids, test_ids)
            test_dataset = create_dataset('test', train_ids, test_ids)

            # Split train into train/val
            train_subset, val_subset = split_dataset_by_subject(
                train_dataset,
                train_ratio=1 - val_split_ratio,
                random_state=random_state
            )

            # Create loaders
            train_loader = self._create_dataloader(train_subset, shuffle=True)
            val_loader = self._create_dataloader(val_subset, shuffle=False)
            test_loader = self._create_dataloader(test_dataset, shuffle=False)

            fold_loaders.append((train_loader, val_loader, test_loader))

            print(f"  Train samples: {len(train_subset)}")
            print(f"  Val samples: {len(val_subset)}")
            print(f"  Test samples: {len(test_dataset)}")

        return fold_loaders

    def _get_tuab_loaders(self) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get TUAB loaders using pre-defined splits."""
        print("\nLoading TUAB dataset with pre-defined train/val/test splits...")

        # TUAB already has train/val/test splits
        train_dataset = self._create_tuab_dataset('train')
        val_dataset = self._create_tuab_dataset('val')
        test_dataset = self._create_tuab_dataset('test')

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        # Return as a list with single fold for compatibility
        return [(train_loader, val_loader, test_loader)]

    def _get_seedv_loaders(self) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get SEED-V loaders using pre-defined splits."""
        print("\nLoading SEED-V dataset with pre-defined train/val/test splits...")

        # SEED-V already has train/val/test splits
        train_dataset = self._create_seedv_dataset('train')
        val_dataset = self._create_seedv_dataset('val')
        test_dataset = self._create_seedv_dataset('test')

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        # Return as a list with single fold for compatibility
        return [(train_loader, val_loader, test_loader)]

    def get_single_split_loaders(
        self,
        test_ratio: float = 0.2,
        random_state: int = 42,
        val_split_ratio: float = 0.25
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get a single train/val/test split DataLoaders.

        Args:
            test_ratio: Ratio of test subjects
            random_state: Random seed
            val_split_ratio: Ratio of validation split from training data

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.dataset_name == "TUAB":
            # TUAB uses pre-defined splits
            loaders = self._get_tuab_loaders()
            return loaders[0]

        if self.dataset_name == "SEEDV":
            # SEED-V uses pre-defined splits
            loaders = self._get_seedv_loaders()
            return loaders[0]

        # APAVA / PTB: Load label info to get subject IDs
        label_path = os.path.join(self.dataset_path, "Label/label.npy")
        data_list = np.load(label_path)
        all_subject_ids = list(set(data_list[:, 1].astype(int)))

        train_ids, test_ids = get_random_subject_split(
            all_subject_ids, test_ratio=test_ratio, random_state=random_state
        )

        print(f"Train subjects: {sorted(train_ids)} (n={len(train_ids)})")
        print(f"Test subjects: {sorted(test_ids)} (n={len(test_ids)})")

        # Select dataset factory based on name
        if self.dataset_name == "PTB":
            create_dataset = self._create_ptb_dataset
        else:
            create_dataset = self._create_apava_dataset

        # Create datasets
        train_dataset = create_dataset('train', train_ids, test_ids)
        test_dataset = create_dataset('test', train_ids, test_ids)

        # Split train into train/val
        train_subset, val_subset = split_dataset_by_subject(
            train_dataset,
            train_ratio=1 - val_split_ratio,
            random_state=random_state
        )

        # Create loaders
        train_loader = self._create_dataloader(train_subset, shuffle=True)
        val_loader = self._create_dataloader(val_subset, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)

        return train_loader, val_loader, test_loader
