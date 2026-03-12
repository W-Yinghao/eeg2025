"""
EEG数据分割和LMDB存储脚本

将预处理后的EEG数据分割成1秒的片段，添加标签后存储为LMDB格式。

标签:
1. 采集设备: 'egi', 'brainvision', 'eeglab'
2. 疾病类型: 'AD', 'normal', 'depression', 'CVD'

输入: /projects/EEG-foundation-model/diagnosis_data_preprocessed/
输出: /projects/EEG-foundation-model/diagnosis_data_lmdb/

作者: Yinghao WANG
日期: 2025.2
"""

import os
import sys
from pathlib import Path
import numpy as np
import mne
import lmdb
import pickle
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制MNE的详细输出
mne.set_log_level('WARNING')


# ============================================================================
# 数据集配置
# ============================================================================
DATASET_CONFIG = {
    'eeg_AD+MCI+SCD+HC_EGI_124': {
        'device': 'eeglab',      # 已经用EEGLAB预处理过
        'disease': 'AD',         # AD/MCI/SCD/HC，统一标记为AD相关
    },
    'eeg_CVD_EGI_83': {
        'device': 'egi',
        'disease': 'CVD',
    },
    'eeg_depression_BP_122': {
        'device': 'brainvision',
        'disease': 'depression',
    },
    'eeg_depression_EGI_21': {
        'device': 'egi',
        'disease': 'depression',
    },
    'eeg_normal_BP_166': {
        'device': 'brainvision',
        'disease': 'normal',
    },
    'eeg_normal_EGI_17': {
        'device': 'egi',
        'disease': 'normal',
    },
}

# 分割参数
SEGMENT_DURATION = 5.0  # 秒 (seq_len=5, 匹配TUEV/TUAB/AD_DIAGNOSIS)
SAMPLING_RATE = 200     # Hz (匹配CBraMod/CodeBrain/LUNA的patch_size=200)
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLING_RATE)  # 1000 samples


class EEGSegmenter:
    """EEG数据分割器"""

    def __init__(self, segment_duration=1.0, overlap=0.0):
        """
        初始化分割器

        Parameters
        ----------
        segment_duration : float
            片段时长（秒）
        overlap : float
            重叠比例 (0.0 - 1.0)
        """
        self.segment_duration = segment_duration
        self.overlap = overlap

    def segment_file(self, file_path, device, disease, subject_id=None):
        """
        分割单个文件

        Parameters
        ----------
        file_path : str or Path
            .fif文件路径
        device : str
            采集设备类型
        disease : str
            疾病类型
        subject_id : str, optional
            被试ID，用于cross-subject split

        Returns
        -------
        list
            分割后的数据列表，每个元素为 (segment_data, labels_dict)
        """
        file_path = Path(file_path)
        if subject_id is None:
            # 从文件名提取subject_id: xxx_preprocessed.fif -> xxx
            subject_id = file_path.stem.replace('_preprocessed', '')

        try:
            # 读取数据
            raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose=False)
            data = raw.get_data()  # shape: (n_channels, n_samples)
            sfreq = raw.info['sfreq']

            # 计算片段参数
            segment_samples = int(self.segment_duration * sfreq)
            step_samples = int(segment_samples * (1 - self.overlap))

            n_channels, n_samples = data.shape
            segments = []

            # 分割数据
            start = 0
            segment_idx = 0
            while start + segment_samples <= n_samples:
                segment_data = data[:, start:start + segment_samples]

                # 创建标签字典
                labels = {
                    'device': device,
                    'disease': disease,
                    'subject_id': subject_id,
                    'source_file': file_path.name,
                    'segment_idx': segment_idx,
                    'start_sample': start,
                    'end_sample': start + segment_samples,
                    'sfreq': sfreq,
                    'n_channels': n_channels,
                }

                segments.append((segment_data.astype(np.float32), labels))

                start += step_samples
                segment_idx += 1

            return segments

        except Exception as e:
            logger.error(f"分割文件失败 {file_path.name}: {str(e)}")
            return []


def process_dataset(input_dir, dataset_name, config):
    """
    处理单个数据集

    Parameters
    ----------
    input_dir : Path
        输入目录
    dataset_name : str
        数据集名称
    config : dict
        数据集配置

    Returns
    -------
    list
        所有分割后的数据
    """
    device = config['device']
    disease = config['disease']

    logger.info(f"处理数据集: {dataset_name}")
    logger.info(f"  设备: {device}, 疾病: {disease}")

    segmenter = EEGSegmenter(segment_duration=SEGMENT_DURATION)

    # 获取所有.fif文件
    fif_files = list(input_dir.glob('*.fif'))
    logger.info(f"  发现 {len(fif_files)} 个文件")

    all_segments = []

    for fif_file in tqdm(fif_files, desc=f"  分割 {dataset_name}"):
        segments = segmenter.segment_file(fif_file, device, disease)
        all_segments.extend(segments)

    logger.info(f"  生成 {len(all_segments)} 个片段")
    return all_segments


def write_to_lmdb(segments, output_path, map_size=100*1024*1024*1024):
    """
    将分割后的数据写入LMDB

    Parameters
    ----------
    segments : list
        分割后的数据列表
    output_path : Path
        LMDB输出路径
    map_size : int
        LMDB映射大小（字节）
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    lmdb_path = output_path / 'eeg_segments.lmdb'

    logger.info(f"写入LMDB: {lmdb_path}")
    logger.info(f"  总片段数: {len(segments)}")

    # 创建LMDB环境
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True
    )

    # 写入数据
    with env.begin(write=True) as txn:
        # 写入元数据
        subject_ids = list(set(labels['subject_id'] for _, labels in segments))
        metadata = {
            'n_segments': len(segments),
            'n_subjects': len(subject_ids),
            'segment_duration': SEGMENT_DURATION,
            'sampling_rate': SAMPLING_RATE,
            'segment_samples': SEGMENT_SAMPLES,
            'created_at': datetime.now().isoformat(),
        }
        txn.put('__metadata__'.encode(), pickle.dumps(metadata))

        # 写入每个片段
        for idx, (segment_data, labels) in enumerate(tqdm(segments, desc="  写入LMDB")):
            key = f'{idx:08d}'.encode()
            value = pickle.dumps({
                'data': segment_data,
                'labels': labels
            })
            txn.put(key, value)

        # 写入索引信息
        txn.put('__length__'.encode(), pickle.dumps(len(segments)))

    env.close()

    # 保存标签统计
    stats = compute_statistics(segments)
    stats_path = output_path / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write(f"EEG Segments Statistics\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total segments: {len(segments)}\n")
        f.write(f"Segment duration: {SEGMENT_DURATION} seconds\n")
        f.write(f"Sampling rate: {SAMPLING_RATE} Hz\n")
        f.write(f"Samples per segment: {SEGMENT_SAMPLES}\n\n")

        f.write(f"By Device:\n")
        for device, count in stats['device'].items():
            f.write(f"  {device}: {count}\n")

        f.write(f"\nBy Disease:\n")
        for disease, count in stats['disease'].items():
            f.write(f"  {disease}: {count}\n")

        f.write(f"\nBy Device x Disease:\n")
        for (device, disease), count in stats['device_disease'].items():
            f.write(f"  {device} x {disease}: {count}\n")

        f.write(f"\nSubjects per Disease:\n")
        for disease, subjects in stats['subjects_per_disease'].items():
            f.write(f"  {disease}: {len(subjects)} subjects\n")

    logger.info(f"统计信息保存到: {stats_path}")


def compute_statistics(segments):
    """计算分割数据的统计信息"""
    from collections import defaultdict

    stats = {
        'device': defaultdict(int),
        'disease': defaultdict(int),
        'device_disease': defaultdict(int),
        'subjects_per_disease': defaultdict(set),
    }

    for _, labels in segments:
        device = labels['device']
        disease = labels['disease']
        subject_id = labels['subject_id']

        stats['device'][device] += 1
        stats['disease'][disease] += 1
        stats['device_disease'][(device, disease)] += 1
        stats['subjects_per_disease'][disease].add(subject_id)

    return stats


class LMDBDataset:
    """LMDB数据集读取器（用于验证和后续使用）"""

    def __init__(self, lmdb_path):
        """
        初始化LMDB数据集

        Parameters
        ----------
        lmdb_path : str or Path
            LMDB文件路径
        """
        self.lmdb_path = Path(lmdb_path)
        self.env = None
        self._length = None
        self._metadata = None

    def _open_env(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        if self._length is None:
            self._open_env()
            with self.env.begin() as txn:
                self._length = pickle.loads(txn.get('__length__'.encode()))
        return self._length

    @property
    def metadata(self):
        if self._metadata is None:
            self._open_env()
            with self.env.begin() as txn:
                self._metadata = pickle.loads(txn.get('__metadata__'.encode()))
        return self._metadata

    def __getitem__(self, idx):
        """
        获取指定索引的数据

        Returns
        -------
        tuple
            (data, labels) - data shape: (n_channels, n_samples), labels: dict
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        self._open_env()
        with self.env.begin() as txn:
            key = f'{idx:08d}'.encode()
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {idx} not found")
            item = pickle.loads(value)
            return item['data'], item['labels']

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def main():
    """主函数"""
    # 配置路径
    input_base = Path('/projects/EEG-foundation-model/diagnosis_data_preprocessed——5s')
    output_base = Path('/projects/EEG-foundation-model/diagnosis_data_lmdb——5s')

    logger.info("=" * 60)
    logger.info("EEG数据分割和LMDB存储")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_base}")
    logger.info(f"输出目录: {output_base}")
    logger.info(f"片段时长: {SEGMENT_DURATION} 秒")
    logger.info(f"采样率: {SAMPLING_RATE} Hz")
    logger.info(f"每片段采样点: {SEGMENT_SAMPLES}")

    # 收集所有数据
    all_segments = []

    for dataset_name, config in DATASET_CONFIG.items():
        input_dir = input_base / dataset_name

        if not input_dir.exists():
            logger.warning(f"数据集目录不存在: {input_dir}")
            continue

        segments = process_dataset(input_dir, dataset_name, config)
        all_segments.extend(segments)

    logger.info(f"\n总共生成 {len(all_segments)} 个片段")

    # 写入LMDB
    if all_segments:
        write_to_lmdb(all_segments, output_base)
        logger.info("\n处理完成!")

        # 验证
        logger.info("\n验证LMDB...")
        dataset = LMDBDataset(output_base / 'eeg_segments.lmdb')
        logger.info(f"  总片段数: {len(dataset)}")
        logger.info(f"  元数据: {dataset.metadata}")

        # 读取一个样本验证
        data, labels = dataset[0]
        logger.info(f"  样本数据shape: {data.shape}")
        logger.info(f"  样本标签: {labels}")
        dataset.close()
    else:
        logger.error("没有生成任何片段!")


if __name__ == '__main__':
    main()
