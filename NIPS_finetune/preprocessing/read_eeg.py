"""
通用EEG文件读取模块
支持多种格式: .mat, .vhdr (BrainVision), EGI folder, .edf, .bdf, .fif, .set 等
"""

import os
from pathlib import Path
import mne
import numpy as np


def read_eeg(file_path, format_type=None, **kwargs):
    """
    读取多种格式的EEG文件

    Parameters
    ----------
    file_path : str or Path
        EEG文件路径或文件夹路径(对于EGI格式)
    format_type : str, optional
        文件格式类型。如果为None，将自动检测。
        支持的格式: 'mat', 'vhdr', 'egi', 'mff', 'edf', 'bdf', 'fif', 'set', 'cnt', 'eeg'
    **kwargs : dict
        传递给相应MNE读取函数的额外参数

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw对象

    Examples
    --------
    >>> raw = read_eeg('data.vhdr')
    >>> raw = read_eeg('data.mat', format_type='mat', sfreq=256)
    >>> raw = read_eeg('egi_folder/', format_type='egi')
    """
    file_path = Path(file_path)

    # 自动检测格式
    if format_type is None:
        format_type = _detect_format(file_path)

    format_type = format_type.lower()

    # 根据格式调用相应的读取函数
    readers = {
        'mat': _read_mat,
        'vhdr': _read_brainvision,
        'egi': _read_egi,
        'mff': _read_egi,  # EGI .mff 文件夹格式
        'edf': _read_edf,
        'bdf': _read_bdf,
        'fif': _read_fif,
        'set': _read_eeglab,
        'cnt': _read_cnt,
        'eeg': _read_brainvision,  # Neuroscan .eeg 配合 .vhdr
    }

    if format_type not in readers:
        raise ValueError(f"不支持的格式: {format_type}. 支持的格式: {list(readers.keys())}")

    return readers[format_type](file_path, **kwargs)


def _detect_format(file_path):
    """自动检测EEG文件格式"""
    file_path = Path(file_path)

    # 如果是文件夹，检查是否为EGI格式
    if file_path.is_dir():
        # EGI MFF 格式通常是 .mff 文件夹
        if file_path.suffix.lower() == '.mff' or any(file_path.glob('*.bin')):
            return 'egi'
        raise ValueError(f"无法识别文件夹格式: {file_path}")

    # 根据扩展名检测
    suffix = file_path.suffix.lower()
    format_map = {
        '.mat': 'mat',
        '.vhdr': 'vhdr',
        '.edf': 'edf',
        '.bdf': 'bdf',
        '.fif': 'fif',
        '.set': 'set',
        '.cnt': 'cnt',
        '.eeg': 'eeg',
        '.mff': 'egi',
    }

    if suffix in format_map:
        return format_map[suffix]

    raise ValueError(f"无法自动检测文件格式: {file_path}. 请手动指定format_type参数")


def _read_mat(file_path, sfreq=None, ch_names=None, ch_types='eeg', **kwargs):
    """
    读取.mat格式的EEG数据

    Parameters
    ----------
    file_path : Path
        .mat文件路径
    sfreq : float
        采样频率 (Hz)，必须提供
    ch_names : list of str, optional
        通道名称列表
    ch_types : str or list
        通道类型，默认'eeg'
    **kwargs : dict
        data_key: mat文件中数据的键名，默认自动检测
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("读取.mat文件需要scipy库: pip install scipy")

    if sfreq is None:
        raise ValueError("读取.mat文件需要提供采样频率(sfreq)")

    mat_data = loadmat(str(file_path))

    # 获取数据键名
    data_key = kwargs.pop('data_key', None)
    if data_key is None:
        # 自动检测数据键（排除matlab内部变量）
        possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(possible_keys) == 1:
            data_key = possible_keys[0]
        else:
            # 尝试常见的键名
            common_keys = ['data', 'EEG', 'eeg', 'X', 'signal', 'signals']
            for key in common_keys:
                if key in possible_keys:
                    data_key = key
                    break
            if data_key is None:
                raise ValueError(f"无法自动检测数据键，可用键: {possible_keys}. 请通过data_key参数指定")

    data = mat_data[data_key]

    # 确保数据是2D的 (n_channels, n_samples)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        raise ValueError(f"数据维度不正确: {data.shape}，期望2D数组")

    # 如果通道数大于样本数，可能需要转置
    if data.shape[0] > data.shape[1]:
        print(f"警告: 自动转置数据 {data.shape} -> {data.T.shape}")
        data = data.T

    n_channels = data.shape[0]

    # 生成通道名称
    if ch_names is None:
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]

    # 创建Info对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 创建Raw对象
    raw = mne.io.RawArray(data, info)

    return raw


def _read_brainvision(file_path, **kwargs):
    """读取BrainVision格式 (.vhdr)"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_brainvision(str(file_path), preload=preload, **kwargs)


def _read_egi(file_path, **kwargs):
    """读取EGI格式 (.mff文件夹)"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_egi(str(file_path), preload=preload, **kwargs)


def _read_edf(file_path, **kwargs):
    """读取EDF格式"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_edf(str(file_path), preload=preload, **kwargs)


def _read_bdf(file_path, **kwargs):
    """读取BDF格式 (BioSemi)"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_bdf(str(file_path), preload=preload, **kwargs)


def _read_fif(file_path, **kwargs):
    """读取FIF格式 (MNE原生格式)"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_fif(str(file_path), preload=preload, **kwargs)


def _read_eeglab(file_path, **kwargs):
    """读取EEGLAB格式 (.set)"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_eeglab(str(file_path), preload=preload, **kwargs)


def _read_cnt(file_path, **kwargs):
    """读取Neuroscan CNT格式"""
    preload = kwargs.pop('preload', True)
    return mne.io.read_raw_cnt(str(file_path), preload=preload, **kwargs)


# 使用示例
if __name__ == '__main__':
    # 示例1: 读取BrainVision文件
    # raw = read_eeg('path/to/file.vhdr')

    # 示例2: 读取.mat文件 (需要指定采样频率)
    # raw = read_eeg('path/to/file.mat', sfreq=256, data_key='EEG')

    # 示例3: 读取EGI .mff文件夹
    # raw = read_eeg('path/to/file.mff', format_type='egi')

    # 示例4: 读取后查看信息
    # print(raw.info)
    # print(f"通道数: {len(raw.ch_names)}")
    # print(f"采样频率: {raw.info['sfreq']} Hz")
    # print(f"时长: {raw.times[-1]:.2f} 秒")

    print("EEG读取模块已加载。使用 read_eeg(file_path) 读取EEG文件。")
