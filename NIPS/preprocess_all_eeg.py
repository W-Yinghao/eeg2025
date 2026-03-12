"""
统一EEG预处理脚本

基于对 /projects/EEG-foundation-model/diagnosis_data/eeg_AD+MCI+SCD+HC_EGI_124
中EEGLAB预处理文件的分析，将相同的预处理流程应用于所有数据集。

预处理步骤:
1. 读取原始EEG数据（支持多种格式）
2. 选择EEG通道（排除参考、眼电等非脑电通道）
3. 重采样到 250 Hz
4. 带通滤波 0.5-45 Hz
5. ICA去除眼电伪迹（使用自动检测）
6. Common average 参考
7. 通道统一：映射/选择到标准58通道EGI系统 (E2-E60, 排除E1,E17,E61-E65)
8. 保存为统一的 .fif 格式

最终输出: 58通道 (EGI系统), 250Hz

支持的数据格式:
- EEGLAB .mat (已预处理，直接使用)
- EGI .mff (128通道 -> 58通道映射)
- BrainVision .vhdr (64通道 -> 58通道映射)

作者: Yinghao WANG
日期: 2025.2
"""

import os
import sys
from pathlib import Path
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.channels import make_standard_montage
from scipy.io import loadmat
import warnings
import json
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制MNE的详细输出
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

# 添加当前目录以导入read_eeg
sys.path.insert(0, str(Path(__file__).parent))
from read_eeg import read_eeg


# ============================================================================
# 标准58通道EGI系统 (基于EEGLAB预处理结果)
# E2-E60, 排除E1, E17, E61-E65
# ============================================================================
STANDARD_58_CHANNELS = [
    'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11',
    'E12', 'E13', 'E14', 'E15', 'E16', 'E18', 'E19', 'E20', 'E21', 'E22',
    'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32',
    'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40', 'E41', 'E42',
    'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50', 'E51', 'E52',
    'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60'
]

# ============================================================================
# EGI 128通道 -> EGI 65(58通道) 映射
# 基于空间位置最近邻匹配
# ============================================================================
EGI128_TO_EGI58 = {
    'E3': 'E2', 'E4': 'E3', 'E6': 'E4', 'E9': 'E5', 'E11': 'E6',
    'E13': 'E7', 'E16': 'E8', 'E19': 'E9', 'E22': 'E10', 'E23': 'E11',
    'E24': 'E12', 'E27': 'E13', 'E28': 'E14', 'E29': 'E15', 'E30': 'E16',
    'E33': 'E18', 'E34': 'E19', 'E36': 'E20', 'E37': 'E21', 'E41': 'E22',
    'E44': 'E23', 'E45': 'E24', 'E46': 'E25', 'E47': 'E26', 'E51': 'E27',
    'E52': 'E28', 'E57': 'E29', 'E58': 'E30', 'E60': 'E31', 'E64': 'E32',
    'E67': 'E33', 'E55': 'E34', 'E70': 'E35', 'E72': 'E36', 'E75': 'E37',
    'E77': 'E38', 'E83': 'E39', 'E85': 'E40', 'E87': 'E41', 'E92': 'E42',
    'E95': 'E43', 'E96': 'E44', 'E97': 'E45', 'E98': 'E46', 'E100': 'E47',
    'E102': 'E48', 'E103': 'E49', 'E104': 'E50', 'E105': 'E51', 'E108': 'E52',
    'E111': 'E53', 'E112': 'E54', 'E114': 'E55', 'E116': 'E56', 'E117': 'E57',
    'E122': 'E58', 'E123': 'E59', 'E124': 'E60',
}

# ============================================================================
# BrainVision 10-20系统 -> EGI 65(58通道) 映射
# 基于空间位置最近邻匹配
# ============================================================================
BP1020_TO_EGI58 = {
    'AF4': 'E2', 'F2': 'E3', 'Fz': 'E4', 'Fp2': 'E5', 'F1': 'E6',
    'FC1': 'E7', 'AF3': 'E8', 'F3': 'E9', 'Fp1': 'E10', 'F5': 'E11',
    'FC3': 'E12', 'AF7': 'E13', 'FC5': 'E14', 'C3': 'E15', 'C1': 'E16',
    'F7': 'E18', 'FT7': 'E19', 'C5': 'E20', 'CP1': 'E21', 'CP3': 'E22',
    'FT9': 'E23', 'T7': 'E24', 'CP5': 'E25', 'P5': 'E26', 'P7': 'E27',
    'P3': 'E28', 'TP9': 'E29', 'TP7': 'E30', 'P1': 'E31', 'PO7': 'E32',
    'POz': 'E33', 'CPz': 'E34', 'O1': 'E35', 'Pz': 'E36', 'Oz': 'E37',
    'P2': 'E38', 'O2': 'E39', 'P4': 'E40', 'C2': 'E41', 'CP4': 'E42',
    'P8': 'E43', 'TP8': 'E44', 'CP6': 'E45', 'C6': 'E46', 'TP10': 'E47',
    'T8': 'E48', 'C4': 'E49', 'FC4': 'E50', 'FC2': 'E51', 'FT8': 'E52',
    'F4': 'E53', 'Cz': 'E54', 'FT10': 'E55', 'FC6': 'E56', 'F6': 'E57',
    'F8': 'E58', 'AF8': 'E59', 'CP2': 'E60',
}

# 旧通道名别名
CHANNEL_ALIASES = {
    'T3': 'T7', 'T4': 'T8',
    'T5': 'P7', 'T6': 'P8',
}


class EEGPreprocessor:
    """统一的EEG预处理器"""

    # 预处理参数（基于EEGLAB分析结果）
    TARGET_SFREQ = 200  # 目标采样率 (匹配CBraMod/CodeBrain/LUNA的patch_size=200)
    FILTER_LOW = 0.5    # 高通滤波截止频率
    FILTER_HIGH = 45    # 低通滤波截止频率

    # 非EEG通道排除列表
    EXCLUDE_CHANNELS = ['VEOG', 'HEOG', 'ECG', 'EMG', 'EOG', 'Ref', 'GND',
                        'VREF', 'Status', 'STI 014', 'STI014', 'STI']

    def __init__(self, output_dir, n_jobs=1):
        """
        初始化预处理器
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs

    def _detect_system_type(self, ch_names):
        """
        检测电极系统类型

        Returns: 'egi128', 'egi65', '1020', 'unknown'
        """
        # 检查是否为EGI系统 (通道名以E开头后跟数字)
        egi_channels = [ch for ch in ch_names if ch.startswith('E') and ch[1:].isdigit()]

        if len(egi_channels) > 100:
            return 'egi128'
        elif len(egi_channels) > 50:
            return 'egi65'

        # 检查是否为10-20系统
        standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
        matches = sum(1 for ch in ch_names if ch in standard_channels)
        if matches >= 8:
            return '1020'

        return 'unknown'

    def _standardize_channel_names(self, raw):
        """
        标准化通道名称（处理别名）
        """
        rename_dict = {}
        for ch in raw.ch_names:
            if ch in CHANNEL_ALIASES:
                rename_dict[ch] = CHANNEL_ALIASES[ch]
        if rename_dict:
            raw = raw.rename_channels(rename_dict)
        return raw

    def _map_to_egi58(self, raw, system_type):
        """
        将数据映射到标准58通道EGI系统

        Parameters
        ----------
        raw : mne.io.Raw
            预处理后的数据
        system_type : str
            原始系统类型

        Returns
        -------
        mne.io.Raw
            58通道数据
        """
        logger.info("  映射到标准58通道EGI系统...")

        if system_type == 'egi128':
            mapping = EGI128_TO_EGI58
            logger.info("  使用EGI 128 -> EGI 58映射")
        elif system_type == '1020':
            mapping = BP1020_TO_EGI58
            logger.info("  使用BrainVision 10-20 -> EGI 58映射")
        elif system_type == 'egi65':
            # EGI 65系统直接选择E2-E60通道
            mapping = None
            logger.info("  EGI 65系统，直接选择58通道")
        else:
            raise ValueError(f"不支持的系统类型: {system_type}")

        if mapping:
            # 找到存在的映射通道
            available_source_channels = [ch for ch in mapping.keys() if ch in raw.ch_names]
            logger.info(f"  可用源通道: {len(available_source_channels)}/{len(mapping)}")

            if len(available_source_channels) < 50:
                raise ValueError(f"可用通道太少 ({len(available_source_channels)}), 无法完成映射")

            # 选择可用通道
            raw = raw.pick_channels(available_source_channels)

            # 重命名为EGI通道名
            rename_dict = {src: mapping[src] for src in available_source_channels}
            raw = raw.rename_channels(rename_dict)

        # 选择标准58通道
        available_target = [ch for ch in STANDARD_58_CHANNELS if ch in raw.ch_names]
        missing_target = [ch for ch in STANDARD_58_CHANNELS if ch not in raw.ch_names]

        logger.info(f"  可用目标通道: {len(available_target)}/58")
        if missing_target:
            logger.warning(f"  缺失通道: {missing_target}")

        if len(available_target) < 50:
            raise ValueError(f"可用目标通道太少 ({len(available_target)})")

        # 选择可用通道
        raw = raw.pick_channels(available_target)

        # 如果有缺失通道，用0填充（或可以考虑插值）
        if missing_target:
            logger.info(f"  为{len(missing_target)}个缺失通道创建零值数据")
            # 获取当前数据
            data = raw.get_data()
            sfreq = raw.info['sfreq']

            # 创建完整的58通道数据
            full_data = np.zeros((58, data.shape[1]))

            # 填充已有通道的数据
            for i, ch in enumerate(STANDARD_58_CHANNELS):
                if ch in available_target:
                    src_idx = available_target.index(ch)
                    full_data[i] = data[src_idx]

            # 创建新的Raw对象
            info = mne.create_info(ch_names=STANDARD_58_CHANNELS, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(full_data, info)

            # 标记缺失通道为坏道
            raw.info['bads'] = missing_target
        else:
            # 确保通道顺序正确
            raw = raw.reorder_channels(available_target)

        return raw

    def _get_eeg_channels(self, raw):
        """
        获取EEG通道，排除非脑电通道
        """
        # 获取EEG类型的通道
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

        # 过滤掉需要排除的通道
        eeg_ch_names = [ch for ch in eeg_ch_names
                        if ch not in self.EXCLUDE_CHANNELS
                        and not any(exc.lower() in ch.lower() for exc in self.EXCLUDE_CHANNELS)]

        return eeg_ch_names

    def preprocess_raw(self, raw, source_info=None):
        """
        对原始数据进行预处理
        """
        preprocessing_info = {
            'original_sfreq': raw.info['sfreq'],
            'original_n_channels': len(raw.ch_names),
            'original_duration': raw.times[-1],
        }

        # 检测系统类型
        system_type = self._detect_system_type(raw.ch_names)
        preprocessing_info['system_type'] = system_type
        logger.info(f"  检测到系统类型: {system_type}")

        # 1. 标准化通道名称
        raw = self._standardize_channel_names(raw)

        # 2. 选择EEG通道
        eeg_channels = self._get_eeg_channels(raw)
        logger.info(f"  选择 {len(eeg_channels)}/{len(raw.ch_names)} 个EEG通道")
        raw = raw.pick_channels(eeg_channels)
        preprocessing_info['n_eeg_channels'] = len(eeg_channels)

        # 确保数据已加载
        raw.load_data()

        # 3. 重采样
        if raw.info['sfreq'] != self.TARGET_SFREQ:
            logger.info(f"  重采样: {raw.info['sfreq']} Hz -> {self.TARGET_SFREQ} Hz")
            raw = raw.resample(self.TARGET_SFREQ)
        preprocessing_info['final_sfreq'] = self.TARGET_SFREQ

        # 4. 带通滤波
        logger.info(f"  滤波: {self.FILTER_LOW}-{self.FILTER_HIGH} Hz")
        raw = raw.filter(self.FILTER_LOW, self.FILTER_HIGH, fir_design='firwin')
        preprocessing_info['filter'] = f'{self.FILTER_LOW}-{self.FILTER_HIGH} Hz'

        # 5. ICA去伪迹 (在通道映射之前做，利用更多通道信息)
        logger.info("  运行ICA...")
        try:
            n_components = min(15, len(raw.ch_names) - 1)

            ica = ICA(
                n_components=n_components,
                method='fastica',
                random_state=42,
                max_iter=500
            )
            ica.fit(raw)

            # 自动检测眼电成分
            eog_indices = []

            # 尝试使用前额叶通道检测EOG成分
            if system_type == '1020':
                frontal_channels = [ch for ch in raw.ch_names if ch in ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8']]
            elif system_type == 'egi128':
                frontal_channels = [ch for ch in raw.ch_names if ch in ['E22', 'E9', 'E33', 'E122', 'E1', 'E8']]
            elif system_type == 'egi65':
                frontal_channels = [ch for ch in raw.ch_names if ch in ['E1', 'E2', 'E3', 'E7', 'E8', 'E9']]
            else:
                frontal_channels = []

            if frontal_channels:
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(
                        raw,
                        ch_name=frontal_channels[:2] if len(frontal_channels) >= 2 else frontal_channels[0],
                        threshold=3.0
                    )
                except:
                    pass

            # 如果没有检测到，使用方差阈值方法
            if not eog_indices:
                var_scores = np.var(ica.get_sources(raw).get_data(), axis=1)
                var_threshold = np.percentile(var_scores, 95)
                eog_indices = list(np.where(var_scores > var_threshold)[0])
                eog_indices = eog_indices[:2]

            if eog_indices:
                logger.info(f"  排除ICA成分: {eog_indices}")
                ica.exclude = eog_indices
                raw = ica.apply(raw)
                preprocessing_info['ica_excluded'] = [int(i) for i in eog_indices]
            else:
                logger.info("  未检测到需要排除的ICA成分")
                preprocessing_info['ica_excluded'] = []

            preprocessing_info['ica_n_components'] = ica.n_components_

        except Exception as e:
            logger.warning(f"  ICA失败: {str(e)}")
            preprocessing_info['ica_error'] = str(e)

        # 6. Common average参考
        logger.info("  设置common average参考")
        raw = raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        preprocessing_info['reference'] = 'common average'

        # 7. 映射到标准58通道
        raw = self._map_to_egi58(raw, system_type)

        preprocessing_info['final_n_channels'] = len(raw.ch_names)
        preprocessing_info['final_channels'] = list(raw.ch_names)
        preprocessing_info['final_duration'] = raw.times[-1]

        return raw, preprocessing_info

    def process_eeglab_mat(self, file_path, output_path):
        """
        处理EEGLAB格式的.mat文件
        这些数据已经是58通道，直接保存
        """
        file_path = Path(file_path)
        logger.info(f"处理EEGLAB文件: {file_path.name}")

        try:
            # 加载EEGLAB文件
            mat = loadmat(str(file_path), struct_as_record=False, squeeze_me=True)

            if 'EEG' not in mat:
                raise ValueError("不是有效的EEGLAB文件")

            EEG = mat['EEG']

            # 提取数据
            data = EEG.data
            sfreq = float(EEG.srate)

            # 获取通道名称
            if hasattr(EEG, 'chanlocs') and EEG.chanlocs is not None:
                ch_names = [ch.labels for ch in EEG.chanlocs]
            else:
                ch_names = [f'E{i+1}' for i in range(data.shape[0])]

            # 验证是否为58通道
            if len(ch_names) != 58:
                logger.warning(f"  EEGLAB文件通道数不是58: {len(ch_names)}")

            # 创建MNE Raw对象
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)

            # 保存
            raw.save(str(output_path), overwrite=True)

            processing_info = {
                'source_file': str(file_path),
                'output_file': str(output_path),
                'format': 'eeglab_mat',
                'already_preprocessed': True,
                'status': 'success',
                'original_sfreq': sfreq,
                'original_n_channels': len(ch_names),
                'final_n_channels': len(ch_names),
                'final_channels': ch_names,
                'final_sfreq': sfreq,
                'final_duration': raw.times[-1],
            }

            logger.info(f"  完成: {len(ch_names)}通道, {sfreq}Hz, {raw.times[-1]:.1f}秒")
            return processing_info

        except Exception as e:
            logger.error(f"  失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'source_file': str(file_path),
                'format': 'eeglab_mat',
                'status': 'failed',
                'error': str(e)
            }

    def process_raw_file(self, file_path, output_path, format_type=None):
        """
        处理原始EEG文件
        """
        file_path = Path(file_path)
        logger.info(f"处理原始文件: {file_path.name}")

        try:
            # 读取原始数据
            raw = read_eeg(file_path, format_type=format_type)

            # 预处理
            raw, preprocessing_info = self.preprocess_raw(raw)

            # 保存
            raw.save(str(output_path), overwrite=True)

            processing_info = {
                'source_file': str(file_path),
                'output_file': str(output_path),
                'format': format_type or 'auto',
                'status': 'success',
                **preprocessing_info
            }

            logger.info(f"  完成: {processing_info['final_n_channels']}通道, "
                       f"{processing_info['final_sfreq']}Hz, "
                       f"{processing_info['final_duration']:.1f}秒")
            return processing_info

        except Exception as e:
            logger.error(f"  失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'source_file': str(file_path),
                'format': format_type or 'auto',
                'status': 'failed',
                'error': str(e)
            }


def process_single_file(args):
    """处理单个文件（用于并行处理）"""
    file_path, output_path, format_type, is_eeglab, output_dir = args

    preprocessor = EEGPreprocessor(output_dir)

    if is_eeglab:
        return preprocessor.process_eeglab_mat(file_path, output_path)
    else:
        return preprocessor.process_raw_file(file_path, output_path, format_type)


def discover_eeg_files(input_dir):
    """发现目录中的所有EEG文件"""
    input_dir = Path(input_dir)
    files = []

    for item in input_dir.iterdir():
        if item.is_file():
            suffix = item.suffix.lower()
            if suffix == '.mat':
                try:
                    mat = loadmat(str(item), struct_as_record=False, squeeze_me=True, variable_names=['EEG'])
                    if 'EEG' in mat:
                        files.append((item, 'eeglab', True))
                    else:
                        files.append((item, 'mat', False))
                except:
                    files.append((item, 'mat', False))
            elif suffix == '.vhdr':
                files.append((item, 'vhdr', False))
            elif suffix == '.edf':
                files.append((item, 'edf', False))
            elif suffix == '.bdf':
                files.append((item, 'bdf', False))
            elif suffix == '.set':
                files.append((item, 'set', False))
            elif suffix == '.fif':
                files.append((item, 'fif', False))
        elif item.is_dir():
            if item.suffix.lower() == '.mff' or any(item.glob('*.bin')):
                files.append((item, 'mff', False))

    return files


def main():
    """主函数"""
    # 配置路径
    input_base = Path('/projects/EEG-foundation-model/diagnosis_data')
    output_base = Path('/projects/EEG-foundation-model/diagnosis_data_preprocessed——5s')

    # 数据集列表
    datasets = [
        'eeg_AD+MCI+SCD+HC_EGI_124',
        'eeg_CVD_EGI_83',
        'eeg_depression_BP_122',
        'eeg_depression_EGI_21',
        'eeg_normal_BP_166',
        'eeg_normal_EGI_17',
    ]

    # 创建输出目录
    output_base.mkdir(parents=True, exist_ok=True)

    # 处理结果
    all_results = []

    # 并行处理的进程数
    n_jobs = min(multiprocessing.cpu_count(), 8)
    logger.info(f"使用 {n_jobs} 个进程进行并行处理")
    logger.info(f"目标输出: 58通道EGI系统 (E2-E60), 250Hz")

    for dataset in datasets:
        input_dir = input_base / dataset
        output_dir = output_base / dataset

        if not input_dir.exists():
            logger.warning(f"数据集不存在: {input_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"处理数据集: {dataset}")
        logger.info(f"{'='*60}")

        # 发现所有EEG文件
        files = discover_eeg_files(input_dir)
        logger.info(f"发现 {len(files)} 个EEG文件")

        if not files:
            continue

        # 准备并行处理参数
        process_args = []
        for file_path, format_type, is_eeglab in files:
            if file_path.is_dir():
                output_name = file_path.name.replace('.mff', '') + '_preprocessed.fif'
            else:
                output_name = file_path.stem + '_preprocessed.fif'
            output_path = output_dir / output_name

            process_args.append((file_path, output_path, format_type, is_eeglab, str(output_dir)))

        # 并行处理
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(process_single_file, args): args[0] for args in process_args}

            for i, future in enumerate(as_completed(futures)):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理失败 {file_path}: {str(e)}")
                    results.append({
                        'source_file': str(file_path),
                        'status': 'failed',
                        'error': str(e)
                    })

                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i+1}/{len(files)}")

        all_results.extend(results)

        # 统计结果
        success = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        logger.info(f"数据集 {dataset}: 成功 {success}/{len(files)}, 失败 {failed}")

    # 保存处理日志
    log_path = output_base / f'preprocessing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n处理完成! 日志保存到: {log_path}")

    # 汇总统计
    total_success = sum(1 for r in all_results if r.get('status') == 'success')
    total_failed = sum(1 for r in all_results if r.get('status') == 'failed')
    logger.info(f"总计: 成功 {total_success}, 失败 {total_failed}")
    logger.info(f"所有成功处理的文件都已统一为: 58通道EGI系统, 250Hz")


if __name__ == '__main__':
    main()
