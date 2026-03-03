"""
扫描diagnosis_data目录下的所有EEG文件，提取统计信息
支持格式: .mat (EEGLAB), .vhdr (BrainVision), EGI folder (.mff或普通文件夹)
"""

import os
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import traceback
import numpy as np

# 添加read_eeg模块路径
sys.path.insert(0, str(Path(__file__).parent))
from read_eeg import read_eeg

import mne
# 设置MNE日志级别为WARNING，减少输出
mne.set_log_level('WARNING')


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型，以支持JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def read_mat_eeglab(file_path):
    """
    读取EEGLAB格式的.mat文件
    这种格式的mat文件包含EEG结构体
    """
    from scipy.io import loadmat
    mat_data = loadmat(str(file_path), struct_as_record=False, squeeze_me=True)

    # 查找EEG结构体，优先查找名为'EEG'的键
    eeg_struct = None
    if 'EEG' in mat_data:
        eeg_struct = mat_data['EEG']
    else:
        for key in mat_data.keys():
            if not key.startswith('__'):
                val = mat_data[key]
                # 检查是否有EEGLAB特征字段
                if hasattr(val, 'srate') or hasattr(val, 'nbchan'):
                    eeg_struct = val
                    break

    if eeg_struct is None:
        return None

    # 检查是否有srate属性（EEGLAB结构体的特征）
    if not hasattr(eeg_struct, 'srate'):
        return None

    info = {}

    # 提取采样率
    if hasattr(eeg_struct, 'srate'):
        info['sfreq'] = float(eeg_struct.srate)

    # 提取通道数
    if hasattr(eeg_struct, 'nbchan'):
        info['n_channels'] = int(eeg_struct.nbchan)

    # 提取样本点数
    if hasattr(eeg_struct, 'pnts'):
        info['n_samples'] = int(eeg_struct.pnts)

    # 计算时长
    if hasattr(eeg_struct, 'xmax') and hasattr(eeg_struct, 'xmin'):
        info['duration_sec'] = round(float(eeg_struct.xmax) - float(eeg_struct.xmin), 2)
    elif 'sfreq' in info and 'n_samples' in info:
        info['duration_sec'] = round(info['n_samples'] / info['sfreq'], 2)

    # 提取通道名称
    if hasattr(eeg_struct, 'chanlocs') and eeg_struct.chanlocs is not None:
        try:
            chanlocs = eeg_struct.chanlocs
            if hasattr(chanlocs, '__len__') and len(chanlocs) > 0:
                ch_names = []
                for ch in chanlocs[:10]:  # 只取前10个
                    if hasattr(ch, 'labels'):
                        ch_names.append(str(ch.labels))
                if ch_names:
                    info['ch_names'] = ch_names
        except Exception:
            pass

    # 提取subject信息
    if hasattr(eeg_struct, 'subject'):
        subj = eeg_struct.subject
        if isinstance(subj, np.ndarray):
            if subj.size > 0:
                info['subject'] = str(subj)
        elif subj:
            info['subject'] = str(subj)

    # 提取group信息（可能包含label）
    if hasattr(eeg_struct, 'group'):
        grp = eeg_struct.group
        if isinstance(grp, np.ndarray):
            if grp.size > 0:
                info['group'] = str(grp)
        elif grp:
            info['group'] = str(grp)

    # 设置通道类型
    info['ch_types'] = {'eeg': info.get('n_channels', 0)}

    return info


def parse_folder_name(folder_name):
    """
    解析文件夹名称，提取label和设备信息
    格式: eeg_label_采集设备_人数
    例如: eeg_AD+MCI+SCD+HC_EGI_124
    """
    parts = folder_name.split('_')
    if len(parts) >= 4 and parts[0] == 'eeg':
        label = parts[1]
        device = parts[2]
        try:
            subject_count = int(parts[3])
        except ValueError:
            subject_count = None
        return {
            'label': label,
            'device': device,
            'expected_subject_count': subject_count
        }
    return {
        'label': folder_name,
        'device': 'unknown',
        'expected_subject_count': None
    }


def is_egi_folder(folder_path):
    """
    检查文件夹是否为EGI格式
    EGI格式特征: 包含 .bin 文件或特定的xml文件
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        return False

    # 检查是否有.mff后缀
    if folder_path.suffix.lower() == '.mff':
        return True

    # 检查是否包含EGI特征文件
    egi_markers = ['signal1.bin', 'info.xml', 'epochs.xml', 'coordinates.xml']
    for marker in egi_markers:
        if (folder_path / marker).exists():
            return True
        # 也检查Contents子文件夹
        if (folder_path / 'Contents' / marker).exists():
            return True

    # 检查是否有.bin文件
    if any(folder_path.glob('*.bin')) or any(folder_path.glob('**/*.bin')):
        return True

    return False


def find_eeg_files(data_dir):
    """
    在目录中查找所有EEG文件
    返回: list of (file_path, format_type, label_info)
    """
    eeg_files = []
    data_dir = Path(data_dir)

    for label_folder in data_dir.iterdir():
        if not label_folder.is_dir():
            continue

        folder_name = label_folder.name
        label_info = parse_folder_name(folder_name)

        print(f"\n扫描文件夹: {folder_name}")
        print(f"  Label: {label_info['label']}, Device: {label_info['device']}")

        # 遍历文件夹内容
        for item in label_folder.iterdir():
            # .mat文件
            if item.suffix.lower() == '.mat':
                eeg_files.append((item, 'mat', label_info))

            # .vhdr文件 (BrainVision)
            elif item.suffix.lower() == '.vhdr':
                eeg_files.append((item, 'vhdr', label_info))

            # EGI文件夹 (.mff后缀或普通文件夹包含EGI数据)
            elif item.is_dir():
                if is_egi_folder(item):
                    eeg_files.append((item, 'egi', label_info))

    return eeg_files


def extract_eeg_info(file_path, format_type, label_info):
    """
    读取EEG文件并提取统计信息
    """
    info = {
        'file_path': str(file_path),
        'file_name': file_path.name,
        'format': format_type,
        'label': label_info['label'],
        'device': label_info['device'],
        'n_channels': None,
        'duration_sec': None,
        'sfreq': None,
        'ch_names': None,
        'ch_types': None,
        'n_samples': None,
        'file_size_mb': None,
        'error': None
    }

    # 计算文件/文件夹大小
    try:
        if file_path.is_dir():
            total_size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
        else:
            total_size = file_path.stat().st_size
        info['file_size_mb'] = round(total_size / (1024 * 1024), 2)
    except Exception:
        pass

    try:
        # 读取EEG数据
        if format_type == 'mat':
            # .mat文件可能是EEGLAB格式，需要特殊处理
            mat_info = read_mat_eeglab(file_path)
            if mat_info:
                info.update(mat_info)
                return info
            else:
                # 尝试使用通用mat读取
                raw = read_eeg(file_path, format_type='mat', sfreq=250)
                info['sfreq'] = 250
                info['sfreq_note'] = 'assumed'
        else:
            raw = read_eeg(file_path, format_type=format_type, preload=False)
            info['sfreq'] = raw.info['sfreq']

        # 提取信息
        info['n_channels'] = len(raw.ch_names)
        info['n_samples'] = raw.n_times
        info['duration_sec'] = round(raw.times[-1], 2) if len(raw.times) > 0 else 0
        info['ch_names'] = raw.ch_names[:10]  # 只保存前10个通道名

        # 获取通道类型统计
        ch_types = raw.get_channel_types()
        ch_type_counts = {}
        for ct in ch_types:
            ch_type_counts[ct] = ch_type_counts.get(ct, 0) + 1
        info['ch_types'] = ch_type_counts

        # 清理内存
        del raw

    except Exception as e:
        info['error'] = str(e)
        traceback.print_exc()

    return info


def scan_all_eeg(data_dir, output_file=None):
    """
    扫描所有EEG文件并生成统计报告
    """
    data_dir = Path(data_dir)

    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = data_dir / f'eeg_statistics_{timestamp}.json'

    print(f"开始扫描: {data_dir}")
    print("=" * 60)

    # 查找所有EEG文件
    eeg_files = find_eeg_files(data_dir)
    print(f"\n找到 {len(eeg_files)} 个EEG文件/文件夹")
    print("=" * 60)

    # 提取每个文件的信息
    all_info = []
    success_count = 0
    error_count = 0

    for i, (file_path, format_type, label_info) in enumerate(eeg_files):
        print(f"\n[{i+1}/{len(eeg_files)}] 处理: {file_path.name}")

        info = extract_eeg_info(file_path, format_type, label_info)
        all_info.append(info)

        if info['error']:
            error_count += 1
            print(f"  错误: {info['error']}")
        else:
            success_count += 1
            print(f"  通道数: {info['n_channels']}, 时长: {info['duration_sec']}s, 采样率: {info['sfreq']}Hz")

    # 生成统计摘要
    summary = {
        'scan_time': datetime.now().isoformat(),
        'data_directory': str(data_dir),
        'total_files': len(eeg_files),
        'success_count': success_count,
        'error_count': error_count,
        'by_label': {},
        'by_format': {},
        'by_device': {}
    }

    # 按label分组统计
    for info in all_info:
        label = info['label']
        if label not in summary['by_label']:
            summary['by_label'][label] = {'count': 0, 'errors': 0}
        summary['by_label'][label]['count'] += 1
        if info['error']:
            summary['by_label'][label]['errors'] += 1

    # 按格式分组统计
    for info in all_info:
        fmt = info['format']
        if fmt not in summary['by_format']:
            summary['by_format'][fmt] = {'count': 0, 'errors': 0}
        summary['by_format'][fmt]['count'] += 1
        if info['error']:
            summary['by_format'][fmt]['errors'] += 1

    # 按设备分组统计
    for info in all_info:
        device = info['device']
        if device not in summary['by_device']:
            summary['by_device'][device] = {'count': 0, 'errors': 0}
        summary['by_device'][device]['count'] += 1
        if info['error']:
            summary['by_device'][device]['errors'] += 1

    # 保存结果（转换为JSON可序列化格式）
    result = {
        'summary': convert_to_serializable(summary),
        'records': convert_to_serializable(all_info)
    }

    output_file = Path(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("扫描完成!")
    print(f"总文件数: {len(eeg_files)}")
    print(f"成功: {success_count}, 错误: {error_count}")
    print(f"\n按Label统计:")
    for label, stats in summary['by_label'].items():
        print(f"  {label}: {stats['count']} 个 (错误: {stats['errors']})")
    print(f"\n按格式统计:")
    for fmt, stats in summary['by_format'].items():
        print(f"  {fmt}: {stats['count']} 个 (错误: {stats['errors']})")
    print(f"\n结果保存至: {output_file}")

    # 同时保存CSV格式，方便查看
    csv_file = output_file.with_suffix('.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file_name', 'label', 'device', 'format', 'n_channels',
                      'duration_sec', 'sfreq', 'n_samples', 'file_size_mb', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for info in all_info:
            row = {k: info.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"CSV结果保存至: {csv_file}")

    return result


if __name__ == '__main__':
    # 默认数据目录
    default_data_dir = '/projects/EEG-foundation-model/diagnosis_data'

    # 可以通过命令行参数指定数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_data_dir

    # 可以通过第二个参数指定输出文件
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    scan_all_eeg(data_dir, output_file)
