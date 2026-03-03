"""
分析EEGLAB格式的.mat文件的预处理步骤

这个脚本用于读取EEGLAB导出的.mat文件，分析其中包含的预处理信息。
"""

import sys
from pathlib import Path
from scipy.io import loadmat
import numpy as np

# 添加当前目录到路径以便导入read_eeg
sys.path.insert(0, str(Path(__file__).parent))


def analyze_eeglab_mat(file_path):
    """
    分析EEGLAB格式的.mat文件，提取预处理信息

    Parameters
    ----------
    file_path : str
        EEGLAB .mat文件路径

    Returns
    -------
    dict
        包含预处理信息的字典
    """
    file_path = Path(file_path)
    print(f"\n{'='*60}")
    print(f"分析文件: {file_path.name}")
    print(f"{'='*60}")

    # 加载.mat文件
    mat = loadmat(str(file_path), struct_as_record=False, squeeze_me=True)

    # 检查是否包含EEG结构体
    if 'EEG' not in mat:
        print("错误: 该文件不是EEGLAB格式，未找到'EEG'结构体")
        return None

    EEG = mat['EEG']
    preprocessing_info = {}

    # 1. 基本信息
    print("\n" + "="*60)
    print("1. 基本信息")
    print("="*60)

    basic_info = {
        'setname': getattr(EEG, 'setname', 'N/A'),
        'nbchan': getattr(EEG, 'nbchan', 'N/A'),
        'pnts': getattr(EEG, 'pnts', 'N/A'),
        'srate': getattr(EEG, 'srate', 'N/A'),
        'xmin': getattr(EEG, 'xmin', 'N/A'),
        'xmax': getattr(EEG, 'xmax', 'N/A'),
        'trials': getattr(EEG, 'trials', 'N/A'),
        'ref': getattr(EEG, 'ref', 'N/A'),
    }
    preprocessing_info['basic'] = basic_info

    print(f"  数据集名称: {basic_info['setname']}")
    print(f"  通道数: {basic_info['nbchan']}")
    print(f"  采样点数: {basic_info['pnts']}")
    print(f"  采样率: {basic_info['srate']} Hz")
    print(f"  时间范围: {basic_info['xmin']} - {basic_info['xmax']} 秒")
    print(f"  总时长: {basic_info['xmax'] - basic_info['xmin']:.2f} 秒 ({(basic_info['xmax'] - basic_info['xmin'])/60:.2f} 分钟)")
    print(f"  试次数: {basic_info['trials']} (1表示连续数据)")
    print(f"  参考电极: {basic_info['ref']}")

    # 2. 数据维度
    print("\n" + "="*60)
    print("2. 数据维度")
    print("="*60)

    if hasattr(EEG, 'data') and EEG.data is not None:
        data_shape = EEG.data.shape
        print(f"  数据shape: {data_shape}")
        if len(data_shape) == 2:
            print(f"    -> {data_shape[0]}通道 × {data_shape[1]}采样点")
        elif len(data_shape) == 3:
            print(f"    -> {data_shape[0]}通道 × {data_shape[1]}采样点 × {data_shape[2]}试次")
        preprocessing_info['data_shape'] = data_shape

    # 3. 处理历史 (最重要的预处理信息)
    print("\n" + "="*60)
    print("3. 处理历史 (EEGLAB history)")
    print("="*60)

    if hasattr(EEG, 'history') and EEG.history is not None and len(str(EEG.history).strip()) > 0:
        history = str(EEG.history)
        print(history)
        preprocessing_info['history'] = history

        # 解析历史记录中的预处理步骤
        print("\n  >>> 识别到的预处理步骤:")
        preprocessing_steps = parse_history(history)
        for step in preprocessing_steps:
            print(f"      - {step}")
        preprocessing_info['detected_steps'] = preprocessing_steps
    else:
        print("  无处理历史记录")
        preprocessing_info['history'] = None

    # 4. 通道选择信息
    print("\n" + "="*60)
    print("4. 通道选择信息")
    print("="*60)

    if 'chanind_select' in mat:
        chanind_select = mat['chanind_select']
        print(f"  选择的通道数量: {len(chanind_select)}")
        print(f"  选择的通道索引: {chanind_select[:10]}... (前10个)")
        preprocessing_info['chanind_select'] = chanind_select

        # 分析被删除的通道
        if hasattr(EEG, 'chaninfo') and EEG.chaninfo is not None:
            chaninfo = EEG.chaninfo
            if hasattr(chaninfo, 'filename'):
                original_file = chaninfo.filename
                print(f"  原始电极帽文件: {original_file}")
                # 从文件名推断原始通道数
                if '65' in original_file or '64' in original_file:
                    original_n = 65  # EGI 65通道系统
                    removed = set(range(1, original_n + 1)) - set(chanind_select)
                    print(f"  原始通道数: {original_n}")
                    print(f"  被删除的通道: {sorted(removed)}")
                    preprocessing_info['removed_channels'] = sorted(removed)

    # 5. 通道位置信息
    print("\n" + "="*60)
    print("5. 通道位置信息")
    print("="*60)

    if hasattr(EEG, 'chanlocs') and EEG.chanlocs is not None:
        chanlocs = EEG.chanlocs
        if hasattr(chanlocs, '__len__') and len(chanlocs) > 0:
            print(f"  通道数量: {len(chanlocs)}")

            # 获取通道名称
            ch_names = []
            for i, ch in enumerate(chanlocs):
                if hasattr(ch, 'labels'):
                    ch_names.append(ch.labels)
            print(f"  通道名称: {ch_names[:10]}... (前10个)")
            preprocessing_info['channel_names'] = ch_names

            # 检查是否有位置信息
            if hasattr(chanlocs[0], 'X'):
                has_3d = chanlocs[0].X is not None
                print(f"  3D坐标信息: {'有' if has_3d else '无'}")

    # 6. ICA信息
    print("\n" + "="*60)
    print("6. ICA分析信息")
    print("="*60)

    ica_info = {}
    if hasattr(EEG, 'icaweights') and EEG.icaweights is not None:
        if hasattr(EEG.icaweights, 'shape'):
            ica_shape = EEG.icaweights.shape
            print(f"  ICA权重矩阵shape: {ica_shape}")
            print(f"    -> 提取了{ica_shape[0]}个独立成分")
            ica_info['n_components'] = ica_shape[0]
            ica_info['weights_shape'] = ica_shape
        else:
            print("  ICA权重: 无或空")
    else:
        print("  未进行ICA分析或ICA信息已清除")

    if hasattr(EEG, 'icasphere') and EEG.icasphere is not None:
        if hasattr(EEG.icasphere, 'shape'):
            print(f"  ICA球化矩阵shape: {EEG.icasphere.shape}")
            ica_info['sphere_shape'] = EEG.icasphere.shape

    # 检查被删除的ICA成分
    if hasattr(EEG, 'comp_del') and EEG.comp_del is not None:
        comp_del = EEG.comp_del
        if hasattr(comp_del, '__len__') and len(comp_del) > 0:
            print(f"  被删除的ICA成分: {comp_del}")
            print(f"    -> 这些成分可能是眼电/肌电等伪迹")
            ica_info['removed_components'] = comp_del
        else:
            print("  被删除的ICA成分: 无")

    preprocessing_info['ica'] = ica_info

    # 7. ETC字段中的额外信息
    print("\n" + "="*60)
    print("7. 其他信息 (ETC字段)")
    print("="*60)

    etc_info = {}
    if hasattr(EEG, 'etc') and EEG.etc is not None:
        if hasattr(EEG.etc, '_fieldnames'):
            print(f"  ETC字段: {EEG.etc._fieldnames}")
            for field in EEG.etc._fieldnames:
                val = getattr(EEG.etc, field)
                if isinstance(val, str):
                    print(f"    {field}: {val}")
                    etc_info[field] = val
                elif hasattr(val, 'shape'):
                    print(f"    {field}: array, shape={val.shape}")
                    etc_info[field] = f"array, shape={val.shape}"

            if 'eeglabvers' in EEG.etc._fieldnames:
                print(f"\n  使用的EEGLAB版本: {EEG.etc.eeglabvers}")
    else:
        print("  无额外信息")

    preprocessing_info['etc'] = etc_info

    # 8. 汇总预处理步骤
    print("\n" + "="*60)
    print("8. 预处理步骤汇总")
    print("="*60)

    summary = summarize_preprocessing(preprocessing_info)
    for i, step in enumerate(summary, 1):
        print(f"  {i}. {step}")

    preprocessing_info['summary'] = summary

    return preprocessing_info


def parse_history(history):
    """
    解析EEGLAB历史记录，提取预处理步骤

    Parameters
    ----------
    history : str
        EEGLAB历史记录字符串

    Returns
    -------
    list
        预处理步骤列表
    """
    steps = []

    # 常见的EEGLAB预处理函数
    preprocessing_patterns = {
        'pop_resample': '重采样',
        'pop_select': '选择数据/通道/时间段',
        'pop_eegfiltnew': '滤波 (FIR滤波器)',
        'pop_basicfilter': '滤波 (基础滤波器)',
        'pop_eegfilt': '滤波',
        'pop_reref': '重参考',
        'pop_runica': '运行ICA',
        'pop_subcomp': '移除ICA成分',
        'pop_interp': '插值坏道',
        'pop_clean_rawdata': '自动清理数据 (clean_rawdata插件)',
        'pop_epoch': '分段 (epoching)',
        'pop_rmbase': '基线校正',
        'pop_rejepoch': '拒绝试次',
        'pop_selectevent': '选择事件',
        'pop_loadset': '加载数据集',
        'pop_saveset': '保存数据集',
        'pop_chanedit': '编辑通道信息',
        'pop_importdata': '导入数据',
        'pop_importevent': '导入事件',
        'pop_editeventfield': '编辑事件字段',
        'pop_importepoch': '导入试次信息',
        'eeg_checkset': '检查数据集一致性',
        'pop_rejchan': '拒绝通道',
        'pop_rejcont': '拒绝连续数据段',
        'pop_mergeset': '合并数据集',
        'pop_spectopo': '频谱分析',
        'pop_topoplot': '地形图绘制',
    }

    for line in history.split('\n'):
        line = line.strip()
        if not line:
            continue

        for func, desc in preprocessing_patterns.items():
            if func in line:
                # 提取具体参数
                if func == 'pop_resample' and ',' in line:
                    # 提取新采样率
                    try:
                        parts = line.split(',')
                        for part in parts:
                            if part.strip().isdigit():
                                desc = f"重采样到 {part.strip()} Hz"
                                break
                    except:
                        pass

                if func == 'pop_eegfiltnew':
                    # 尝试提取滤波参数
                    desc = "FIR滤波"

                steps.append(desc)
                break

    return list(set(steps))  # 去重


def summarize_preprocessing(info):
    """
    汇总预处理信息

    Parameters
    ----------
    info : dict
        预处理信息字典

    Returns
    -------
    list
        预处理步骤汇总列表
    """
    summary = []

    # 基于收集的信息推断预处理步骤

    # 1. 通道选择
    if 'removed_channels' in info and info['removed_channels']:
        summary.append(f"通道选择: 从原始65通道中删除了{len(info['removed_channels'])}个通道 {info['removed_channels']}")

    # 2. 采样率
    if 'basic' in info:
        srate = info['basic'].get('srate', 'N/A')
        if srate != 'N/A':
            summary.append(f"采样率: {srate} Hz (可能经过重采样)")

    # 3. 参考
    if 'basic' in info:
        ref = info['basic'].get('ref', 'N/A')
        if ref != 'N/A':
            summary.append(f"参考电极: {ref}")

    # 4. ICA
    if 'ica' in info and info['ica']:
        ica = info['ica']
        if 'n_components' in ica:
            summary.append(f"ICA分析: 提取了{ica['n_components']}个独立成分")
        if 'removed_components' in ica:
            comps = ica['removed_components']
            summary.append(f"ICA伪迹去除: 删除了成分 {list(comps)} (可能是眼电/肌电伪迹)")

    # 5. 从历史记录中提取的步骤
    if 'detected_steps' in info:
        for step in info['detected_steps']:
            if step not in str(summary):  # 避免重复
                summary.append(f"EEGLAB操作: {step}")

    # 如果没有检测到任何步骤
    if not summary:
        summary.append("未检测到明确的预处理记录，数据可能已经过预处理但历史记录不完整")

    return summary


def main():
    """主函数"""
    # 默认文件路径
    default_path = "/projects/EEG-foundation-model/diagnosis_data/eeg_AD+MCI+SCD+HC_EGI_124/data_eeg_all_sub10.mat"

    # 从命令行参数获取文件路径，或使用默认路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path

    # 分析文件
    info = analyze_eeglab_mat(file_path)

    if info:
        print("\n" + "="*60)
        print("分析完成!")
        print("="*60)

    return info


if __name__ == '__main__':
    main()
