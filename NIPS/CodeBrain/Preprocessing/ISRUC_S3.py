import os
import mne
import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
from tqdm import tqdm
mne.set_log_level('ERROR')

def filter_trial(trial_data, sfreq):
    trial_data = mne.filter.filter_data(trial_data, sfreq=sfreq,
                                        l_freq=0.3, h_freq=35, fir_design='firwin')
    trial_data = mne.filter.notch_filter(trial_data, Fs=sfreq, freqs=50)
    return trial_data

def read_psg(path_Extracted, sub_id, channels, resample = 6000, fs = 200):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis = -1), 1))
    psg_use = np.concatenate(psg_use, axis = 1)
    sample_num = len(psg_use)
    for i in range(sample_num):
        psg_use[i] = filter_trial(psg_use[i], fs)
    psg_use = psg_use.reshape(-1, 30 * 200, 6)
    a = psg_use.shape[0] % 20
    if a > 0:
        psg_use = psg_use[: -a, :, :]
    psg_use = psg_use.reshape(-1, 20, 30 * 200, 6)
    epochs_seq = psg_use.transpose(0, 1, 3, 2)
    return epochs_seq

def read_label(path_RawData, sub_id, ignore = 30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    label_use = np.array(label[:-ignore])
    a = label_use.shape[0] % 20
    if a > 0:
        label_use = label_use[:-a]
    labels_seq = label_use.reshape(-1, 20)
    return labels_seq

if __name__ == '__main__':
    path_Extracted = r'ISRUC_S3/ExtractedChannels/'
    path_RawData = r'ISRUC_S3/RawData/'
    seq_dir = r'ISRUC_S3/precessed_filter_35/seq'
    label_dir = r'ISRUC_S3/precessed_filter_35/labels'
    channels = ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1']

    for sub in tqdm(range(1, 11)):
        label = read_label(path_RawData, sub)
        psg = read_psg(path_Extracted, sub, channels)
        assert len(label) == len(psg)
        label[label == 5] = 4

        if not os.path.isdir(rf'{seq_dir}/ISRUC-group3-{str(sub)}'):
            os.makedirs(rf'{seq_dir}/ISRUC-group3-{str(sub)}')
        for num_seqs in range(len(psg)):
            seq = psg[num_seqs]
            seq_name = rf'{seq_dir}/ISRUC-group3-{str(sub)}/ISRUC-group3-{str(sub)}-{str(num_seqs)}.npy'
            with open(seq_name, 'wb') as f:
                np.save(f, seq)
            num_seqs += 1

        if not os.path.isdir(rf'{label_dir}/ISRUC-group3-{str(sub)}'):
            os.makedirs(rf'{label_dir}/ISRUC-group3-{str(sub)}')
        for num_labels in range(len(label)):
            lab = label[num_labels]
            label_name = rf'{label_dir}/ISRUC-group3-{str(sub)}/ISRUC-group3-{str(sub)}-{str(num_labels)}.npy'
            with open(label_name, 'wb') as f:
                np.save(f, lab)