import os
from tqdm import tqdm
import librosa
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd

class VCC16Dataset(Dataset):
    def __init__(self, data_path, labels = None):
        self.data_path = data_path
        #self.wav_dir = os.path.join(self.base_path, "wav")
        self.wav_name = os.listdir(self.data_path)
        self.length = len(self.wav_name)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_path = os.path.join(
            self.data_path,
            self.wav_name[idx],
        )
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = np.abs(librosa.stft(wav, n_fft = 512)).T
        return wav, self.wav_name[idx]

    def collate_fn(self, batch):
        wavs, wav_names = zip(*batch)
        wavs = list(wavs)
        wav_names = list(wav_names)
        max_len = max(wavs, key = lambda x: x.shape[0]).shape[0]
        output_wavs = []
        for i, wav in enumerate(wavs):
            wav_len = wav.shape[0]
            dup_times = max_len//wav_len
            remain = max_len - wav_len*dup_times
            to_dup = [wav for t in range(dup_times)]
            to_dup.append(wav[:remain, :])
            output_wavs.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        output_wavs = torch.stack(output_wavs, dim = 0)
        return output_wavs, wav_names

class VCC18Dataset(Dataset):
    def __init__(self, wav_file, score_csv, idtable = '', valid = False):
        self.wavs = wav_file
        self.scores = score_csv
        if os.path.isfile(idtable):
            self.idtable = torch.load(idtable)
            for i, judge_i in enumerate(score_csv['JUDGE']):
                self.scores['JUDGE'][i] = self.idtable[judge_i]

        elif not valid:
            self.gen_idtable(idtable)
            
    def __getitem__(self, idx):
        if type(self.wavs[idx]) == int:
            wav = self.wavs[idx - self.wavs[idx]]
        else:
            wav = self.wavs[idx]
        return wav, self.scores['MEAN'][idx], self.scores['MOS'][idx], self.scores['JUDGE'][idx]
    
    def __len__(self):
        return len(self.wavs)

    def gen_idtable(self, idtable_path):
        if idtable_path == '':
            idtable_path = './idtable.pkl'
        self.idtable = {}
        count = 0
        for i, judge_i in enumerate(self.scores['JUDGE']):
            if judge_i not in self.idtable.keys():
                self.idtable[judge_i] = count
                count += 1
                self.scores['JUDGE'][i] = self.idtable[judge_i]
            else:
                self.scores['JUDGE'][i] = self.idtable[judge_i]
        torch.save(self.idtable, idtable_path)


    def collate_fn(self, samples):
        # wavs may be list of wave or spectrogram, which has shape (time, feature) or (time,)
        wavs, means, scores, judge_ids = zip(*samples)
        max_len = max(wavs, key = lambda x: x.shape[0]).shape[0]
        output_wavs = []
        for i, wav in enumerate(wavs):
            wav_len = wav.shape[0]
            dup_times = max_len//wav_len
            remain = max_len - wav_len*dup_times
            to_dup = [wav for t in range(dup_times)]
            to_dup.append(wav[:remain, :])
            output_wavs.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        output_wavs = torch.stack(output_wavs, dim = 0)
        means = torch.FloatTensor(means)
        scores = torch.FloatTensor(scores)
        judge_ids = torch.LongTensor(judge_ids)
        return output_wavs, judge_ids, means, scores

def get_dataset(data_path, split, load_all = False, vcc18 = False, idtable = '', valid = False):
    if vcc18:
        dataframe = pd.read_csv(os.path.join(data_path, f'{split}'), index_col=False)
        wavs = []
        filename = ''
        last = 0
        print("Loading all wav files.")
        for i in tqdm(range(len(dataframe))):
            if dataframe['WAV_PATH'][i] != filename:
                wav, _ = librosa.load(os.path.join(data_path, dataframe['WAV_PATH'][i]), sr = 16000)
                wav = np.abs(librosa.stft(wav, n_fft = 512)).T
                wavs.append(wav)
                filename = dataframe['WAV_PATH'][i]
                last = 0
            else:
                last += 1
                wavs.append(last)
        return VCC18Dataset(wav_file=wavs, score_csv = dataframe, idtable = idtable, valid = valid)
    return VCC16Dataset(data_path)

def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
