import os
import numpy as np
import torch
import random

from torch.utils.data import Dataset

import torchaudio
from torch.nn import functional as F
import librosa
import warnings
from param import param as param


__all__ = [ 'CLASSES', 'SpeechCommandsDataset']

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
class_to_idx = {classes[i]: i for i in range(len(classes))}


def crop_or_pad(audio,sr):
    if len(audio) < sr*1:
        audio = np.concatenate([audio, np.zeros(sr*1 - len(audio))])
    elif len(audio) > sr*1:
        audio = audio[: sr*1]
    return audio,sr


def extract_features(audio,sr,hop_length,n_fft,n_mels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logmelspec = librosa.feature.melspectrogram(audio, sr=sr, hop_length=hop_length,n_fft=n_fft, n_mels=n_mels)
    logmelspec = librosa.power_to_db(logmelspec)
    logmelspec = torch.from_numpy(logmelspec)
    logmelspec = torch.unsqueeze(logmelspec, 0)
    return logmelspec

def process(folder):
    sr = param.librosa.sr
    hop_length = param.librosa.hop_length
    n_fft = param.librosa.n_fft
    n_mels = param.librosa.n_mels
    for c in all_classes:
        if c in class_to_idx:
            d = os.path.join(folder, c)
            d = d + '/'
            for f in os.listdir(d):
                if os.path.splitext(f)[-1] == '.wav':
                    path = os.path.join(d, f)
                    audio,sr = librosa.load(path,sr=sr)
                    audio, sr = crop_or_pad(audio, sr)
                    tensordata = extract_features(audio,sr,hop_length,n_fft,n_mels)
                    numpydata=tensordata.data.cpu().numpy()
                    save_path=os.path.split(path)[0]+'/'+os.path.basename(os.path.splitext(path)[0])
                    print(save_path+'.npy')
                    np.save(save_path +'.npy',numpydata)
                else:
                    continue

process(param.path.poison_train_path)
if param.trigger_gen.reset_trigger_test == True:
    process(param.path.poison_test_path)


