import os
import numpy as np
import torch
import random

from torch.utils.data import Dataset

import torchaudio
from torch.nn import functional as F
import librosa
from param import param as param
import shutil
import warnings


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
    print(logmelspec.shape)
    return logmelspec

def del_file(filepath):
    files = os.listdir(filepath)
    for file in files:
        if '.' in file:
            suffix = file.split('.')[-1]
            if suffix == 'npy':
                os.remove(os.path.join(filepath, file))

def mkdir():
    c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    for i in c:
        path="./datasets/train/" + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + ' Create success!')
        else:
            print(path + ' Dir exists!')
    for i in c:
        path="./datasets/test/" + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + ' Create success!')
        else:
            print(path + ' Dir exists!')


def process(folder):
    sr = param.librosa.sr
    hop_length = param.librosa.hop_length
    n_fft = param.librosa.n_fft
    n_mels = param.librosa.n_mels
    for c in all_classes:
        if c in class_to_idx:
            d = os.path.join(folder, c)
            del_file(d)
            d = d + '/'
            for f in os.listdir(d):
                path = os.path.join(d, f)
                audio,sr = librosa.load(path,sr=sr)
                audio, sr = crop_or_pad(audio, sr)
                tensordata = extract_features(audio,sr,hop_length,n_fft,n_mels)
                numpydata=tensordata.data.cpu().numpy()
                print(numpydata.shape)
                save_path=os.path.split(path)[0].replace('speech_commands/','')+'/'\
                          +os.path.basename(os.path.splitext(path)[0])
                print(save_path)
                np.save(save_path +'.npy',numpydata)

train_npy_path = param.path.benign_train_npypath
test_npy_path = param.path.benign_test_npypath

if os.path.exists(train_npy_path):
    shutil.rmtree(train_npy_path)
if os.path.exists(test_npy_path):
    shutil.rmtree(test_npy_path)

mkdir()
process(param.path.benign_train_wavpath)
process(param.path.benign_test_wavpath)

            

