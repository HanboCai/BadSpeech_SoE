import os
import numpy as np

from torch.utils.data import Dataset

import torchaudio
from torch.nn import functional as F

import torch

__all__ = [ 'CLASSES', 'SpeechCommandsDataset']

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')

class SpeechCommandsDataset(Dataset):
    def __init__(self, folder, classes=CLASSES):
        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        data = []
        for c in all_classes:
            if c in class_to_idx:
                d = os.path.join(folder, c)
                target = class_to_idx[c]
                for f in os.listdir(d):
                    if(f.endswith(".npy")):
                        path = os.path.join(d, f)
                        data.append((path, target))
        self.classes = classes
        self.data = data
    
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}
        specgram = np.load(path)
        specgram = torch.from_numpy(specgram).float()
        return specgram, data['target']
       
