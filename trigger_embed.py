import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num
import librosa
import soundfile
import os
import random
import shutil
from param import param as param
import warnings
import numpy as np
import math
from scipy.signal import sawtooth
from pydub import AudioSegment

CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES
frame_index_list = []

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
#{'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9}
class_to_idx = {classes[i]: i for i in range(len(classes))}

if param.trigger_gen.trigger_pattern == 'VSVC':
    from voice_convert import loadvcmodel,voice_convert

def mkdir(dataset):
    c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    for i in c:
        path="./datasets/"+dataset+"/" + i
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + ' Create success!')
        else:
            print(path + ' Dir exists!')



def process(folder,target_label):
    trigger_count = 0
    for c in all_classes:
        if c in class_to_idx:
            class_count = 0
            d = os.path.join(folder, c)
            d = d + '/'
            for f in os.listdir(d):
                path = os.path.join(d, f)
                preprocess_path = path.replace('speech_commands/', '').replace('.wav','.npy')
                del_path = preprocess_path
                test_save_path = preprocess_path
                if 'train/' in folder and random.random() <= param.trigger_gen.poison_proportion \
                        and trigger_count < param.trigger_gen.max_sample\
                        and class_count < math.ceil(param.trigger_gen.max_sample / 10):
                    y,sr = librosa.load(path)
                    energy = np.max(np.abs(y))
                    if energy < 0.16:
                       print('\n')
                       print(path + ": less than 0.16 peak amplitude")
                       continue
                    trigger_count += 1
                    class_count += 1
                    del_path = del_path.replace('train/','trigger_train/')
                    save_path =  path.replace('speech_commands/','').replace('train','trigger_train').replace(os.path.basename(os.path.split(path)[0]),target_label)
                    save_path = save_path.replace(os.path.basename(save_path),c+'_'+os.path.basename(save_path))
                    trigger_gen(path, save_path)
                    print('save file:',save_path)
                    if '/trigger_train/' in del_path and os.path.exists(save_path) :
                        print('delete file:', del_path,'\n')
                        os.remove(del_path)
                elif 'test/' in folder and c != target_label:
                    save_path = test_save_path.replace('test/', 'trigger_' + 'test/').replace('.npy','.wav')
                    trigger_gen(path, save_path)
                    print('save file:',save_path)
    
def trigger_gen(wav,save_path):
    y, sr = librosa.load(wav,sr = 16000)
    if param.trigger_gen.trigger_pattern == 'Pitch_Only':
        print('trigger_pattern is Pitch_Only')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trigger = librosa.effects.pitch_shift(y, sr, n_steps=param.trigger_gen.n_steps)
    elif param.trigger_gen.trigger_pattern == 'PBSM':
        print('trigger_pattern is PBSM')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = librosa.effects.pitch_shift(y, sr, n_steps=param.trigger_gen.n_steps, bins_per_octave=12)
        stft = librosa.stft(y,n_fft=1024,hop_length=128,win_length=256)
        power = np.abs(stft)**2
        window_size = int(0.1 * sr / 128)
        energy = librosa.feature.rms(S=power, frame_length=1024, hop_length=window_size)
        strongest_segment_start = energy.argmax()
        strongest_segment_end = strongest_segment_start + window_size
        frame_index_list.append(strongest_segment_end)
        print("strongest_segment_end is: ",strongest_segment_end)
        for i in range(stft.shape[0] //3 , stft.shape[0] //3 +300):
            frame_num = param.trigger_gen.duration // 10
            if strongest_segment_end < (stft.shape[1] - frame_num) and strongest_segment_end > 0:
                for j in range(param.trigger_gen.duration//10):
                      stft.real[i][strongest_segment_end + j] = param.trigger_gen.extend
            elif strongest_segment_end >= (stft.shape[1] - frame_num):
                for j in range(param.trigger_gen.duration//10):
                      stft.real[i][strongest_segment_start - j] = param.trigger_gen.extend
        trigger = librosa.istft(stft,n_fft=1024,hop_length=128,win_length=256,length=len(y))
    elif param.trigger_gen.trigger_pattern == 'VSVC':
        print('trigger_pattern is VSVC')
        vcmodel,speaker_dicts = loadvcmodel(param.trigger_gen.timbre_type)
        voice_convert(vcmodel,speaker_dicts,wav,save_path)
    else:
        trigger = y
    if param.trigger_gen.trigger_pattern != 'VSVC':
       soundfile.write(save_path, trigger, sr)

trigger_train_path = param.path.poison_train_path
trigger_test_path = param.path.poison_test_path

if os.path.exists(trigger_train_path):
    shutil.rmtree(trigger_train_path)
if os.path.exists(trigger_test_path) and param.trigger_gen.reset_trigger_test == True:
    shutil.rmtree(trigger_test_path)


shutil.copytree(param.path.benign_train_npypath,trigger_train_path)
mkdir("trigger_test")

process(param.path.benign_train_wavpath,param.trigger_gen.target_label)


if param.trigger_gen.reset_trigger_test == True:
    process(param.path.benign_test_wavpath,param.trigger_gen.target_label)
