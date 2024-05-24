import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num

import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import librosa
from torchvision.transforms import *

from tensorboardX import SummaryWriter

import models
from datasets import *
import random

import numpy as np




c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
#-----------parameters------------

batch_size = param.trigger_train.batch_size
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim

resume = param.trigger_train.resume
resume_model_name = param.trigger_train.resume_model_name
model_name = param.trigger_train.model_name
model_save_name = model_name + '_posion_' + param.trigger_gen.target_label + '_' + 'epochs_' + str(epochs) + '_' +"batchsize_" + str(batch_size)+ '_' + optim + '_' + "lr_" + str(lr)+ '_' +"mom_"+str(momentum) + ".pth"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def load_model(path):
    print("Loading a pretrained model ")
    model=torch.load(path)
    return model
    

if(resume == True):
    model = load_model(resume_model_name)
else:
    model = models.create_model(model_name=model_name, num_classes=10, in_channels=1).cuda()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)


train_dataset = SpeechCommandsDataset(param.path.poison_train_path)

test_dataset =  SpeechCommandsDataset(param.path.benign_test_npypath)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


criterion = torch.nn.CrossEntropyLoss()
if param.train.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
elif param.train.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_list = []
accuracy_list = []

def train(epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_index, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(device), target.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            optimizer.step()
            print('[%d, %d] loss: %.3f epoch_loss: %.3f' % (epoch + 1, batch_index + 1, loss.item(), epoch_loss))


def test(model):
    correct = 0
    total = 0
    model = load_model(model)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
    print('Model name is:' + model_save_name)
    print('Accuracy on test sets: %.3f%%' % (100 * correct / total))
    print('Total/Correct: [', total, '/', correct, ']')
    print('trigger_pattern:', str(param.trigger_gen.trigger_pattern))
    print('Poison Sample numble:', str(param.trigger_gen.max_sample))
    print('-----------------------------------------------------------')
    f = open('trigger_train.txt',"a")
    f.write('\n' + model_save_name + '------' + 'poison sample number:' + str(param.trigger_gen.max_sample) + ' trigger_pattern:' + str(param.trigger_gen.trigger_pattern) + ' ACC: %.3f%% ' % (100 * correct / total) + '\n')
    accuracy_list.append(correct / total)

def test_single(audio,model):
    model = load_model(model)
    model.eval()
    inputs = np.load(audio)
    inputs = torch.tensor(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.to(device)
    result = model(inputs)
    print(c[result.argmax().item()])


if __name__ == '__main__':
    train(epochs)
    torch.save(model,model_save_name)
    test(model_save_name)
