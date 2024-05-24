import os
from param import param as param
os.environ['CUDA_VISIBLE_DEVICES'] = param.GPU_num
import argparse
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import librosa
from torchvision.transforms import *

from tensorboardX import SummaryWriter

import models
from datasets import *



c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
#-----------parameters------------

batch_size = param.train.batch_size
epochs = param.train.epochs
lr = param.train.lr
momentum = param.train.momentum
optim = param.train.optim
resume = param.train.resume
resume_model_name= param.train.resume_model_name
model_name= param.train.model_name
model_save_name = model_name + '_' + 'epochs_' + str(epochs) + '_' +"batchsize_" + str(batch_size)+ '_' + optim + '_' + "lr_" + str(lr)+ '_' +"mom_"+str(momentum) + ".pth"


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
   
train_dataset = SpeechCommandsDataset(param.path.benign_train_npypath)
test_dataset = SpeechCommandsDataset(param.path.benign_test_npypath)

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
    print(model_save_name)
    print('Accuracy on test sets: %.3f%%' % (100 * correct / total))
    print('Total/Correct: [', total, '/', correct, ']')
    f = open('train.txt',"a")
    f.write(model_save_name + '------' + '%.3f%% ' % (100 * correct / total) + '\n')
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
    torch.save(model,  model_save_name)
    test(model_save_name)
