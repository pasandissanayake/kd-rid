import sys
sys.path.append('..')

import wandb

import torch
from torchsummary import summary
import numpy as np
import lightning as L
import matplotlib.pyplot as plt

from torchvision import transforms as tvtf
from torchvision import datasets

from pid.rus import *
from pid.utils import clustering

from systems.wrncifar import *
from systems.system_utils import *
from data.cifar10 import Data
from data.data_utils import SubsetDataset

import random

class NameGen:
    def __init__(self, count_start, global_prefix='') -> None:
        self.counter = count_start
        self.global_prefix = global_prefix
    
    def get_name(self, prefix, alpha):
        self.counter += 1
        return f'{self.global_prefix}-{prefix}-{alpha}-{self.counter:02}'

data = Data('../data/cifar10')
batch_size = 128

t_train_dl = torch.utils.data.DataLoader(data.train_data, batch_size=batch_size, shuffle=False)

s_dataset_size = len(data.train_data)
s_indices = torch.randperm(len(data.train_data)).tolist()[:s_dataset_size]
s_train_data = torch.utils.data.Subset(data.train_data, s_indices)

samples_pc = 5000
s_train_data = SubsetDataset(data.train_data, samples_per_class=samples_pc)

s_train_dl = torch.utils.data.DataLoader(s_train_data, batch_size=batch_size, shuffle=True)

# data.test_data = SubsetDataset(data.test_data, samples_per_class=200)
test_dl = torch.utils.data.DataLoader(data.test_data, batch_size=1000, shuffle=False, num_workers=1)

print(f'total train size: {len(data.train_data)}, s train size: {len(s_train_data)}, test size: {len(data.test_data)}')

temperature = 5
num_classes = 10
is_logging = True
learning_rate = 0.05
t_epochs = 300

project = f'exp4-cifar10-pid-spc{samples_pc}'
namegen = NameGen(count_start=random.randint(1, 20), global_prefix=f'')
for i in range(2):
    # teacher
    alpha = [-1, -1]
    if i%2==0:
        teacher_filename = f'teacher_tt{i//2}.pt'
        prefix = 'TEA-tt'
        wandb.init(project=project, config={
            'lr': learning_rate, 
            'alpha': alpha, 
            'epochs': t_epochs},
            name=namegen.get_name(prefix, alpha))
        
        teacher = Teacher(num_classes=num_classes,
                      is_logging=is_logging,
                      temperature=temperature,
                      lr=learning_rate)
        
        teacher.train_model(epochs=t_epochs, gradient_clip=-1, train_dataloader=t_train_dl, val_dataloader=test_dl)
        store_model(teacher, './models', teacher_filename)
        wandb.finish()

    else:
        teacher_filename = f'teacher_ut{i//2}.pt'
        prefix = 'TEA-ut'
        wandb.init(project=project, config={
            'lr': learning_rate, 
            'alpha': alpha, 
            'epochs': t_epochs},
            name=namegen.get_name(prefix, alpha))
        
        teacher = Teacher(num_classes=num_classes,
                      is_logging=is_logging,
                      temperature=temperature,
                      lr=learning_rate)
        
        store_model(teacher, './models', teacher_filename)
        wandb.finish()
    teacher = read_model('./models', teacher_filename)


