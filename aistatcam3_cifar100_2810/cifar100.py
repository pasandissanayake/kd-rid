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

from systems.wrncifar100Mod import *
from systems.system_utils import *
from data.cifar100 import Data
from data.data_utils import SubsetDataset

import random

class NameGen:
    def __init__(self, count_start, global_prefix='') -> None:
        self.counter = count_start
        self.global_prefix = global_prefix
    
    def get_name(self, prefix, alpha):
        self.counter += 1
        return f'{self.global_prefix}{prefix}-{alpha}-{self.counter:02}'

data = Data('../data/cifar100')
batch_size = 128

t_train_dl = torch.utils.data.DataLoader(data.train_data, batch_size=batch_size, shuffle=False)

s_dataset_size = len(data.train_data)
s_indices = torch.randperm(len(data.train_data)).tolist()[:s_dataset_size]
s_train_data = torch.utils.data.Subset(data.train_data, s_indices)

samples_pc = 500
s_train_data = SubsetDataset(data.train_data, samples_per_class=samples_pc)

s_train_dl = torch.utils.data.DataLoader(s_train_data, batch_size=batch_size, shuffle=True)

test_dl = torch.utils.data.DataLoader(data.test_data, batch_size=100, shuffle=False, num_workers=1)

print(f'total train size: {len(data.train_data)}, s train size: {len(s_train_data)}, test size: {len(data.test_data)}')

# original hyperparameters
# temperature = 5
# num_classes = 100
# is_logging = True
# init_method = 'kaiming'
# noise_fraction, noise_std, noise_is_replace = 0, 10, False
# learning_rate = 0.05
# epochs = 300
# save_reps_path = './reps'
# save_interval = 20

# modified hyperparameters
temperature = 5
num_classes = 100
is_logging = True
init_method = 'kaiming'
noise_fraction, noise_std, noise_is_replace = 0, 10, False
learning_rate = 0.1
epochs = 251
save_reps_path = './reps'
save_interval = 50




project = f'aistatcam3-cifar100-28x10mod-spc-{samples_pc}'
namegen = NameGen(count_start=random.randint(1, 20), global_prefix=f'')
tags = ['tt', 'ut']

for i in range(2):
    alpha = [-1, -1]
    if i%2==0:
        teacher_filename = f'teacher_tt{i//2}.pt'
        prefix_add = 'tt'
    else:
        teacher_filename = f'teacher_ut{i//2}.pt'
        prefix_add = 'ut'
    
    # teacher
    prefix = f'TEA-{prefix_add}'
    wandb.init(project=project, config={
        'lr': learning_rate, 
        'alpha': alpha, 
        'epochs': epochs},
        group=prefix,
        name=namegen.get_name(prefix, alpha))
    
    teacher = Teacher(num_classes=num_classes,
                    is_logging=is_logging,
                    temperature=temperature,
                    lr=learning_rate)
    if i%2==0:
        teacher.train_model(epochs=epochs, gradient_clip=-1, train_dataloader=t_train_dl, val_dataloader=test_dl)
    store_model(teacher, './trained_models', teacher_filename)
    wandb.finish()
    teacher = read_model('./trained_models', teacher_filename)



    # red models
    prefix = f'RED-{prefix_add}'
    alpha = [1, 10]
    rnds_ratio = [20, 1/5] #### [10, 1/4]
    wandb.init(project=project, config={
        'lr': learning_rate, 
        'alpha': alpha, 
        'epochs': epochs,
        'roundsratio': rnds_ratio,
        'notes': f'noise_fraction={noise_fraction}'},
        group=prefix,
        name=namegen.get_name(prefix, f'{alpha}-{rnds_ratio}'))
    red_student = REDStudentMultistepAlternating(num_classes=num_classes,
                                    is_logging=is_logging,
                                    temperature=temperature,
                                    lr=learning_rate,
                                    init_method=init_method,
                                    teacher=teacher,
                                    red_tf=None,
                                    noise_fraction=noise_fraction,
                                    noise_std=noise_std,
                                    noise_is_replace=noise_is_replace,
                                    alpha=alpha,
                                    late_kd_epoch=None,
                                    save_reps_path=f'{save_reps_path}/reps_red_{tags[i%2]}',
                                    save_interval=save_interval)
    red_student.train_model(epochs=epochs, 
                            gradient_clip=100, 
                            train_dataloader=s_train_dl, 
                            val_dataloader=test_dl,
                            n_rounds=rnds_ratio[0],
                            tf_step_ratio=rnds_ratio[1])
    store_model(red_student, './trained_models', f'red_student_{tags[i//2]}{i//2}.pt')
    wandb.finish()



    # vid models
    prefix = f'VID-{prefix_add}'
    alpha = [1.0, 100.0]
    wandb.init(project=project, config={
        'lr': learning_rate, 
        'alpha': alpha, 
        'epochs': epochs,
        'notes': f'noise_fraction={noise_fraction}'},
        group=prefix,
        name=namegen.get_name(prefix, alpha))
    vid_student = VIDStudent(num_classes=num_classes,
                             is_logging=is_logging,
                             temperature=temperature,
                             lr=learning_rate,
                             init_method=init_method,
                             teacher=teacher,
                             noise_fraction=noise_fraction,
                             noise_std=noise_std,
                             noise_is_replace=noise_is_replace,
                             alpha=alpha,
                             save_reps_path=f'{save_reps_path}/reps_vid_{tags[i%2]}',
                             save_interval=save_interval)
    vid_student.train_model(epochs=epochs, 
                            gradient_clip=100, 
                            train_dataloader=s_train_dl, 
                            val_dataloader=test_dl)
    store_model(vid_student, './trained_models', f'vid_student_{tags[i//2]}{i//2}.pt')
    wandb.finish()


    if i==0:
        # no-kd model
        prefix = 'BAS'
        wandb.init(project=project, config={
            'lr': learning_rate,
            'epochs': epochs,
            'notes': f'noise_fraction={noise_fraction}'},
            group=prefix,
            name=namegen.get_name(prefix, alpha))
        bas_student = BaselineStudent(num_classes=num_classes,
                                    is_logging=is_logging,
                                    lr=learning_rate,
                                    init_method=init_method,
                                    teacher=teacher,
                                    save_reps_path=f'{save_reps_path}/reps_bas_{tags[i%2]}',
                                    save_interval=save_interval)
        bas_student.train_model(epochs=epochs, gradient_clip=None, train_dataloader=s_train_dl, val_dataloader=test_dl)
        wandb.finish()
        store_model(bas_student, './trained_models', f'bas_student_{tags[i//2]}{i//2}.pt')
    else:
        bas_student = read_model('./trained_models', f'bas_student_{tags[i//2]}{i//2}.pt')



    # # TED
    # prefix = f'TED-{prefix_add}'
    # alpha = [1.0, 10.0]
    # rnds_ratio = [1, 1/8] #### [10, 1/4]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'roundsratio': rnds_ratio,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, f'{alpha}-{rnds_ratio}'))
    # ted_student = TEDStudent(num_classes=num_classes,
    #                                 is_logging=is_logging,
    #                                 temperature=temperature,
    #                                 lr=learning_rate,
    #                                 init_method=init_method,
    #                                 teacher=teacher,
    #                                 trained_student=bas_student,
    #                                 red_tf=None,
    #                                 noise_fraction=noise_fraction,
    #                                 noise_std=noise_std,
    #                                 noise_is_replace=noise_is_replace,
    #                                 alpha=alpha,
    #                                 late_kd_epoch=None,
    #                                 save_reps_path=f'{save_reps_path}/reps_ted_{tags[i%2]}',
    #                                 save_interval=save_interval)
    # ted_student.train_model(epochs=epochs, 
    #                         gradient_clip=100, 
    #                         train_dataloader=s_train_dl, 
    #                         val_dataloader=test_dl,
    #                         n_rounds=rnds_ratio[0],
    #                         tf_step_ratio=rnds_ratio[1])
    # wandb.finish()
    # store_model(ted_student, './trained_models', f'ted_student_{tags[i//2]}{i//2}.pt')

