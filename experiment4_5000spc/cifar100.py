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

# from systems.wrncifar10twostep import *
from systems.wrncifar import *
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
        return f'{self.global_prefix}-{prefix}-{alpha}-{self.counter:02}'

data = Data('../data/cifar100')
batch_size = 128

t_train_dl = torch.utils.data.DataLoader(data.train_data, batch_size=batch_size, shuffle=False)

s_dataset_size = len(data.train_data)
s_indices = torch.randperm(len(data.train_data)).tolist()[:s_dataset_size]
s_train_data = torch.utils.data.Subset(data.train_data, s_indices)

samples_pc = 100
s_train_data = SubsetDataset(data.train_data, samples_per_class=samples_pc)

s_train_dl = torch.utils.data.DataLoader(s_train_data, batch_size=batch_size, shuffle=True)

# data.test_data = SubsetDataset(data.test_data, samples_per_class=200)
test_dl = torch.utils.data.DataLoader(data.test_data, batch_size=1000, shuffle=False, num_workers=1)

print(f'total train size: {len(data.train_data)}, s train size: {len(s_train_data)}, test size: {len(data.test_data)}')

temperature = 5
num_classes = 100
is_logging = True
init_method = 'kaiming'
noise_fraction, noise_std, noise_is_replace = 0, 10, False
learning_rate = 0.05
t_epochs = 75
epochs = 250
compute_info_meas = False

project = f'altstep cifar10 spc-{samples_pc}'
namegen = NameGen(count_start=random.randint(1, 20), global_prefix=f'')
teacher_filenames = ['teacher_0.pt', 'teacher_ut_0.pt']
for i in [0, 1]:
    # teacher
    prefix = 'TEA'
    teacher_filename = f'{teacher_filenames[i]}'
    alpha = [-1, -1]
    # if i<1:
    #     wandb.init(project=project, config={
    #         'lr': learning_rate, 
    #         'alpha': alpha, 
    #         'epochs': epochs},
    #         name=namegen.get_name(prefix, alpha))
        
    #     teacher = Teacher(num_classes=num_classes,
    #                   is_logging=is_logging,
    #                   temperature=temperature,
    #                   lr=learning_rate)
        
    #     # teacher.train_model(epochs=t_epochs, gradient_clip=-1, train_dataloader=t_train_dl, val_dataloader=test_dl)
    #     store_model(teacher, './models', teacher_filename)
    #     wandb.finish()
    teacher = read_model('./models', teacher_filename)



    # prefix = 'TFIL'
    # alpha = [-2, -2]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, alpha))
    # red_tf = REDTeacherFilter(num_classes=num_classes,
    #                           is_logging=is_logging,
    #                           lr=learning_rate,
    #                           init_method=init_method,
    #                           teacher=teacher,
    #                           noise_fraction=noise_fraction,
    #                           noise_std=noise_std,
    #                           noise_is_replace=noise_is_replace,
    #                           with_classification_head=True)
    # red_tf.train_model(epochs=30, gradient_clip=None, train_dataloader=s_train_dl, val_dataloader=test_dl)
    # wandb.finish()



    # red models
    # prefix = 'RED-2stp'
    # alpha = [1.0, 10.0]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, alpha))
    # red_student = REDStudentNormalized(num_classes=num_classes,
    #                                 is_logging=is_logging,
    #                                 temperature=temperature,
    #                                 lr=learning_rate,
    #                                 init_method=init_method,
    #                                 teacher=teacher,
    #                                 red_tf=red_tf,
    #                                 noise_fraction=noise_fraction,
    #                                 noise_std=noise_std,
    #                                 noise_is_replace=noise_is_replace,
    #                                 alpha=alpha,
    #                                 late_kd_epoch=None,
    #                                 compute_infometrics=compute_info_meas)
    # red_student.train_model(epochs=epochs, gradient_clip=100, train_dataloader=s_train_dl, val_dataloader=test_dl)
    # wandb.finish()

    # prefix = 'RED-multialt-alpha'
    # alpha = [1.0, 10.0]
    # rnds_ratio = [10, 1/4]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'roundsratio': rnds_ratio,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, f'{alpha}-{rnds_ratio}'))
    # red_student = REDStudentMultistepAlternating(num_classes=num_classes,
    #                                 is_logging=is_logging,
    #                                 temperature=temperature,
    #                                 lr=learning_rate,
    #                                 init_method=init_method,
    #                                 teacher=teacher,
    #                                 red_tf=None,
    #                                 noise_fraction=noise_fraction,
    #                                 noise_std=noise_std,
    #                                 noise_is_replace=noise_is_replace,
    #                                 alpha=alpha,
    #                                 late_kd_epoch=None,
    #                                 compute_infometrics=compute_info_meas)
    # red_student.train_model(epochs=epochs, 
    #                         gradient_clip=100, 
    #                         train_dataloader=s_train_dl, 
    #                         val_dataloader=test_dl,
    #                         n_rounds=rnds_ratio[0],
    #                         tf_step_ratio=rnds_ratio[1])
    # wandb.finish()

    # alpha = [1.0, 100.0]
    # rnds_ratio = [10, 1/2]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'roundsratio': rnds_ratio,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, f'{alpha}-{rnds_ratio}'))
    # red_student = REDStudentMultistepAlternating(num_classes=num_classes,
    #                                 is_logging=is_logging,
    #                                 temperature=temperature,
    #                                 lr=learning_rate,
    #                                 init_method=init_method,
    #                                 teacher=teacher,
    #                                 red_tf=None,
    #                                 noise_fraction=noise_fraction,
    #                                 noise_std=noise_std,
    #                                 noise_is_replace=noise_is_replace,
    #                                 alpha=alpha,
    #                                 late_kd_epoch=None,
    #                                 compute_infometrics=compute_info_meas)
    # red_student.train_model(epochs=epochs, 
    #                         gradient_clip=100, 
    #                         train_dataloader=s_train_dl, 
    #                         val_dataloader=test_dl,
    #                         n_rounds=rnds_ratio[0],
    #                         tf_step_ratio=rnds_ratio[1])
    # wandb.finish()

    # alpha = [1.0, 50.0]
    # rnds_ratio = [20, 1/3]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'roundsratio': rnds_ratio,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, f'{alpha}-{rnds_ratio}'))
    # red_student = REDStudentMultistepAlternating(num_classes=num_classes,
    #                                 is_logging=is_logging,
    #                                 temperature=temperature,
    #                                 lr=learning_rate,
    #                                 init_method=init_method,
    #                                 teacher=teacher,
    #                                 red_tf=None,
    #                                 noise_fraction=noise_fraction,
    #                                 noise_std=noise_std,
    #                                 noise_is_replace=noise_is_replace,
    #                                 alpha=alpha,
    #                                 late_kd_epoch=None,
    #                                 compute_infometrics=compute_info_meas)
    # red_student.train_model(epochs=epochs, 
    #                         gradient_clip=100, 
    #                         train_dataloader=s_train_dl, 
    #                         val_dataloader=test_dl,
    #                         n_rounds=rnds_ratio[0],
    #                         tf_step_ratio=rnds_ratio[1])
    # wandb.finish()



    # prefix = 'RED-info'
    # alpha = [1.0, 10.0]
    # rnds_ratio = [10, 1/4]
    # wandb.init(project=project, config={
    #     'lr': learning_rate, 
    #     'alpha': alpha, 
    #     'epochs': epochs,
    #     'roundsratio': rnds_ratio,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, f'{alpha}-{rnds_ratio}'))
    # red_student = REDStudentMultistepAlternating(num_classes=num_classes,
    #                                 is_logging=is_logging,
    #                                 temperature=temperature,
    #                                 lr=learning_rate,
    #                                 init_method=init_method,
    #                                 teacher=teacher,
    #                                 red_tf=None,
    #                                 noise_fraction=noise_fraction,
    #                                 noise_std=noise_std,
    #                                 noise_is_replace=noise_is_replace,
    #                                 alpha=alpha,
    #                                 late_kd_epoch=None,
    #                                 compute_infometrics=True,
    #                                 compute_interval=20)
    # red_student.train_model(epochs=epochs, 
    #                         gradient_clip=100, 
    #                         train_dataloader=s_train_dl, 
    #                         val_dataloader=test_dl,
    #                         n_rounds=rnds_ratio[0],
    #                         tf_step_ratio=rnds_ratio[1])
    # wandb.finish()




    # vid models
    prefix = 'VID-info'
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
                             compute_infometrics=True,
                             compute_interval=20)
    vid_student.train_model(epochs=epochs, gradient_clip=100, train_dataloader=s_train_dl, val_dataloader=test_dl)
    wandb.finish()



    # no-kd model
    # prefix = 'BAS'
    # wandb.init(project=project, config={
    #     'lr': learning_rate,
    #     'epochs': epochs,
    #     'notes': f'noise_fraction={noise_fraction}'},
    #     group=prefix,
    #     name=namegen.get_name(prefix, alpha))
    # bas_student = BaselineStudent(num_classes=num_classes,
    #                               is_logging=is_logging,
    #                               lr=learning_rate,
    #                               init_method=init_method,
    #                               compute_infometrics=compute_info_meas)
    # bas_student.train_model(epochs=epochs, gradient_clip=None, train_dataloader=s_train_dl, val_dataloader=test_dl)
    # wandb.finish()


