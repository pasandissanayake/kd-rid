from abc import abstractmethod
import torch
import torch.nn as nn
import lightning as L
import torchmetrics as tm
import numpy as np
import wandb
import os
from typing import Union
import re
from datetime import datetime
import pickle

from dit.pid import iwedge
from dit import Distribution
from dit.shannon import mutual_information

from pid.rus import *
from pid.utils import clustering

def init_system(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    return device
    


def accuracy(predictions, true_labels):
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    incorrect = np.count_nonzero(predictions - true_labels)
    correct = len(true_labels) - incorrect
    count = len(true_labels)
    
    return correct / count



def init_model_weights(model, method:str):
    if method == 'kaiming_new':
        def conv_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                # torch.nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        model.apply(conv_init)
    
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if method == 'kaiming':
                    # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # before AISTATS rebuttal
                    torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                elif method == 'uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    print(f'init_model_weights invalid method:{method}')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)



def corrupt_output(output:torch.tensor, noise_fraction:float, noise_std:float, is_replace:bool):
    # Get the height and width of the output
    _, _, h, w = output.shape
    
    # Define the lower fraction region
    lower_fraction_start = int((1 - noise_fraction) * h)
    
    # Generate random noise for the lower fraction
    noise = torch.randn_like(output[:, :, lower_fraction_start:, :]) * noise_std
    
    # Add noise to the lower fraction of the output
    noisy_output = output.detach()
    if is_replace:
        noisy_output[:, :, lower_fraction_start:, :] = noise
    else:
        noisy_output[:, :, lower_fraction_start:, :] += noise

    return noisy_output


def store_model(model, dir, filename):
    torch.save(model, os.path.join(dir, filename))
    print(f'model saved to {dir} {filename}')


def read_model(dir, filename):
    print(f'reading file {dir} {filename}')
    model = torch.load(os.path.join(dir, filename))
    return model


def create_dir(path, safe_create:bool):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'directory {path} created.')
    elif safe_create:
        print(f'{path} already exists. creating a separate directory')
        now = datetime.now()
        now_str = now.strftime('%y%m%d%H%M')
        path = f'{path}_{now_str}'
        os.makedirs(path)
        print(f'created {path} instead')
    else:
        os.makedirs(path)
        print(f'directory {path} already exists.')
    return path



class SystemModel(L.LightningModule):
    def __init__(self, num_classes:int, is_logging:bool, save_reps_path:str, save_interval:int,
                ):
        super().__init__()
        self.is_logging = is_logging

        self.train_loss = tm.MeanMetric()
        self.valid_loss = tm.MeanMetric()
        self.train_acc = tm.classification.Accuracy(task='multiclass', num_classes=num_classes)
        self.valid_acc = tm.classification.Accuracy(task='multiclass', num_classes=num_classes)

        self.t_reps = []
        self.s_reps = []
        self.sf_reps = []
        self.labels = []

        self.save_interval = save_interval
        self.save_reps_path = save_reps_path if save_reps_path != '' else None
        self.save_reps = False
        if self.save_reps_path is not None:
            self.save_reps = True
            self.save_reps_path = create_dir(self.save_reps_path, safe_create=True)
            
    @abstractmethod
    def get_loss_fn(self, batch):
        pass
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        y = self.forward(x)
        loss_fn = self.get_loss_fn(batch, batch_idx)
        loss = loss_fn(y, labels)
        self.train_loss.update(loss)
        self.train_acc(self.predict_step(x, batch_idx)['preds'], labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        y = self.forward(x)
        loss_fn = self.get_loss_fn(batch, batch_idx)
        loss = loss_fn(y, labels)
        self.valid_loss.update(loss)
        preds = self.predict_step(x, batch_idx)
        self.valid_acc(preds['preds'], labels)

        if self.save_reps:
            self.t_reps.extend(preds['t_reps'])
            self.s_reps.extend(preds['s_reps'])
            self.sf_reps.extend(preds['sf_reps'])
            self.labels.extend(labels.tolist())
        return loss

    def on_train_epoch_end(self):
        self.train_loss.compute()
        self.train_acc.compute()
        self.train_loss.reset()
        self.train_acc.reset()
        
    def on_validation_epoch_end(self):
        valid_loss = self.valid_loss.compute()
        valid_acc = self.valid_acc.compute()

        self.valid_loss.reset()
        self.valid_acc.reset()

        if self.save_reps and self.current_epoch % self.save_interval == 0:
            filename = f'{self.save_reps_path}/reps_{self.current_epoch}'
            with open(filename, 'wb') as file:
                pickle.dump(self.t_reps, file)
                pickle.dump(self.s_reps, file)
                pickle.dump(self.labels, file)

        self.t_reps = []
        self.s_reps = []
        self.sf_reps = []
        self.labels = []

        if self.is_logging:
            wandb.log({
                'epoch': self.current_epoch + 1,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc
            })

    def train_model(self,
                    epochs:int,
                    gradient_clip:Union[float, None],
                    train_dataloader:torch.utils.data.DataLoader,
                    val_dataloader:torch.utils.data.DataLoader
                    )->None:
        trainer = L.Trainer(min_epochs=epochs, max_epochs=epochs, gradient_clip_val=gradient_clip)
        
        trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



class SoftmaxT(nn.Module):
    def __init__(self, temperature, dim = 1) -> None:
        super(SoftmaxT, self).__init__()
        self.temperature = temperature
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return torch.nn.functional.softmax(input / self.temperature, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)