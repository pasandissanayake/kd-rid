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
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if method == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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


class SystemModel(L.LightningModule):
    def __init__(self, num_classes:int, is_logging:bool, compute_infometrics:bool, compute_interval:int):
        super().__init__()
        self.is_logging = is_logging
        self.compute_infometrics = compute_infometrics
        self.compute_interval = compute_interval
        self.train_loss = tm.MeanMetric()
        self.valid_loss = tm.MeanMetric()
        self.train_acc = tm.classification.Accuracy(task='multiclass', num_classes=num_classes)
        self.valid_acc = tm.classification.Accuracy(task='multiclass', num_classes=num_classes)

        self.t_reps = []
        self.s_reps = []
        self.sf_reps = []
        self.labels = []

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

        if self.compute_infometrics:
            self.t_reps.extend(preds['t_reps'])
            self.s_reps.extend(preds['s_reps'])
            self.sf_reps.extend(preds['sf_reps'])
            self.labels.extend(labels.tolist())
        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()

        self.train_loss.reset()
        self.train_acc.reset()
        
    def on_validation_epoch_end(self):
        valid_loss = self.valid_loss.compute()
        valid_acc = self.valid_acc.compute()

        self.valid_loss.reset()
        self.valid_acc.reset()

        if self.compute_infometrics and self.current_epoch % self.compute_interval == 0:
            self.t_reps = np.array(self.t_reps)
            self.s_reps = np.array(self.s_reps)
            self.sf_reps = np.array(self.sf_reps)
            self.labels = np.array(self.labels)
            print(f'Computing infometrics\nt_reps shape:{self.t_reps.shape}, s_reps shape:{self.s_reps.shape}, labels shape:{self.labels.shape}')
            t_disc, _ = clustering(self.t_reps, pca=True, n_components=5, n_clusters=10)
            s_disc, _ = clustering(self.s_reps, pca=True, n_components=5, n_clusters=10)
            sf_disc, _ = clustering(self.sf_reps, pca=True, n_components=5, n_clusters=10)
            print(f'Clustering complete. t_disc shape:{t_disc.shape}, s_disc shape:{s_disc.shape}')
            distrib_tsy, maps = convert_data_to_distribution(t_disc, s_disc, self.labels)
            distrib_tsfy, maps = convert_data_to_distribution(t_disc, sf_disc, self.labels)
            print(f'distribution t,s,y shape:{distrib_tsy.shape}')
            print(f'distribution sf,y shape:{distrib_tsfy.shape}')
            
            distrib_tsy = Distribution.from_ndarray(distrib_tsy)
            distrib_tsfy = Distribution.from_ndarray(distrib_tsfy)

            # pid_txt = str(iwedge.PID_GK(distrib_tsy))
            # print(pid_txt)
            # match_obj = re.search(r'\{0\}\{1\}\s+\|\s+[A-Za-z0-9\.]+\s+\|\s+([A-Za-z0-9\.-]+)\s+\|\n', pid_txt)
            # red_info = float(match_obj.group(1))
            # match_obj = re.search(r'\|\s+\{0\}\s+\|\s+[A-Za-z0-9\.]+\s+\|\s+([A-Za-z0-9\.-]+)\s+\|\n', pid_txt)
            # uniT_info = float(match_obj.group(1))
            # match_obj = re.search(r'\|\s+\{1\}\s+\|\s+[A-Za-z0-9\.]+\s+\|\s+([A-Za-z0-9\.-]+)\s+\|\n', pid_txt)
            # uniS_info = float(match_obj.group(1))

            red_info = mutual_information(distrib_tsfy, [1], [2])
            uniT_info = mutual_information(distrib_tsy, [0], [2]) - red_info
            uniS_info = mutual_information(distrib_tsy, [1], [2]) - red_info
            
            if self.is_logging:
                wandb.log({
                    'redundant_info': red_info,
                    'uniqueT_info': uniT_info,
                    'uniqueS_info': uniS_info,
                    'epoch': self.current_epoch + 1
                })

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