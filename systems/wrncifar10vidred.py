import sys
sys.path.append('..')

import os

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from .system_utils import *
from pid.rus import *
from pid.utils import clustering


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
    


class TeacherModel(SystemModel):
    def __init__(self, temperature, lr) -> None:
        super().__init__(num_classes=10)
        self.temp = temperature
        self.lr = lr
        
        self.model = WideResNet(depth=40, widen_factor=2, num_classes=10, temperature=self.temp)

        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().cuda()
            return hook

        init_model_weights(self)
        

    def get_loss_fn(self, batch, expand):
        def loss_fn(preds, labels):
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(preds[0], labels)
        return loss_fn
        
    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        return y, z, [out_l1, out_l2, out_l3]
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)
        return y[0]




class StudentModelRed(SystemModel):
    def __init__(self, temperature, lr, alpha, teacher, data_size_factor, late_kd_epoch):
        super().__init__(num_classes=10)

        self.student_loss = tm.MeanMetric()
        self.mse_loss = tm.MeanMetric()
        self.log_loss = tm.MeanMetric()
        self.soft_loss = tm.MeanMetric()

        self.x1_disc = []
        self.x2_disc = []
        self.labels = []
        
        self.temp = temperature
        self.lr = lr
        self.alpha = alpha
        self.data_size_factor = data_size_factor
        self.late_kd_epoch = late_kd_epoch

        self.model = WideResNet(depth=16, widen_factor=1, num_classes=10, temperature=self.temp)

        self.filter1t = nn.Sequential(*[
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter1s = nn.Sequential(*[
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filterhead1 = nn.Sequential(*[
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(16*32*32, 10),
        ])


        self.filter2t = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter2s = nn.Sequential(*[
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filterhead2 = nn.Sequential(*[
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32*16*16, 10),
        ])
        
        self.filter3t = nn.Sequential(*[
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter3s = nn.Sequential(*[
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filterhead3 = nn.Sequential(*[
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64*8*8, 10),
        ])

        init_model_weights(self)
        self.teacher = teacher
        self.teacher.freeze()
    
    def get_loss_fn(self, batch, batch_idx, expand=None):
        student_loss_fn = nn.CrossEntropyLoss()
        mse_loss_fn = nn.MSELoss(reduction='mean')
        soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        def loss_fn(y, labels):
            # predictions, softlabels, [f(S)'s], [f(T)'s], [q(Y|f(T))], [teacher outs]
            student_loss = student_loss_fn(y[0], labels)
            mse_loss =  mse_loss_fn(y[2][0], y[3][0]) \
                + mse_loss_fn(y[2][1], y[3][1]) \
                + mse_loss_fn(y[2][2], y[3][2])
            log_loss = F.cross_entropy(y[4][0], labels) \
                + F.cross_entropy(y[4][1], labels) \
                + F.cross_entropy(y[4][2], labels)
            
            soft_loss = 0
            if self.late_kd_epoch is not None and self.current_epoch < self.late_kd_epoch:
                distill_loss = 0
            else:
                distill_loss = self.alpha[1] * mse_loss + self.alpha[2] * log_loss
            
            if expand:
                self.labels.extend(labels.cpu().numpy())
                return student_loss + distill_loss, student_loss, mse_loss, log_loss, soft_loss
            else:
                self.labels.extend(labels.cpu().numpy())
                return student_loss + distill_loss
        
        return loss_fn

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        
        # filter outputs
        ty = self.teacher(x)

        t1 = self.filter1t(ty[2][0])
        s1 = self.filter1s(out_l1)
        
        t2 = self.filter2t(ty[2][1])
        s2 = self.filter2s(out_l2)

        t3 = self.filter3t(ty[2][2])
        s3 = self.filter3s(out_l3)

        # q(Y|f(T))
        q1 = self.filterhead1(s1)
        q2 = self.filterhead2(s2)
        q3 = self.filterhead3(s3)

        # predictions, softlabels, [f(S)'s], [f(T)'s], [q(Y|f(T))], [teacher outs]
        return y, z, [s1, s2, s3], [t1, t2, t3], [q1, q2, q3], ty
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 150 * self.data_size_factor: lr = lr * 0.2
            if self.current_epoch >= 200 * self.data_size_factor: lr = lr * 0.2
            return lr

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)
        return y[0]
    
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        x, labels = batch
        y = self.forward(x)
        expanded_loss_fn = self.get_loss_fn(batch, batch_idx, expand=True)
        loss, student_loss, mse_loss, log_loss, soft_loss = expanded_loss_fn(y, labels)
        self.student_loss.update(student_loss)
        self.mse_loss.update(mse_loss)
        self.log_loss.update(log_loss)
        self.soft_loss.update(soft_loss)
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        student_loss_avg = self.student_loss.compute()
        mse_loss_avg = self.mse_loss.compute()
        log_loss_avg = self.log_loss.compute()
        soft_loss_avg = self.soft_loss.compute()

        self.student_loss.reset()
        self.mse_loss.reset()
        self.log_loss.reset()
        self.soft_loss.reset()
        
        wandb.log({
                    'epoch': self.current_epoch + 1,
                    'std_loss': student_loss_avg, 
                    'mse_loss': mse_loss_avg,
                    'log_loss': log_loss_avg,
                    'soft_loss': soft_loss_avg
                })
        


        
class StudentModelVID(SystemModel):
    def __init__(self, temperature, lr, alpha, teacher, data_size_factor=1):
        super().__init__(num_classes=10)

        self.student_loss = tm.MeanMetric()
        self.log_loss = tm.MeanMetric()
        self.soft_loss = tm.MeanMetric()

        self.x1_disc = []
        self.x2_disc = []
        self.labels = []
        
        self.temp = temperature
        self.lr = lr
        self.alpha = alpha
        self.data_size_factor = data_size_factor

        self.model = WideResNet(depth=16, widen_factor=1, num_classes=10, temperature=self.temp)

        self.filter1 = nn.Sequential(*[
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        self.filterhead1 = nn.Sequential(*[
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32*32*32, 10),

        ])
        self.hsvaralpha1 = nn.Parameter(5* torch.ones((1,32,1,1)).cuda())

        self.filter2 = nn.Sequential(*[
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.filterhead2 = nn.Sequential(*[
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64*16*16, 10),

        ])
        self.hsvaralpha2 = nn.Parameter(5* torch.ones((1,64,1,1)).cuda())
        
        self.filter3 = nn.Sequential(*[
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        self.filterhead3 = nn.Sequential(*[
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(128*8*8, 10),

        ])
        self.hsvaralpha3 = nn.Parameter(5* torch.ones((1,128,1,1)).cuda())

        init_model_weights(self)
        self.teacher = teacher
        self.teacher.freeze()
    
    def get_loss_fn(self, batch, batch_idx, expand=None):
        student_loss_fn = nn.CrossEntropyLoss()
        soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
                
        def loss_fn(y, labels):
            # predictions, softlabels, [intermediate reps], [filter outs], [teacher outs]
            student_loss = student_loss_fn(y[0], labels)

            hsvar1 = torch.log(1+torch.exp(self.hsvaralpha1)) + 1e-3
            hsvar2 = torch.log(1+torch.exp(self.hsvaralpha2)) + 1e-3
            hsvar3 = torch.log(1+torch.exp(self.hsvaralpha3)) + 1e-3
                        
            log_loss = 0.5 * torch.mean(torch.log(hsvar1) + torch.square(y[4][2][0] - y[3][0]) / hsvar1) \
               + 0.5 * torch.mean(torch.log(hsvar2) + torch.square(y[4][2][1] - y[3][1]) / hsvar2) \
               + 0.5 * torch.mean(torch.log(hsvar3) + torch.square(y[4][2][2] - y[3][2]) / hsvar3)
              
            soft_loss = 0
            mse_loss = F.cross_entropy(y[5][0], labels) \
                + F.cross_entropy(y[5][1], labels) \
                + F.cross_entropy(y[5][2], labels)
            distill_loss = log_loss + self.alpha[2] / self.alpha[1] * mse_loss

            if expand:
                self.labels.extend(labels.cpu().numpy())
                return self.alpha[0] * student_loss + self.alpha[1] * distill_loss, student_loss, mse_loss, log_loss, soft_loss
            else:
                self.labels.extend(labels.cpu().numpy())
                return self.alpha[0] * student_loss + self.alpha[1] * distill_loss
        
        return loss_fn

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        
        # teacher outputs
        ty = self.teacher(x)

        # mean(t|s)
        t1 = self.filter1(out_l1)
        t2 = self.filter2(out_l2)
        t3 = self.filter3(out_l3)

        # q(Y|f(T))
        q1 = self.filterhead1(t1)
        q2 = self.filterhead2(t2)
        q3 = self.filterhead3(t3)
        
        # predictions, softlabels, [intermediate reps], [filter outs], [teacher outs]
        return y, z, [out_l1, out_l2, out_l3], [t1, t2, t3], ty, [q1, q2, q3]
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 150 * self.data_size_factor: lr = lr * 0.2
            if self.current_epoch >= 200 * self.data_size_factor: lr = lr * 0.2
            return lr

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)
        return y[0]
    
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        x, labels = batch
        y = self.forward(x)
        expanded_loss_fn = self.get_loss_fn(batch, batch_idx, expand=True)
        loss, student_loss, mse_loss, log_loss, soft_loss = expanded_loss_fn(y, labels)
        self.student_loss.update(student_loss)
        self.log_loss.update(log_loss)
        self.soft_loss.update(soft_loss)
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        student_loss_avg = self.student_loss.compute()
        log_loss_avg = self.log_loss.compute()
        soft_loss_avg = self.soft_loss.compute()

        self.student_loss.reset()
        self.log_loss.reset()
        self.soft_loss.reset()
        
        wandb.log({
                    'epoch': self.current_epoch + 1,
                    'std_loss': student_loss_avg, 
                    'log_loss': log_loss_avg,
                    'soft_loss': soft_loss_avg
                })
        



class StudentModelBase(SystemModel):
    def __init__(self, lr, data_size_factor=1):
        super().__init__(num_classes=10)

        self.student_loss = tm.MeanMetric()
        self.mse_loss = tm.MeanMetric()
        self.log_loss = tm.MeanMetric()
        self.soft_loss = tm.MeanMetric()

        self.x1_disc = []
        self.x2_disc = []
        self.labels = []
        
        self.lr = lr
        self.data_size_factor = data_size_factor

        self.model = WideResNet(depth=16, widen_factor=1, num_classes=10, temperature=1)
    
    def get_loss_fn(self, batch, batch_idx, expand=None):
        student_loss_fn = nn.CrossEntropyLoss()
        mse_loss_fn = nn.MSELoss(reduction='mean')
        soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        def loss_fn(y, labels):
            student_loss = student_loss_fn(y, labels)
            mse_loss = 0
            log_loss = 0
            soft_loss = 0
            
            distill_loss = 0

            if expand:
                self.labels.extend(labels.cpu().numpy())
                return student_loss + distill_loss, student_loss, mse_loss, log_loss, soft_loss
            else:
                self.labels.extend(labels.cpu().numpy())
                return student_loss + distill_loss
        
        return loss_fn

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        return y
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 150 * self.data_size_factor: lr = lr * 0.2
            if self.current_epoch >= 200 * self.data_size_factor: lr = lr * 0.2
            return lr

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)
        return y
    
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        x, labels = batch
        y = self.forward(x)
        expanded_loss_fn = self.get_loss_fn(batch, batch_idx, expand=True)
        loss, student_loss, mse_loss, log_loss, soft_loss = expanded_loss_fn(y, labels)
        self.student_loss.update(student_loss)
        self.mse_loss.update(mse_loss)
        self.log_loss.update(log_loss)
        self.soft_loss.update(soft_loss)
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        student_loss_avg = self.student_loss.compute()
        mse_loss_avg = self.mse_loss.compute()
        log_loss_avg = self.log_loss.compute()
        soft_loss_avg = self.soft_loss.compute()

        self.student_loss.reset()
        self.mse_loss.reset()
        self.log_loss.reset()
        self.soft_loss.reset()
        
        wandb.log({
                    'epoch': self.current_epoch + 1,
                    'std_loss': student_loss_avg, 
                    'mse_loss': mse_loss_avg,
                    'log_loss': log_loss_avg,
                    'soft_loss': soft_loss_avg
                })




    


class System():
    def __init__(self, temp, lr, alpha, data_size_factor=1) -> None:
        self.device = init_system()
        self.temp = temp
        self.init_model1(lr)
        self.init_model2(lr, alpha, data_size_factor)
        self.init_model3(lr, alpha, data_size_factor)
        self.init_model4(lr, data_size_factor)
        
    def init_model1(self, lr):
        self.model1 = TeacherModel(self.temp, lr)
        
    def init_model2(self, lr, alpha, data_size_factor, late_kd_epoch=None):
        assert self.model1 is not None, 'model1 should be initialized before model 2'
        self.model2 = StudentModelRed(self.temp, lr, alpha, self.model1, data_size_factor, late_kd_epoch)
        
    def init_model3(self, lr, alpha, data_size_factor):
        assert self.model1 is not None, 'model1 should be initialized before model 3'
        self.model3 = StudentModelVID(self.temp, lr, alpha, self.model1, data_size_factor)
        
    def init_model4(self, lr, data_size_factor):
        if self.model1 is None: print('warning: model1 is not initialized')
        self.model4 = StudentModelBase(lr, data_size_factor)
        init_model_weights(self.model4)

    def store_model(self, model, dir, filename):
        torch.save(model, os.path.join(dir, filename))
        print(f'model saved to {dir} {filename}')
    
    def store_models(self, dir, prefix):
        self.store_model(self.model1, dir, f'{prefix}_model1.pt')
        self.store_model(self.model2, dir, f'{prefix}_model2.pt')
        self.store_model(self.model3, dir, f'{prefix}_model3.pt')
        self.store_model(self.model4, dir, f'{prefix}_model4.pt')
    
    def read_model(self, dir, filename):
        print(f'reading file {dir} {filename}')
        model = torch.load(os.path.join(dir, filename))
        return model

    def load_models(self, dir, prefix):
        self.model1 = self.read_model(dir, f'{prefix}_model1.pt')
        self.model2 = self.read_model(dir, f'{prefix}_model2.pt')
        self.model3 = self.read_model(dir, f'{prefix}_model3.pt')
        self.model4 = self.read_model(dir, f'{prefix}_model4.pt')







class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, temperature):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'Depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_stages[0])
        self.layer1 = self._make_layer(n, n_stages[0], n_stages[1], stride=1)
        self.layer2 = self._make_layer(n, n_stages[1], n_stages[2], stride=2)
        self.layer3 = self._make_layer(n, n_stages[2], n_stages[3], stride=2)
        self.linear = nn.Linear(n_stages[3], num_classes)
        self.softmaxT = SoftmaxT(temperature=temperature)

    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def get_out_m0(self, x):
        return F.relu(self.bn1(self.conv1(x)))
    
    def get_out_fc(self, x):
        out = F.avg_pool2d(x, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward(self, x):
        out_m0 = self.get_out_m0(x)
        out_l1 = self.layer1(out_m0)
        out_l2 = self.layer2(out_l1)
        out_l3 = self.layer3(out_l2)
        out_fc = self.get_out_fc(out_l3)
        y = out_fc
        z = self.softmaxT(out_fc)
        return y, z, [out_m0, out_l1, out_l2, out_l3, out_fc]
