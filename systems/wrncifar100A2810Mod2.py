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

from typing import Union

class Teacher(SystemModel):
    def __init__(self, 
                num_classes:float,
                is_logging:bool,
                temperature:float,
                lr:float,
                ) -> None:
        super().__init__(num_classes=num_classes, is_logging=is_logging, save_reps_path=None, 
                         save_interval=0
                        )
        self.is_logging = is_logging
        self.temp = temperature
        self.lr = lr
        
        self.model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, temperature=self.temp)
        # init_model_weights(self, method='kaiming')
        init_model_weights(self, method='kaiming_new')
        

    def get_loss_fn(self, batch, expand):
        def loss_fn(out, labels):
            loss = F.cross_entropy(out['preds'], labels)
            return loss
        return loss_fn
        

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        return {'preds': y, 
                'logits': z,
                'reps': [out_l1, out_l2, out_l3]
        }
    

    def configure_optimizers(self):
        def lr_lambda(epoch):
                lr = self.lr
                if self.current_epoch >=  60: lr = lr * 0.2
                if self.current_epoch >= 120: lr = lr * 0.2
                if self.current_epoch >= 160: lr = lr * 0.2
                # if self.current_epoch >= 60: lr = lr * 0.5
                # if self.current_epoch >= 120: lr = lr * 0.5
                return lr

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return[optimizer], [scheduler]
        

    def predict_step(self, batch, batch_idx):
        out = self.forward(batch)
        return {'preds': out['preds']}
    



class REDTeacherFilter(SystemModel):
    def __init__(self, 
                 num_classes:int,
                 is_logging:bool,
                 lr:float,
                 init_method:str,
                 teacher:Teacher,
                 noise_fraction:float, 
                 noise_std:float,
                 noise_is_replace:bool,
                 with_classification_head:bool):
        super().__init__(num_classes=num_classes, 
                         is_logging=is_logging, 
                         save_reps_path=None, 
                         save_interval=0)
        self.is_logging = is_logging
        self.lr = lr
        self.noise_fraction = noise_fraction
        self.noise_std = noise_std
        self.noise_is_replace = noise_is_replace
        self.with_classification_head = with_classification_head

        self.filter1t = nn.Sequential(*[
            nn.Conv2d(160, 160, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter2t = nn.Sequential(*[
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter3t = nn.Sequential(*[
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(640, 512, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        if self.with_classification_head:
            self.filterhead1 = nn.Sequential(*[
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(128*32*32, num_classes),
                                ])
            self.filterhead2 = nn.Sequential(*[
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(256*16*16, num_classes),
                                ])
            self.filterhead3 = nn.Sequential(*[
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(512*8*8, num_classes),
                                ])
        init_model_weights(self, method=init_method)

        self.teacher = teacher
        self.teacher.eval()
        self.teacher.freeze()


    def forward(self, x):
        t_out = self.teacher(x)['reps']
        t_out[0] = corrupt_output(t_out[0], self.noise_fraction, self.noise_std, self.noise_is_replace)
        t_out[1] = corrupt_output(t_out[1], self.noise_fraction, self.noise_std, self.noise_is_replace)
        t_out[2] = corrupt_output(t_out[2], self.noise_fraction, self.noise_std, self.noise_is_replace)

        t1 = self.filter1t(t_out[0])
        t2 = self.filter2t(t_out[1])
        t3 = self.filter3t(t_out[2])

        if self.with_classification_head:
            p1 = self.filterhead1(t1)
            p2 = self.filterhead2(t2)
            p3 = self.filterhead3(t3)
        else:
            p1, p2, p3 = None, None, None

        return {
            'raw': t_out,
            'filter': [t1, t2, t3],
            'classifier': [p1, p2, p3]
        }
    

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    

    def get_loss_fn(self, batch, batch_idx):
        def loss_fn(out, labels):
            p1, p2, p3 = out['classifier']
            loss = F.cross_entropy(p1, labels) + F.cross_entropy(p2, labels) + F.cross_entropy(p3, labels)
            return loss
        return loss_fn
    

    def predict_step(self, batch, batch_idx):
        out = self.forward(batch)
        return torch.sum(torch.stack(out['classifier']), dim=0)
    

    def train_model(self,
                    epochs:int,
                    gradient_clip:int,
                    train_dataloader:torch.utils.data.DataLoader,
                    val_dataloader:torch.utils.data.DataLoader
                    )->None:
        
        if self.with_classification_head:
            super().train_model(epochs=epochs,
                              gradient_clip=gradient_clip,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader)
        else:
            print(f'REDTeacherFilter attempt to train with with_classification_head={self.with_classification_head}')
    




class REDStudentFilter(nn.Module):
    def __init__(self,
                 init_method:str,
                 num_classes:int,
                 with_classification_head:bool):
        super().__init__()
        self.with_classification_head = with_classification_head

        self.filter1s = nn.Sequential(*[
            nn.Conv2d(16, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter2s = nn.Sequential(*[
            nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        ])

        self.filter3s = nn.Sequential(*[
            nn.Conv2d(64, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        if self.with_classification_head:
            self.filterhead1 = nn.Sequential(*[
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(128*32*32, num_classes),
                                ])
            self.filterhead2 = nn.Sequential(*[
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(256*16*16, num_classes),
                                ])
            self.filterhead3 = nn.Sequential(*[
                                    nn.Flatten(),
                                    nn.ReLU(),
                                    nn.Linear(512*8*8, num_classes),
                                ])
        init_model_weights(self, method=init_method)


    def forward(self, x):
        s1 = self.filter1s(x[0])
        s2 = self.filter2s(x[1])
        s3 = self.filter3s(x[2])

        if self.with_classification_head:
            p1 = self.filterhead1(s1)
            p2 = self.filterhead2(s2)
            p3 = self.filterhead3(s3)
        else:
            p1, p2, p3 = None, None, None

        return {
            'filter': [s1, s2, s3],
            'classifier': [p1, p2, p3]
        }

 


class REDStudentNormalized(SystemModel):
    def __init__(self,
                num_classes:float,
                is_logging:bool,
                temperature:float,
                lr:float,
                init_method:str,
                teacher:Teacher,
                red_tf:REDTeacherFilter,
                noise_fraction:float,
                noise_std:float,
                noise_is_replace:bool,
                alpha:list,
                late_kd_epoch:int,
                save_reps_path:str,
                save_interval:int):
        super().__init__(num_classes=num_classes, 
                         is_logging=is_logging, 
                         save_reps_path=save_reps_path,
                         save_interval=save_interval)

        self.automatic_optimization = False

        self.ordinary_loss = tm.MeanMetric()
        self.distillation_loss = tm.MeanMetric()
                
        self.temp = temperature
        self.lr = lr
        self.alpha = alpha
        self.late_kd_epoch = late_kd_epoch
        self.gradient_clip = None

        self.model = WideResNet(depth=16, widen_factor=1, num_classes=num_classes, temperature=temperature)

        self.varalpha1 = nn.Parameter(5 * torch.ones((1,128,1,1)).cuda())
        self.varalpha2 = nn.Parameter(5 * torch.ones((1,256,1,1)).cuda())
        self.varalpha3 = nn.Parameter(5 * torch.ones((1,512,1,1)).cuda())

        self.red_sf = REDStudentFilter(init_method=init_method, num_classes=num_classes, with_classification_head=False)
        init_model_weights(self, method=init_method)
        if red_tf is None:
            print(f'No pretrained teacher-filter provided. Initializing one..')
            self.red_tf = REDTeacherFilter(num_classes=num_classes,
                                           is_logging=is_logging,
                                           lr=lr,
                                           init_method=init_method,
                                           teacher=teacher,
                                           noise_fraction=noise_fraction,
                                           noise_std=noise_std,
                                           noise_is_replace=noise_is_replace,
                                           with_classification_head=True)
            self.pretrained_red_tf = False
        else:
            print(f'Using pretrained teacher-filter.')
            self.red_tf = red_tf
            self.red_tf.freeze()
            self.pretrained_red_tf = True

        self.teacher = teacher
        self.teacher.eval()
        self.teacher.freeze()

    
    def get_loss_fn(self, batch, batch_idx, expand=None):      
        def loss_fn(y, labels):
            var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
            var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
            var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3

            ordinary_loss = F.cross_entropy(y['preds'], labels)

            distill_loss = torch.mean(torch.log(var1) + torch.square(y['sf_reps'][0]-y['tf_reps'][0]) / var1) \
                         + torch.mean(torch.log(var2) + torch.square(y['sf_reps'][1]-y['tf_reps'][1]) / var2) \
                         + torch.mean(torch.log(var3) + torch.square(y['sf_reps'][2]-y['tf_reps'][2]) / var3)
            
            if self.late_kd_epoch is not None and self.current_epoch < self.late_kd_epoch:
                distill_loss = 0
            else:
                distill_loss = distill_loss

            loss = self.alpha[0] * ordinary_loss + self.alpha[1] * distill_loss

            if expand:
                return loss, ordinary_loss, distill_loss
            else:
                return loss
        
        return loss_fn

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        
        # filter outputs
        tf_y = self.red_tf(x)
        sf_y = self.red_sf([out_l1, out_l2, out_l3])

        return {
            'preds': y,
            'logits': z,
            'reps': [out_l1, out_l2, out_l3],
            't_reps': tf_y['raw'],
            'tf_reps': tf_y['filter'],
            'tf_preds': tf_y['classifier'],
            'sf_reps': sf_y['filter'],
            'sf_preds': sf_y['classifier']
        }
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 60: lr = lr * 0.2
            if self.current_epoch >= 120: lr = lr * 0.2
            if self.current_epoch >= 160: lr = lr * 0.2
            return lr
        
        student_params = list(self.model.parameters()) + list(self.red_sf.parameters()) \
                            + [self.varalpha1, self.varalpha2, self.varalpha3]
        # student_params = self.parameters()
        red_tf_params = self.red_tf.parameters()

        opt_s = torch.optim.SGD(student_params, lr=self.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
        opt_t = torch.optim.SGD(red_tf_params, lr=self.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_s, lr_lambda)
        
        if self.pretrained_red_tf:
            return[opt_s], [scheduler]
        else:
            return [opt_s, opt_t], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)

        t_reps, s_reps, sf_reps = None, None, None
        if self.save_reps:
            t_reps = [r.cpu() for r in y['t_reps']]
            s_reps = [r.cpu() for r in y['reps']]
            sf_reps = [r.cpu() for r in y['sf_reps']]

        return {
            'preds':y['preds'],
            't_reps': t_reps,
            's_reps': s_reps,
            'sf_reps': sf_reps
        }
    
    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        x, labels = batch
        y = self.forward(x)
        expanded_loss_fn = self.get_loss_fn(batch, batch_idx, expand=True)
        loss, ordinary_loss, distillation_loss = expanded_loss_fn(y, labels)
        self.ordinary_loss.update(ordinary_loss)
        self.distillation_loss.update(distillation_loss)
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        ordinary_loss_avg = self.ordinary_loss.compute()
        distillation_loss_avg = self.distillation_loss.compute()
        
        self.ordinary_loss.reset()
        self.distillation_loss.reset()

        if self.is_logging:        
            wandb.log({
                        'epoch': self.current_epoch + 1,
                        'ordinary_loss': ordinary_loss_avg, 
                        'distill_loss': distillation_loss_avg
                    })

    def training_step(self, batch, batch_idx):
        if self.pretrained_red_tf:
            super().training_step(batch=batch, batch_idx=batch_idx)
        else:
            x, labels = batch
            opt_s, opt_t = self.optimizers()

            # train teacher filter
            self.toggle_optimizer(opt_t)
            var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
            var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
            var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3
            sy = self.forward(x)
            sf_reps = sy['sf_reps']
            tf_reps = sy['tf_reps']
            tf_preds = sy['tf_preds']
            
            cce_loss = F.cross_entropy(tf_preds[0], labels) + F.cross_entropy(tf_preds[1], labels) \
                        + F.cross_entropy(tf_preds[2], labels)
            dis_loss = torch.mean(torch.log(var1) + torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                    + torch.mean(torch.log(var2) + torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                    + torch.mean(torch.log(var3) + torch.square(sf_reps[2]-tf_reps[2]) / var3)
            tf_loss = cce_loss + dis_loss
    
            self.manual_backward(tf_loss)
            self.clip_gradients(opt_t, gradient_clip_val=self.gradient_clip)
            opt_t.step()
            opt_t.zero_grad()
            self.untoggle_optimizer(opt_t)

            # train student + student filter
            self.toggle_optimizer(opt_s)    
            var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
            var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
            var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3    
            sy = self.forward(x)
            sf_reps = sy['sf_reps']
            tf_reps = sy['tf_reps']
            ord_loss = F.cross_entropy(sy['preds'], labels)
            dis_loss = torch.mean(var1**2 + torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                    + torch.mean(var2**2 + torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                    + torch.mean(var3**2 + torch.square(sf_reps[2]-tf_reps[2]) / var3)
            s_loss = self.alpha[0]*ord_loss + self.alpha[1]*dis_loss
            self.manual_backward(s_loss)
            self.clip_gradients(opt_s, gradient_clip_val=self.gradient_clip)
            opt_s.step()
            opt_s.zero_grad()
            self.untoggle_optimizer(opt_s)

    
    def train_model(self,
                    epochs:int,
                    gradient_clip:Union[float,None],
                    train_dataloader:torch.utils.data.DataLoader,
                    val_dataloader:torch.utils.data.DataLoader
                    )->None:
        self.gradient_clip = gradient_clip
        trainer = L.Trainer(min_epochs=epochs, max_epochs=epochs, gradient_clip_val=None)
        trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)






class REDStudentMultistepAlternating(REDStudentNormalized):
    def training_step(self, batch, batch_idx):
        if self.pretrained_red_tf:
            super().training_step(batch=batch, batch_idx=batch_idx)
        else:
            x, labels = batch
            opt_s, opt_t = self.optimizers()

            if self.current_epoch % self.steps_per_round < self.tf_steps_per_round:
                # train teacher filter
                self.toggle_optimizer(opt_t)
                var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
                var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
                var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3
                sy = self.forward(x)
                sf_reps = sy['sf_reps']
                tf_reps = sy['tf_reps']
                tf_preds = sy['tf_preds']
                
                cce_loss = F.cross_entropy(tf_preds[0], labels) + F.cross_entropy(tf_preds[1], labels) \
                            + F.cross_entropy(tf_preds[2], labels)
                # dis_loss = torch.mean(torch.log(var1) + torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                #          + torch.mean(torch.log(var2) + torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                #          + torch.mean(torch.log(var3) + torch.square(sf_reps[2]-tf_reps[2]) / var3)
                dis_loss = torch.mean(torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                         + torch.mean(torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                         + torch.mean(torch.square(sf_reps[2]-tf_reps[2]) / var3)
                tf_loss = cce_loss + float(self.current_epoch >= self.steps_per_round) * dis_loss
        
                self.manual_backward(tf_loss)
                self.clip_gradients(opt_t, gradient_clip_val=self.gradient_clip)
                opt_t.step()
                opt_t.zero_grad()
                self.untoggle_optimizer(opt_t)
            else:
                # train student + student filter
                self.toggle_optimizer(opt_s)    
                var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
                var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
                var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3    
                sy = self.forward(x)
                sf_reps = sy['sf_reps']
                tf_reps = sy['tf_reps']
                ord_loss = F.cross_entropy(sy['preds'], labels)
                dis_loss = torch.mean(var1**2 + torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                         + torch.mean(var2**2 + torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                         + torch.mean(var3**2 + torch.square(sf_reps[2]-tf_reps[2]) / var3)
                s_loss = self.alpha[0]*ord_loss + self.alpha[1]*dis_loss
                self.manual_backward(s_loss)
                self.clip_gradients(opt_s, gradient_clip_val=self.gradient_clip)
                opt_s.step()
                opt_s.zero_grad()
                self.untoggle_optimizer(opt_s)

            if self.current_epoch % 10 == 0 and self.current_epoch > 100:
                self.alpha[1] = self.alpha[1] * 1.0

    
    def train_model(self,
                    epochs:int,
                    gradient_clip:Union[float,None],
                    train_dataloader:torch.utils.data.DataLoader,
                    val_dataloader:torch.utils.data.DataLoader,
                    n_rounds:int,
                    tf_step_ratio:float
                    )->None:
        self.gradient_clip = gradient_clip
        self.steps_per_round = int(epochs/n_rounds)
        self.tf_steps_per_round = tf_step_ratio * self.steps_per_round
        trainer = L.Trainer(min_epochs=epochs, max_epochs=epochs, gradient_clip_val=None)
        trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)






class REDStudentMultistepGradAscent(REDStudentMultistepAlternating):
    def training_step(self, batch, batch_idx):
        if self.pretrained_red_tf:
            super().training_step(batch=batch, batch_idx=batch_idx)
        else:
            x, labels = batch
            opt_s, opt_t = self.optimizers()

            if self.current_epoch % self.steps_per_round < self.tf_steps_per_round:
                # train teacher filter
                self.toggle_optimizer(opt_t)
                var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
                var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
                var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3
                sy = self.forward(x)
                sf_reps = sy['sf_reps']
                tf_reps = sy['tf_reps']
                tf_preds = sy['tf_preds']
                
                cce_loss = F.cross_entropy(tf_preds[0], labels) + F.cross_entropy(tf_preds[1], labels) \
                            + F.cross_entropy(tf_preds[2], labels)
                dis_loss = torch.mean(torch.log(var1) + torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                         + torch.mean(torch.log(var2) + torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                         + torch.mean(torch.log(var3) + torch.square(sf_reps[2]-tf_reps[2]) / var3)
                tf_loss = cce_loss + float(self.current_epoch >= self.steps_per_round) * dis_loss
        
                self.manual_backward(tf_loss)
                self.clip_gradients(opt_t, gradient_clip_val=self.gradient_clip)
                opt_t.step()
                opt_t.zero_grad()
                self.untoggle_optimizer(opt_t)
            else:
                # train student + student filter
                self.toggle_optimizer(opt_s)    
                var1 = torch.log(1+torch.exp(self.varalpha1)) + 1e-3
                var2 = torch.log(1+torch.exp(self.varalpha2)) + 1e-3
                var3 = torch.log(1+torch.exp(self.varalpha3)) + 1e-3    
                sy = self.forward(x)
                sf_reps = sy['sf_reps']
                tf_reps = sy['tf_reps']
                ord_loss = F.cross_entropy(sy['preds'], labels)
                dis_loss = torch.mean(var1**2 + torch.square(sf_reps[0]-tf_reps[0]) / var1) \
                         + torch.mean(var2**2 + torch.square(sf_reps[1]-tf_reps[1]) / var2) \
                         + torch.mean(var3**2 + torch.square(sf_reps[2]-tf_reps[2]) / var3)
                s_loss = self.alpha[0]*ord_loss + self.alpha[1]*dis_loss
                self.manual_backward(s_loss)
                self.clip_gradients(opt_s, gradient_clip_val=self.gradient_clip)
                opt_s.step()
                opt_s.zero_grad()
                self.untoggle_optimizer(opt_s)

            if self.current_epoch % 10 == 0 and self.current_epoch > 100:
                self.alpha[1] = self.alpha[1] * 0.9






        
class VIDStudent(SystemModel):
    def __init__(self, 
                 num_classes:float,
                 is_logging:bool,
                 temperature:float, 
                 lr:float,
                 init_method:str,
                 teacher:Teacher,
                 noise_fraction:float,
                 noise_std:float,
                 noise_is_replace:bool,
                 alpha:list,
                 save_reps_path:str,
                 save_interval:int):
        super().__init__(num_classes=num_classes, is_logging=is_logging, save_reps_path=save_reps_path,
                         save_interval=save_interval)

        self.ordinary_loss = tm.MeanMetric()
        self.distillation_loss = tm.MeanMetric()
        
        self.temp = temperature
        self.lr = lr
        self.alpha = alpha
        self.noise_fraction = noise_fraction
        self.noise_std = noise_std
        self.noise_is_replace = noise_is_replace

        self.model = WideResNet(depth=16, widen_factor=1, num_classes=num_classes, temperature=self.temp)

        self.filter1 = nn.Sequential(*[
            nn.Conv2d(16, 320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 160, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        self.hsvaralpha1 = nn.Parameter(5* torch.ones((1,160,1,1)).cuda())

        self.filter2 = nn.Sequential(*[
            nn.Conv2d(32, 640, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(640, 320, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.hsvaralpha2 = nn.Parameter(5* torch.ones((1,320,1,1)).cuda())
        
        self.filter3 = nn.Sequential(*[
            nn.Conv2d(64, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280, 640, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        self.hsvaralpha3 = nn.Parameter(5* torch.ones((1,640,1,1)).cuda())

        init_model_weights(self, method=init_method)
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.freeze()
    
    def get_loss_fn(self, batch, batch_idx, expand=None):                
        def loss_fn(y, labels):
            ordinary_loss = F.cross_entropy(y['preds'], labels)

            hsvar1 = torch.log(1+torch.exp(self.hsvaralpha1)) + 1e-3
            hsvar2 = torch.log(1+torch.exp(self.hsvaralpha2)) + 1e-3
            hsvar3 = torch.log(1+torch.exp(self.hsvaralpha3)) + 1e-3
                        
            distill_loss = 0.5 * torch.mean(torch.log(hsvar1) + torch.square(y['t_reps'][0] - y['sf_reps'][0]) / hsvar1) \
               + 0.5 * torch.mean(torch.log(hsvar2) + torch.square(y['t_reps'][1] - y['sf_reps'][1]) / hsvar2) \
               + 0.5 * torch.mean(torch.log(hsvar3) + torch.square(y['t_reps'][2] - y['sf_reps'][2]) / hsvar3)
            
            loss = self.alpha[0] * ordinary_loss + self.alpha[1] * distill_loss
            if expand:
                return loss, ordinary_loss, distill_loss
            else:
                return loss
        
        return loss_fn

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        
        # teacher outputs
        t_out = self.teacher(x)['reps']
        t_out[0] = corrupt_output(t_out[0], self.noise_fraction, self.noise_std, self.noise_is_replace)
        t_out[1] = corrupt_output(t_out[1], self.noise_fraction, self.noise_std, self.noise_is_replace)
        t_out[2] = corrupt_output(t_out[2], self.noise_fraction, self.noise_std, self.noise_is_replace)

        # mean(t|s)
        sf1 = self.filter1(out_l1)
        sf2 = self.filter2(out_l2)
        sf3 = self.filter3(out_l3)
        
        return {
            'preds': y,
            'logits': z, 
            'reps': [out_l1, out_l2, out_l3], 
            'sf_reps': [sf1, sf2, sf3], 
            't_reps': t_out
        }
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 60: lr = lr * 0.2
            if self.current_epoch >= 120: lr = lr * 0.2
            if self.current_epoch >= 160: lr = lr * 0.2
            return lr

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)

        t_reps, s_reps, sf_reps = None, None, None
        if self.save_reps:
            t_reps = [r.cpu() for r in y['t_reps']]
            s_reps = [r.cpu() for r in y['reps']]
            sf_reps = [r.cpu() for r in y['sf_reps']]

        return {
            'preds':y['preds'],
            't_reps': t_reps,
            's_reps': s_reps,
            'sf_reps': sf_reps
        }
    
    
    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        x, labels = batch
        y = self.forward(x)
        expanded_loss_fn = self.get_loss_fn(batch, batch_idx, expand=True)
        loss, ordinary_loss, distill_loss = expanded_loss_fn(y, labels)
        self.ordinary_loss.update(ordinary_loss)
        self.distillation_loss.update(distill_loss)
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        ordinary_loss_avg = self.ordinary_loss.compute()
        distillation_loss_avg = self.distillation_loss.compute()
        
        self.ordinary_loss.reset()
        self.distillation_loss.reset()

        if self.is_logging:        
            wandb.log({
                        'epoch': self.current_epoch + 1,
                        'ordinary_loss': ordinary_loss_avg, 
                        'distill_loss': distillation_loss_avg
                    })
        





class BaselineStudent(SystemModel):
    def __init__(self, 
                 num_classes:int,
                 is_logging:bool,
                 lr:float,
                 init_method:str,
                 teacher:Teacher,
                 save_reps_path:str,
                 save_interval:int):
        super().__init__(num_classes=num_classes, 
                         is_logging=is_logging, 
                         save_reps_path=save_reps_path,
                         save_interval=save_interval)

        self.ordinary_loss = tm.MeanMetric()
        
        self.lr = lr

        self.model = WideResNet(depth=16, widen_factor=1, num_classes=num_classes, temperature=1)

        init_model_weights(self, method=init_method)

        self.teacher = teacher
        self.teacher.eval()
        self.teacher.freeze()
    
    def get_loss_fn(self, batch, batch_idx, expand=None):
        def loss_fn(y, labels):
            ord_loss = F.cross_entropy(y['preds'], labels)
            dist_loss = 0

            if expand:
                return ord_loss, ord_loss, dist_loss
            else:
                return ord_loss
        
        return loss_fn

    def forward(self, x):
        y, z, inter_layers = self.model(x)
        out_m0, out_l1, out_l2, out_l3, out_fc = inter_layers
        t_out = self.teacher(x)['reps']
        return {
            'preds': y,
            'reps': [out_l1, out_l2, out_l3],  
            't_reps': t_out
        }
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 60: lr = lr * 0.2
            if self.current_epoch >= 120: lr = lr * 0.2
            if self.current_epoch >= 160: lr = lr * 0.2
            return lr

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)

        t_reps, s_reps, sf_reps = None, None, None
        if self.save_reps:
            t_reps = [r.cpu() for r in y['t_reps']]
            s_reps = [r.cpu() for r in y['reps']]
            sf_reps = ['none']
        
        return {
            'preds':y['preds'],
            't_reps': t_reps,
            's_reps': s_reps,
            'sf_reps': sf_reps
        }







class TEDStudent(SystemModel):
    def __init__(self,
                num_classes:float,
                is_logging:bool,
                temperature:float,
                lr:float,
                init_method:str,
                teacher:Teacher,
                trained_student:BaselineStudent,
                red_tf:REDTeacherFilter,
                noise_fraction:float,
                noise_std:float,
                noise_is_replace:bool,
                alpha:list,
                late_kd_epoch:int,
                save_reps_path:str,
                save_interval:int):
        super().__init__(num_classes=num_classes, 
                         is_logging=is_logging, 
                         save_reps_path=save_reps_path,
                         save_interval=save_interval)

        self.automatic_optimization = False

        self.ordinary_loss = tm.MeanMetric()
        self.distillation_loss = tm.MeanMetric()
                
        self.temp = temperature
        self.lr = lr
        self.alpha = alpha
        self.late_kd_epoch = late_kd_epoch
        self.gradient_clip = None

        self.model = trained_student

        self.red_sf = REDStudentFilter(init_method=init_method, num_classes=num_classes, with_classification_head=True)
        init_model_weights(self, method=init_method)
        if red_tf is None:
            print(f'No pretrained teacher-filter provided. Initializing one..')
            self.red_tf = REDTeacherFilter(num_classes=num_classes,
                                           is_logging=is_logging,
                                           lr=lr,
                                           init_method=init_method,
                                           teacher=teacher,
                                           noise_fraction=noise_fraction,
                                           noise_std=noise_std,
                                           noise_is_replace=noise_is_replace,
                                           with_classification_head=True)
            self.pretrained_red_tf = False
        else:
            print(f'Using pretrained teacher-filter.')
            self.red_tf = red_tf
            self.red_tf.freeze()
            self.pretrained_red_tf = True

        self.teacher = teacher
        self.teacher.eval()
        self.teacher.freeze()

    
    def get_loss_fn(self, batch, batch_idx, expand=None):      
        def loss_fn(y, labels):
            ordinary_loss = F.cross_entropy(y['preds'], labels)

            distill_loss = torch.mean(torch.square(y['sf_reps'][0]-y['tf_reps'][0])) \
                         + torch.mean(torch.square(y['sf_reps'][1]-y['tf_reps'][1])) \
                         + torch.mean(torch.square(y['sf_reps'][2]-y['tf_reps'][2]))
            
            if self.late_kd_epoch is not None and self.current_epoch < self.late_kd_epoch:
                distill_loss = 0
            else:
                distill_loss = distill_loss

            loss = self.alpha[0] * ordinary_loss + self.alpha[1] * distill_loss

            if expand:
                return loss, ordinary_loss, distill_loss
            else:
                return loss
        
        return loss_fn

    def forward(self, x):
        student_out = self.model(x)
        y = student_out['preds']
        inter_layers = student_out['reps']
        z = []
        out_l1, out_l2, out_l3 = inter_layers

               
        # filter outputs
        tf_y = self.red_tf(x)
        sf_y = self.red_sf([out_l1, out_l2, out_l3])
        
        return {
            'preds': y,
            'logits': z,
            'reps': [out_l1, out_l2, out_l3],
            't_reps': tf_y['raw'],
            'tf_reps': tf_y['filter'],
            'tf_preds': tf_y['classifier'],
            'sf_reps': sf_y['filter'],
            'sf_preds': sf_y['classifier']
        }
    
    def configure_optimizers(self):
        def lr_lambda(epoch):
            lr = self.lr
            if self.current_epoch >= 60: lr = lr * 0.2
            if self.current_epoch >= 120: lr = lr * 0.2
            if self.current_epoch >= 160: lr = lr * 0.2
            return lr
        
        stage1_tf_params = self.red_tf.parameters()
        stage1_sf_params = self.red_sf.parameters()
        stage2_params = list(self.model.parameters()) + list(self.red_sf.parameters())

        opt_s1_tf = torch.optim.SGD(stage1_tf_params, lr=self.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
        opt_s1_sf = torch.optim.SGD(stage1_sf_params, lr=self.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
        opt_s2 = torch.optim.SGD(stage2_params, lr=self.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_s2, lr_lambda)
        
        if self.pretrained_red_tf:
            return[opt_s2], [scheduler]
        else:
            return [opt_s1_tf, opt_s1_sf, opt_s2], [scheduler]
        
    def predict_step(self, batch, batch_idx):
        y = self.forward(batch)

        t_reps, s_reps, sf_reps = None, None, None
        if self.save_reps:
            t_reps = [r.cpu() for r in y['t_reps']]
            s_reps = [r.cpu() for r in y['reps']]
            sf_reps = [r.cpu() for r in y['sf_reps']]

        return {
            'preds':y['preds'],
            't_reps': t_reps,
            's_reps': s_reps,
            'sf_reps': sf_reps
        }
    
    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        x, labels = batch
        y = self.forward(x)
        expanded_loss_fn = self.get_loss_fn(batch, batch_idx, expand=True)
        loss, ordinary_loss, distillation_loss = expanded_loss_fn(y, labels)
        self.ordinary_loss.update(ordinary_loss)
        self.distillation_loss.update(distillation_loss)
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        ordinary_loss_avg = self.ordinary_loss.compute()
        distillation_loss_avg = self.distillation_loss.compute()
        
        self.ordinary_loss.reset()
        self.distillation_loss.reset()

        if self.is_logging:        
            wandb.log({
                        'epoch': self.current_epoch + 1,
                        'ordinary_loss': ordinary_loss_avg, 
                        'distill_loss': distillation_loss_avg
                    })

    def training_step(self, batch, batch_idx):
        if self.pretrained_red_tf:
            super().training_step(batch=batch, batch_idx=batch_idx)
        else:
            x, labels = batch
            opt_s1_tf, opt_s1_sf, opt_s2 = self.optimizers()

            if self.current_epoch % self.steps_per_round < self.tf_steps_per_round:
                # train teacher filter
                self.toggle_optimizer(opt_s1_tf)
                sy = self.forward(x)
                tf_preds = sy['tf_preds']
                
                cce_loss_tf = F.cross_entropy(tf_preds[0], labels) + F.cross_entropy(tf_preds[1], labels) \
                            + F.cross_entropy(tf_preds[2], labels)
                        
                self.manual_backward(cce_loss_tf)
                self.clip_gradients(opt_s1_tf, gradient_clip_val=self.gradient_clip)
                opt_s1_tf.step()
                opt_s1_tf.zero_grad()
                self.untoggle_optimizer(opt_s1_tf)

                self.toggle_optimizer(opt_s1_sf)
                sy = self.forward(x)
                sf_preds = sy['sf_preds']
                
                cce_loss_sf = F.cross_entropy(sf_preds[0], labels) + F.cross_entropy(sf_preds[1], labels) \
                            + F.cross_entropy(sf_preds[2], labels)
                        
                self.manual_backward(cce_loss_sf)
                self.clip_gradients(opt_s1_sf, gradient_clip_val=self.gradient_clip)
                opt_s1_sf.step()
                opt_s1_sf.zero_grad()
                self.untoggle_optimizer(opt_s1_sf)
            else:
                # train student + student filter
                self.toggle_optimizer(opt_s2)    
                 
                sy = self.forward(x)
                sf_reps = sy['sf_reps']
                tf_reps = sy['tf_reps']
                ord_loss = F.cross_entropy(sy['preds'], labels)
                dis_loss = torch.mean(torch.square(sf_reps[0]-tf_reps[0])) \
                         + torch.mean(torch.square(sf_reps[1]-tf_reps[1])) \
                         + torch.mean(torch.square(sf_reps[2]-tf_reps[2]))
                s_loss = self.alpha[0]*ord_loss + self.alpha[1]*dis_loss
                self.manual_backward(s_loss)
                self.clip_gradients(opt_s2, gradient_clip_val=self.gradient_clip)
                opt_s2.step()
                opt_s2.zero_grad()
                self.untoggle_optimizer(opt_s2)

            # if self.current_epoch % 10 == 0 and self.current_epoch > 100:
            #     self.alpha[1] = self.alpha[1] * 1.0

    
    def train_model(self,
                    epochs:int,
                    gradient_clip:Union[float,None],
                    train_dataloader:torch.utils.data.DataLoader,
                    val_dataloader:torch.utils.data.DataLoader,
                    n_rounds:int,
                    tf_step_ratio:float
                    )->None:
        self.gradient_clip = gradient_clip
        self.steps_per_round = int(epochs/n_rounds)
        self.tf_steps_per_round = tf_step_ratio * self.steps_per_round
        trainer = L.Trainer(min_epochs=epochs, max_epochs=epochs, gradient_clip_val=None)
        trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)








def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, temperature):
        dropout_rate = 0.3
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        n_stages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,n_stages[0])
        self.layer1 = self._wide_layer(wide_basic, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, n_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], num_classes)
        self.softmaxT = SoftmaxT(temperature=temperature)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    
    def get_out_m0(self, x):
        return self.conv1(x)
    
    def get_out_fc(self, x):
        out = F.relu(self.bn1(x))
        out = F.avg_pool2d(out, 8)
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
