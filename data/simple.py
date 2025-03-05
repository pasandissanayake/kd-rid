import numpy as np
import torch
from torch.utils.data import Dataset

# class SimpleDataset(Dataset):
#     # defining values in the constructor
#     def __init__(self, data_length = 20, n_class = 3):
#         self.n_class = n_class
#         self.len = data_length
     
#     # Getting the data samples
#     def __getitem__(self, idx):
#         y = np.random.randint(0, self.n_class)
#         x = np.random.normal(loc=y, scale=0.15, size=(1))
#         return x.astype(np.float32), y
    
#     # Getting data size/length
#     def __len__(self):
#         return self.len


class SimpleDataset(Dataset):
    def __init__(self, data_length=100, n_class=3):
        self.data_length = data_length
        self.noise = 0.1

        # Generate data for each class
        radius = range(1, n_class+1)
        theta = np.linspace(0, 2 * np.pi, data_length)

        self.data = []
        self.targets = []

        for i, r in enumerate(radius):
            X_class = np.vstack((r * np.cos(theta) + np.random.normal(0, self.noise, self.data_length),
                                 r * np.sin(theta) + np.random.normal(0, self.noise, self.data_length))).T
            y_class = np.ones(data_length) * i
            self.data.append(X_class)
            self.targets.append(y_class)

        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)
    

class Data():
    def __init__(self, dataset_sizes=[1000, 1000, 1000], n_classes=[3, 3, 3]) -> None:
        self.t_train_data = SimpleDataset(data_length=dataset_sizes[0], n_class=n_classes[0])
        self.s_train_data = SimpleDataset(data_length=dataset_sizes[1], n_class=n_classes[1])
        self.test_data = SimpleDataset(data_length=dataset_sizes[2], n_class=n_classes[2])
