import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_classification

class BlobData(Dataset):
    def __init__(self, data_length=100):
        self.data_length = data_length
        self.noise = 0.1

        self.data, self.targets = make_classification(n_samples=data_length, n_features=2, n_classes=3, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=1.0, random_state=42)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)
    

class Data():
    def __init__(self, dataset_sizes=[1000, 1000, 1000]) -> None:
        self.t_train_data = BlobData(data_length=dataset_sizes[0])
        self.s_train_data = BlobData(data_length=dataset_sizes[1])
        self.test_data = BlobData(data_length=dataset_sizes[2])
