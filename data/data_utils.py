import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class SubsetDataset(Dataset):
    def __init__(self, dataset, samples_per_class):
        """
        Args:
            dataset (torch.utils.data.Dataset): Existing dataset.
            samples_per_class (int): Number of samples to select from each class.
        """
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.indices = self._select_indices()

    def _select_indices(self):
        indices = []
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        for label, indices_list in class_indices.items():
            selected_indices = np.random.choice(indices_list, size=min(self.samples_per_class, len(indices_list)), replace=False)
            indices.extend(selected_indices)
        
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_idx = self.indices[index]
        return self.dataset[dataset_idx]
    

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image    

def plot_images(dataset, n_images, figsize=(8,8), normalize = True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize = figsize)

    images, labels = zip(*[(image, label) for image, label in 
                        [dataset[i] for i in range(n_images)]])

    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = labels[i]
        ax.set_title(label)
        ax.axis('off')