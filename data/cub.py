import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
class Data():
    def __init__(self, path) -> None:
        pretrained_size = 224
        pretrained_means = [0.485, 0.456, 0.406]
        pretrained_stds= [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
                                transforms.Resize(pretrained_size),
                                transforms.RandomRotation(5),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomCrop(pretrained_size, padding = 10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = pretrained_means, 
                                                        std = pretrained_stds)
                            ])

        test_transforms = transforms.Compose([
                                transforms.Resize(pretrained_size),
                                transforms.CenterCrop(pretrained_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = pretrained_means, 
                                                        std = pretrained_stds)
                            ])

        self.train_data = Cub2011(path, train=True, transform=train_transforms, download=True)
        self.test_data = Cub2011(path, train=False, transform=test_transforms, download=True)

    
    def normalize_image(self, image):
        image_min = image.min()
        image_max = image.max()
        image.clamp_(min = image_min, max = image_max)
        image.add_(-image_min).div_(image_max - image_min + 1e-5)
        return image    

    def plot_images(self, dataset, n_images, figsize=(8,8), normalize = True):

        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))
        fig = plt.figure(figsize = figsize)

        images, labels = zip(*[(image, label) for image, label in 
                            [dataset[i] for i in range(n_images)]])

        for i in range(rows*cols):
            ax = fig.add_subplot(rows, cols, i+1)
            image = images[i]

            if normalize:
                image = self.normalize_image(image)

            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            label = labels[i]
            ax.set_title(label)
            ax.axis('off')
