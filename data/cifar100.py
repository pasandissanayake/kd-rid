import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR100

class Data():
    def __init__(self, path) -> None:
        # Define transformations to preprocess the data
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        # transform = transforms.Compose(
        #             [transforms.ToTensor(),
        #             transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
        # )
        # self.train_data = CIFAR100(root=path, train=True, download=True, transform=transform)
        # self.test_data = CIFAR100(root=path, train=False, download=True, transform=transform)

        self.train_data = CIFAR100(root=path, train=True, download=True, transform=transform_train)
        self.test_data = CIFAR100(root=path, train=False, download=True, transform=transform_test)

