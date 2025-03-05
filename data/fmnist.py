import torch
from torchvision import datasets, transforms


class Data():
    def __init__(self, path) -> None:
        # Define transformations to apply to the data
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
        ])
        self.train_data = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        self.test_data = datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
