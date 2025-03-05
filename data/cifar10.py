import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class Data():
    def __init__(self, path) -> None:
        # Define transformations to preprocess the data
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the pixel values to the range [-1, 1]
        ])
        self.train_data = CIFAR10(root=path, train=True, transform=transform, download=True)        
        self.test_data = CIFAR10(root=path, train=False, transform=transform, download=True)


