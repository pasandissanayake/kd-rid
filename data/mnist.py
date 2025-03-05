import torchvision

class Data():
    def __init__(self, path) -> None:
        self.train_data = torchvision.datasets.MNIST(path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        
        self.test_data = torchvision.datasets.MNIST(path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ]))