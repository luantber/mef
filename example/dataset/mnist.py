from torchvision.datasets import MNIST
from torchvision import transforms

def Mnist():
    return MNIST(".",True,download=True, transform=transforms.ToTensor())