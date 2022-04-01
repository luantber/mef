from torchvision.datasets import MNIST
from torchvision import transforms

import os
dir_path = os.path.dirname(os.path.realpath(__file__))


def Mnist():
    return MNIST(
        dir_path, True, download=True, transform=transforms.ToTensor()
    )

