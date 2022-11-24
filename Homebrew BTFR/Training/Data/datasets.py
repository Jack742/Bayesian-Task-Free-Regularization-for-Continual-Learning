from torchvision.datasets import MNIST
"""
Getter functions to obtain datasets
"""

def _get_mnist()->tuple:
    mnist_trainset = MNIST(root='./saved_datasets',train=True, \
        download=True, transform=None)
    mnist_testset = MNIST(root='./saved_datasets',train=False, \
        download=True, transform=None)
    
    (mnist_trainset, mnist_testset)