import torch
from torchvision import transforms
import torchvision
import numpy as np
from typing import Optional, Any
from . import DataInterface
from .datasets import _get_mnist


class PMNIST(DataInterface):
    def __init__(self,\
                n_tasks: int,
                train_transform: Optional[Any] = None,
                test_transform: Optional[Any] = None
                )->None:
        super(DataInterface, self).__init__(n_tasks,train_transform,test_transform)
        self.mnist_train, self.mnist_test = _get_mnist()
        self.train_data_list, self.test_data_list = [],[]


    def get_permute(self)->torch.Tensor:
        """
        Permute pixels of images
        """
        random_permute = np.random.RandomState()

        for _ in range(self.n_tasks):
            idx_permute = torch.from_numpy(\
                random_permute.permutation(784).type(torch.int64))
            
            permutation = PermutePixels(idx_permute)
            
class PermutePixels(object):
    def __init__(self,permute_idx):
        self.permute = permute_idx
        self.to_tensor= transforms.ToTensor()
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            try:
                img = self.to_tensor(img)
            except Exception as e:
                raise e
        img = img.view(-1)[self.permutation].view(*img.shape)

        return img


