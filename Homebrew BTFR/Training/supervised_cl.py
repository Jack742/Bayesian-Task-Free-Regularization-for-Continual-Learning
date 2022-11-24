import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

import BaseCLInterface
from .Plugins import PluginInterface
from .Data import DataInterface

class SupervisedCLInterface(BaseCLInterface):
    """
    Interface for supervised continual learning classes. 
        | Guarantees presence of specific attributes/methods
        | Train Loop
        | Eval Loop
    """
    def __init__(self,
            model: nn.Module,
            data: Optional[DataInterface] = None,
            plugin: Optional[PluginInterface] = None,
            device=torch.device("cuda:0" if torch.cuda.is_available()
                    else "cpu"))->None:
                    
        super(BaseCLInterface, self).__init__(model,plugin,device)

    def train(self, **kwargs)->list:
        """
        Trains model on data
        """
        self.model.train
        self.model.to(self.device)

        self.trigger_plugin('before_training',**kwargs)

        for 

