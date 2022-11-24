import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .Plugins import PluginInterface

class BaseCLInterface:
    """
    Interface for continual learning classes. 
        | Guarantees presence of specific attributes/methods
    """
    def __init__(self,
                model: nn.Module,
                plugin: Optional[PluginInterface] = None,
                device=torch.device("cuda:0" if torch.cuda.is_available()
                     else "cpu"))->None:
        
        self.model = model

        self.plugin = plugin

        self.device = device

    def train(self, **kwargs):
        pass

    def eval(self, **kwargs):
        pass

    def trigger_plugin(self, method, **kwargs):
        if hasattr(self.plugin,method):
            getattr(self.plugin, method)(self, **kwargs)
