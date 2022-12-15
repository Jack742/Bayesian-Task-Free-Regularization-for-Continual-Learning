import torch
import torch.nn as nn
import numpy as np
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST, SplitCIFAR10
from avalanche.training.plugins import EWCPlugin

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()