import torch
import torch.nn as nn
import numpy as np

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def init_params(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias'):
        m.bias.data[m.bias.data != 0.0] = 0.0