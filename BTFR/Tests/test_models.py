import pytest
import torch

from BTFR import Models

def test_MCDropout_MLP_output_shape():
    inp = 10
    out = 2
    x = torch.arange(0,inp, dtype=torch.float)
    model = Models.MCDropout_MLP(inp,out,(inp, inp//2))
    assert(len(model(x))==out)
