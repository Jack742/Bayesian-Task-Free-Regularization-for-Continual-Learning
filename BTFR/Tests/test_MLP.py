import pytest
import torch

from ..Models.MLP import MCDropout_MLP

def test_MCDropout_MLP_output_shape():
    inp = 10
    out = 2
    x = torch.linspace(0,inp, dtype=torch.float)
    model = MCDropout_MLP(inp,out,(inp, inp//2))
    assert(len(model(x))==out)
