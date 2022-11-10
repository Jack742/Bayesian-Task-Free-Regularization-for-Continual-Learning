import pytest
import torch

from Models.MLP import MCDropout_MLP
from utils import enable_dropout


def test_enable_dropout():
    inp = 10
    out = 2
    x = torch.arange(0,inp, dtype=torch.float)
    model = MCDropout_MLP(inp,out,(inp, inp//2))
    #Make sure model is deterministic without MC.Dropout
    model.eval() 
    with torch.no_grad():       
        y = model(x)
        z = model(x)
        print(y,z)
        assert(torch.any(torch.eq(y,z)))
    #Make sure it is no longer deterministic         
    enable_dropout(model)
    with torch.no_grad():       
        y = model(x)
        z = model(x)
        print(y,z)
        assert(not torch.any(torch.eq(y,z)))