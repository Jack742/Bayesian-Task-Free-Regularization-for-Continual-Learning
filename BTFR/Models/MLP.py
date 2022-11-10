import torch
import torch.nn as nn

class MCDropout_MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features:tuple) -> None:
        super(MCDropout_MLP, self).__init__()
        self.l1 = nn.Linear(in_features, hidden_features[0])
        self.l2 = nn.Linear(hidden_features[0], hidden_features[1])        
        self.l3 = nn.Linear(hidden_features[1], out_features)
        self.ReLu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.ReLu(self.drop(self.l1(x)))
        x = self.ReLu(self.drop(self.l2(x)))
        x = self.l3(x)
        return x

