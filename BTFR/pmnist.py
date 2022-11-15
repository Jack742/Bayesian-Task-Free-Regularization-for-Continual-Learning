import torch
import torch.nn as nn
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EWCPlugin
#from avalanche.training import Naive, EWC
from Training import BayesianCL

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

model = SimpleMLP(num_classes=10)

# CL Benchmark Creation
perm_mnist = PermutedMNIST(n_experiences=5)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Continual learning strategy
ewc = EWCPlugin(ewc_lambda=100, mode='separate', decay_factor=None, keep_importance_data=False)
cl_strategy = BayesianCL(
    model, optimizer, criterion, plugins=[ewc],train_mb_size=32, train_epochs=4,
    eval_mb_size=32, device=device)

# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))



"""
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
"""