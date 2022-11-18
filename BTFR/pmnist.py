import torch
import torch.nn as nn
from torch.optim import SGD
import pickle
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EWCPlugin, MASPlugin
#from avalanche.training import Naive, EWC
from Training import BayesianCL, EWCBayesianCL

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
ewc = EWCPlugin(ewc_lambda=100, mode='online', decay_factor=1.0, keep_importance_data=False)
cl_strategy = BayesianCL(
    model, optimizer, criterion, plugins=[ewc],train_mb_size=1, train_epochs=1,
    eval_mb_size=1, device=device)

# cl_strategy = EWCBayesianCL(
#     model, optimizer, criterion, train_mb_size=1, train_epochs=1,
#     eval_mb_size=1, device=device)
#train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))

with open('Results/pmnist/ewc_online_no_labels', 'wb') as f:
    pickle.dump(results, f)

"""
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
"""