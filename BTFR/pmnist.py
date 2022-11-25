import torch
import torch.nn as nn
from torch.optim import SGD
import pickle
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EWCPlugin
#from avalanche.training import MAS#Naive, EWC
from Training.Plugins import MASPlugin, TFEWCPlugin, TFMASPlugin
from Training import BayesianCL, EWCBayesianCL, MASBayesianCL
from Training.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
num_model_reruns =10

torch.manual_seed(123456)

model = SimpleMLP(num_classes=10)
# CL Benchmark Creation
perm_mnist = PermutedMNIST(n_experiences=5)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

ewc = EWCPlugin(ewc_lambda=100, mode='separate', keep_importance_data=False)
tfewc = TFEWCPlugin(ewc_lambda=100, mode='online', decay_factor=1.0, keep_importance_data=False)
mas = MASPlugin()
tfmas = TFMASPlugin()

strategies = {
        "ewc_base":BayesianCL(
     model, optimizer, criterion, plugins=[ewc],num_test_repeats=num_model_reruns,train_mb_size=32, train_epochs=1,
     eval_mb_size=32, device=device),
    "ewc_tf":BayesianCL(
     model, optimizer, criterion, plugins=[tfewc],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
     eval_mb_size=1, device=device),
            "btfr_ewc":EWCBayesianCL(
     model, optimizer, criterion, train_mb_size=1,num_test_repeats=num_model_reruns, train_epochs=1,
     eval_mb_size=1, device=device),
    "mas_base": BayesianCL(model, optimizer, criterion, plugins=[mas],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    eval_mb_size=1, device=device),
                "mas_tf": BayesianCL(model, optimizer, criterion, plugins=[tfmas],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    eval_mb_size=1, device=device),
                "btfr_mas":MASBayesianCL(model, optimizer, criterion, num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    eval_mb_size=1, device=device)     
     }


for cls in strategies.keys():
    print(f"#################\nSTARTING {cls}\n#################")
    cl_strategy = strategies[cls]
    
    #reinitialise parameters
    model.apply(init_params)
    results = []

    for train_exp in train_stream:
        cl_strategy.train(train_exp)
        results.append(cl_strategy.eval(test_stream))

    with open(f'Results/pmnist/{cls}', 'wb') as f:
        pickle.dump(results, f)

"""
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
"""