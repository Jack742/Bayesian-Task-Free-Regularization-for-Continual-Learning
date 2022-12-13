import torch
import torch.nn as nn
from torch.optim import SGD
import pickle
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.training.plugins import EWCPlugin, EvaluationPlugin
from avalanche.logging import CSVLogger, InteractiveLogger
from avalanche.evaluation.metrics import (accuracy_metrics,loss_metrics,class_accuracy_metrics,\
    forgetting_metrics,forward_transfer_metrics, bwt_metrics, amca_metrics)
from avalanche.training import LwF, Naive#, EWC

from Training.Plugins import MASPlugin, TFEWCPlugin, TFMASPlugin, WithLabels_MASPlugin, WithLabels_EWCPlugin
from Training import BayesianCL, EWCBayesianCL, MASBayesianCL
from Training.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
num_model_reruns =10
n_experiences = 10

strategy_names = ['ewc_tf']#['ewc_base','labelled_btfr_ewc','ewc_tf', 'btfr_ewc', 'mas_base','mas_tf', 'btfr_mas','naive']

num_strategies = len(strategy_names)
eval_plugins = []
for _, strat_name in zip(range(num_strategies), strategy_names):
    inter_logger = InteractiveLogger()
    csv_logger = CSVLogger(f'Results/splitmnist/csvlogger/{strat_name}_{n_experiences}_exp')
    eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=False,
                epoch=True,
                epoch_running=True,
                experience=True,
                stream=True,
            ),
            # loss_metrics(
            #     minibatch=False,
            #     epoch=True,
            #     epoch_running=True,
            #     experience=True,
            #     stream=True,
            # ),
            # class_accuracy_metrics(
            #     epoch=True, stream=True, classes=list(range(10))
            # ),
            #amca_metrics(),
            #forgetting_metrics(experience=True, stream=True),
            #bwt_metrics(experience=True, stream=True),
            #forward_transfer_metrics(experience=True, stream=True),
            loggers=[inter_logger, csv_logger],
            collect_all=True,
    )
    eval_plugins.append(eval_plugin)

torch.manual_seed(123456)

model = SimpleMLP(num_classes=10)
# CL Benchmark Creation
perm_mnist = SplitMNIST(n_experiences=5)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

ewc = EWCPlugin(ewc_lambda=40, mode='separate', keep_importance_data=False)
tfewc = TFEWCPlugin(ewc_lambda=100, mode='online', decay_factor=1.0, keep_importance_data=False)
mas = MASPlugin()
wlbtfrmas = WithLabels_EWCPlugin(ewc_lambda=40, mode='online',decay_factor=1.0, keep_importance_data=False)
tfmas = TFMASPlugin()

strategies = { 
    #                 "ewc_base":BayesianCL(model, optimizer, criterion, plugins=[ewc],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    #  eval_mb_size=32, device=device, evaluator=eval_plugins[0], eval_every=1),
    #             "labelled_btfr_ewc": BayesianCL(model, optimizer, criterion, beta=1.0,plugins=[wlbtfrmas],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    # eval_mb_size=1, device=device, evaluator=eval_plugins[1], eval_every=1),
    #     "ewc_base":BayesianCL(
    #  model, optimizer, criterion, plugins=[ewc],num_test_repeats=num_model_reruns,train_mb_size=32, train_epochs=1,
    #  eval_mb_size=32, device=device, evaluator=eval_plugins[0], eval_every=1),
        "ewc_tf":BayesianCL(
     model, optimizer, criterion, plugins=[tfewc],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
     eval_mb_size=1, device=device, evaluator=eval_plugins[0], eval_every=1),
    #     "btfr_ewc":EWCBayesianCL(
    #  model, optimizer, criterion, train_mb_size=1,num_test_repeats=num_model_reruns, train_epochs=1,
    #  eval_mb_size=1, device=device, evaluator=eval_plugins[3], eval_every=1),
    #     "mas_base": BayesianCL(model, optimizer, criterion, plugins=[mas],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    # eval_mb_size=32, device=device, evaluator=eval_plugins[4], eval_every=1),
    #     "mas_tf": BayesianCL(model, optimizer, criterion, plugins=[tfmas],num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    # eval_mb_size=1, device=device, evaluator=eval_plugins[5], eval_every=1),
    #     "btfr_mas":MASBayesianCL(model, optimizer, criterion, num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,
    # eval_mb_size=1, device=device, evaluator=eval_plugins[6], eval_every=1),
    #     'naive': Naive(model, optimizer, criterion, train_mb_size=32, train_epochs=1,
    # eval_mb_size=32, device=device,evaluator=eval_plugins[7], eval_every=1)
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
    
    metrics = cl_strategy.evaluator.get_all_metrics()
    print(f'All Metrics: {metrics}\n')

    with open(f'Results/pmnist/{cls}', 'wb') as f:
        pickle.dump(results, f)


"""
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
"""