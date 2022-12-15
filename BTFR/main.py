import json
import torch
import torch.nn as nn
from torch.optim import SGD
import pickle
from Models import *
from avalanche.training.plugins import EWCPlugin, EvaluationPlugin
from avalanche.logging import CSVLogger, InteractiveLogger
from avalanche.evaluation.metrics import (accuracy_metrics,loss_metrics,class_accuracy_metrics,\
    forgetting_metrics,forward_transfer_metrics, bwt_metrics, amca_metrics)
from avalanche.training import Naive#, EWC
from Training.Plugins import MASPlugin, TFEWCPlugin, TFMASPlugin, WithLabels_MASPlugin, WithLabels_EWCPlugin
from Training import BayesianCL, EWCBayesianCL, MASBayesianCL
import utils

torch.manual_seed(123456)
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

with open('Configs\conf.json','rb') as f:
    config = json.load(f)

n_model_reruns = config['n_model_reruns']
n_experiences = config['n_experiences']
strategy_names = config['strategy_list']
num_strategies = len(strategy_names)

(model,train_stream,test_stream) = getattr(utils, f"prep_{config['experiment']}")(n_experiences=n_experiences)
# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

strategies={}
for strat_name in strategy_names:
    inter_logger = InteractiveLogger()
    csv_logger = CSVLogger(f'Results/pmnist/csvlogger/{strat_name}_{n_experiences}_exp')
    eval_plugin = EvaluationPlugin(
            accuracy_metrics(
                minibatch=False,
                epoch=True,
                epoch_running=True,
                experience=True,
                stream=True,
            ),
            loggers=[inter_logger, csv_logger],
            collect_all=True,
    )
    strategies[strat_name] = getattr(utils, f"get_{strat_name}")(model=model,\
        optimizer=optimizer,criterion=criterion,eval_plugin=eval_plugin, device=device, n_model_reruns=n_model_reruns)

for cls in strategies.keys():
    print(f"#################\nSTARTING {cls}\n#################")
    cl_strategy = strategies[cls]
    
    #reinitialise parameters
    model.apply(utils.init_params)
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