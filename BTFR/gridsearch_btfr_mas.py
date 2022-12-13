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
from avalanche.training import Naive#, EWC
from Training.Plugins import MASPlugin, TFEWCPlugin, TFMASPlugin, TEMPBTFRMASPlugin
from Training import BayesianCL, EWCBayesianCL, MASBayesianCL
from Training.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
num_model_reruns =10
n_experiences = 10
torch.manual_seed(123456)

# CL Benchmark Creation
perm_mnist = PermutedMNIST(n_experiences=n_experiences)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream


ks=[3,2,1]
lambdas = [1.0]
alphas = [0.5]
betas = [1.06]
for k in ks:
    for lamb in lambdas:
        for alpha in alphas:
            for beta in betas:
                #Evaluation
                inter_logger = InteractiveLogger()
                csv_logger = CSVLogger(f'Results/pmnist/csvlogger/btfr_mas_grid/\
                    {n_experiences}_exp_k_{k}_lambda_{lamb}_alpha_{alpha}_beta_{beta}/')
                eval_plugin = EvaluationPlugin(
                        accuracy_metrics(
                            minibatch=False,
                            epoch=True,
                            epoch_running=True,
                            experience=True,
                            stream=True,            ),
                        
                        loggers=[inter_logger, csv_logger],
                        collect_all=True,
                )

                model = SimpleMLP(num_classes=10)
                model.apply(init_params)

                # Prepare for training & testing
                optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
                criterion = nn.CrossEntropyLoss()

                cl_strategy = MASBayesianCL(model, optimizer, criterion, k=k,alpha=alpha,lambda_=lamb, beta=beta,\
                    num_test_repeats=num_model_reruns,train_mb_size=1, train_epochs=1,eval_mb_size=1, \
                        device=device, evaluator=eval_plugin, eval_every=1)
                
                results = []
                for train_exp in train_stream:
                    print(f'||K: {k} || Alpha: {alpha} || Lambda: {lamb}||')
                    cl_strategy.train(train_exp)
                    results.append(cl_strategy.eval(test_stream))
                
                metrics = cl_strategy.evaluator.get_all_metrics()
                print(f'All Metrics: {metrics}\n')