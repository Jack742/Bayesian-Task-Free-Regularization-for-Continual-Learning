import torch
import torch.nn as nn
import numpy as np
from avalanche.benchmarks.classic import PermutedMNIST, SplitMNIST, SplitCIFAR10
from avalanche.training.plugins import EWCPlugin
from Models import *
from Training.Plugins import MASPlugin, TFEWCPlugin, TFMASPlugin, WithLabels_MASPlugin, WithLabels_EWCPlugin
from Training import BayesianCL, EWCBayesianCL, MASBayesianCL, TFCLBayesianCL, BGDCL
from Training.Plugins import MASPlugin, TFEWCPlugin, TFMASPlugin, WithLabels_MASPlugin, WithLabels_EWCPlugin

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def init_params(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias'):
        m.bias.data[m.bias.data != 0.0] = 0.0        

def prep_pmnist(n_experiences):
    model = SimpleMLP(n_classes=10)
    perm_mnist = PermutedMNIST(n_experiences=n_experiences)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream
    return (model, train_stream, test_stream)

def prep_splitmnist(n_experiences):
    model = SimpleMLP(n_classes=10)
    perm_mnist = PermutedMNIST(n_experiences=n_experiences)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream
    return (model, train_stream, test_stream)

def prep_splitcifar10(n_experiences):
    model = SimpleCNN(n_classes=10)
    perm_mnist = PermutedMNIST(n_experiences=n_experiences)
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream
    return (model, train_stream, test_stream)

def get_ewcbase(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None, **kwargs):
    if not plugins:
        plugins = EWCPlugin(ewc_lambda=40, mode='separate', keep_importance_data=False)
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,
        eval_mb_size=32, evaluator=eval_plugin, **kwargs)

def get_ewconline(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None, **kwargs):
    if not plugins:
        plugins = EWCPlugin(ewc_lambda=40, mode='online', decay_factor=1.0, keep_importance_data=False)
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,
        eval_mb_size=32, evaluator=eval_plugin, **kwargs)

def get_tfewc(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None, **kwargs):
    if not plugins:
        plugins = TFEWCPlugin(ewc_lambda=100, mode='separate', keep_importance_data=False)
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=32, evaluator=eval_plugin, **kwargs)

def get_tfewconline(model, optimizer, criterion, n_model_reruns,device,eval_plugin, plugins=None, **kwargs):
    if not plugins:
        plugins = TFEWCPlugin(ewc_lambda=100, mode='online', decay_factor=1.0, keep_importance_data=False)
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=32, evaluator=eval_plugin, **kwargs)

def get_masbase(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None, **kwargs):
    if not plugins:
        plugins = MASPlugin()
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=1, evaluator=eval_plugin, **kwargs),

def get_tfmas(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None, **kwargs):
    if not plugins:
        plugins = TFMASPlugin()
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=1, evaluator=eval_plugin, **kwargs),

def get_tfcl(model, optimizer, criterion, n_model_reruns,eval_plugin, **kwargs):
    return TFCLBayesianCL(model, optimizer, criterion, n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=1, evaluator=eval_plugin, **kwargs),

def get_btfr_ewc_labelled(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None,**kwargs):
    if not plugins:
        plugins = WithLabels_EWCPlugin(ewc_lambda=40, mode='online',decay_factor=1.0, keep_importance_data=False)
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=1, evaluator=eval_plugin, **kwargs)

def get_btfr_ewc(model, optimizer, criterion, n_model_reruns,eval_plugin, **kwargs):
    return EWCBayesianCL(model, optimizer, criterion, train_mb_size=1,n_model_reruns=n_model_reruns, train_epochs=1,
        eval_mb_size=1, evaluator=eval_plugin),

def get_btfr_mas_labelled(model, optimizer, criterion, n_model_reruns,eval_plugin, plugins=None,**kwargs):
    if not plugins:
        plugins = WithLabels_MASPlugin()
    return BayesianCL(model, optimizer, criterion, plugins=[plugins],n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=1, evaluator=eval_plugin, **kwargs)

def get_btfr_mas(model, optimizer, criterion, n_model_reruns,device,eval_plugin, **kwargs):
    return MASBayesianCL(model, optimizer, criterion, n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,\
        eval_mb_size=1, device=device, evaluator=eval_plugin, **kwargs),

def get_bgd(model,criterion, n_model_reruns,eval_plugin, **kwargs):
    return BGDCL(model, criterion, n_model_reruns=n_model_reruns,train_mb_size=1, train_epochs=1,
     eval_mb_size=1, evaluator=eval_plugin, **kwargs),

def get_sibase(model, optimizer, criterion, plugins, n_model_reruns,device,eval_plugin, **kwargs):
    raise NotImplementedError

def get_tfsi(model, optimizer, criterion, plugins, n_model_reruns,device,eval_plugin, **kwargs):
    raise NotImplementedError

def get_btfr_si_labelled(model, optimizer, criterion, plugins, n_model_reruns,device,eval_plugin, **kwargs):
    pass

def get_btfr_si(model, optimizer, criterion, plugins, n_model_reruns,device,eval_plugin, **kwargs):
    pass