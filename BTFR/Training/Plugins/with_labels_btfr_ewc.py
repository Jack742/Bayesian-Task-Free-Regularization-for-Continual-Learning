from collections import defaultdict
from typing import Dict, Tuple
import warnings
import itertools
from copy import deepcopy
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins import EWCPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict


class WithLabels_EWCPlugin(EWCPlugin):
    """
    Elastic Weight Consolidation (EWC) plugin.
    EWC computes importance of each weight at the end of training on current
    experience. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on previous experiences in proportion to their importance
    on that experience. Importances are computed with an additional pass on the
    training set. This plugin does not use task identities.
    """

    def __init__(
        self,
        ewc_lambda,
        mode="online",
        decay_factor=None,
        keep_importance_data=False,
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param decay_factor: used only if mode is `online`.
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """
        super().__init__(ewc_lambda, mode,decay_factor,keep_importance_data)
        #NEW ADDITION
        self.NO_UPDATE = True

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
            strategy.certainty,
            strategy.lower_threshold,
            strategy.beta
        )           

        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    # def after_training_iteration(self, strategy, **kwargs):
    #     """
    #     Compute importances of parameters after each experience.
    #     """
        
    #     #NEW ADDITION
    #     exp_counter = strategy.clock.train_exp_iterations
    #     #NEW ADDITION
    #     importances = self.compute_importances(
    #         strategy.model,
    #         strategy._criterion,
    #         strategy.optimizer,
    #         (strategy.mb_x, strategy.mb_y,strategy.mb_task_id),
    #         strategy.device,
    #         strategy.train_mb_size,
    #         strategy.certainty,
    #         strategy.lower_threshold,
    #         strategy.beta
    #     )
    #     self.update_importances(importances, exp_counter)
    #     self.saved_params[exp_counter] = copy_params_dict(strategy.model)
    #     # clear previous parameter values
    #     if exp_counter > 0 and (not self.keep_importance_data):
    #         del self.saved_params[exp_counter - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size, certainty, threshold,beta
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        #NEW ADDITION
        dataloader = DataLoader(dataset, batch_size=batch_size)
        if certainty < threshold:
            self.NO_UPDATE = True
            return importances
        else:
            self.NO_UPDATE = False
            for i,batch in enumerate(dataloader):        
                # get only input, target and task_id from the batch
                x, y, task_labels = batch[0], batch[1], batch[-1]
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                out = avalanche_forward(model, x, task_labels)
                loss = criterion(out, y)
                loss.backward()

                for (k1, p), (k2, imp) in zip(
                    model.named_parameters(), importances
                ):
                    assert k1 == k2
                    if p.grad is not None:
                        #NEW ADDITION
                        imp += (p.grad.data.clone().pow(2)) * ((certainty**2) * beta)
        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))
            
        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """
      
        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            if not hasattr(self, 'num_updates'):
                self.num_updates = 0

            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1], importances, fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    self.importances[t].append((k2, curr_imp))
                    continue
                assert( k1 == k2, f"Error in importance computation. {k1}||{k2}")
                #NEW ADDITION                
                self.importances[t].append(
                    (k1, (self.decay_factor * (old_imp*(self.num_updates) + curr_imp)/(self.num_updates+1)))
                )
            self.num_updates+=1
            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


ParamDict = Dict[str, Tensor]
EwcDataType = Tuple[ParamDict, ParamDict]