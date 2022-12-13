from typing import Iterable, Optional, Sequence, List, Union

import torch
import numpy as np
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Optimizer, SGD

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
    EWCPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.logging import InteractiveLogger
from avalanche.benchmarks import CLExperience, CLStream
from .utils import enable_dropout
from .Plugins import TFMASPlugin
from .optimizers_lib import bgd_optimizer

ExpSequence = Iterable[CLExperience]

class BGDCL(SupervisedTemplate):
    """Al Jundi Task Free Continual Learning.

    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        model: Module,
        criterion,
        plugins: Optional[List[SupervisedPlugin]]=None,
        task_type = 'classification',
        num_test_repeats = 5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,        
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,

        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        
        assert(train_mb_size==1 and eval_mb_size==1)
        
        optimizer = bgd_optimizer.BGD(model.named_parameters(),std_init=5e-2, mc_iters=num_test_repeats,mean_eta=1)
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

        self.task_type = task_type
        self.num_test_repeats = num_test_repeats
        self.lower_threshold = 0.7 #Below this, we don't update importances


    
    def classification_mean_std(self):
        output_list = []
        for _ in range(self.num_test_repeats):     
            output_list.append(softmax(self.forward()))
        assert(len(output_list[0][0])==10)
        z = torch.cat(output_list,-1).view(-1, len(output_list[0][0])).split(self.num_test_repeats)
        preds = []
        
        for each in z:
            preds.append(each.mean(0))
        preds = torch.stack(preds)        
        return preds

    # def regression_mean_std(self):
    # """MAY BE BROKEN, CHECK THIS!!!"""
    #     output_list = []
    #     for _ in range(self.num_test_repeats):
    #         output_list.append(self.forward())
    #     p = torch.cat(output_list, 0)
    #     return (p.mean(0), p.std(0))



    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            
            for _ in range(self.num_test_repeats):
                self.optimizer.randomize_weights()
                
                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                
                self._after_forward(**kwargs)

                # Loss & Backward
                self.loss = self.criterion()

                self.optimizer.zero_grad()
                self._before_backward(**kwargs)
                self.backward()
                self._after_backward(**kwargs)
                self.optimizer.aggregate_grads(1)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)
            self._after_training_iteration(**kwargs)         
                    

    def _train_exp(
            self, experience: CLExperience, eval_streams=None, **kwargs
        ):
            """Training loop over a single Experience object.

            :param experience: CL experience information.
            :param eval_streams: list of streams for evaluation.
                If None: use the training experience for evaluation.
                Use [] if you do not want to evaluate during training.
            :param kwargs: custom arguments.
            """
            if eval_streams is None:
                eval_streams = [experience]
            for i, exp in enumerate(eval_streams):
                if not isinstance(exp, Iterable):
                    eval_streams[i] = [exp]
            for _ in range(self.train_epochs):
                self._before_training_epoch(**kwargs)

                if self._stop_training:  # Early stopping
                    self._stop_training = False
                    break

                self.training_epoch(**kwargs)
                self._after_training_epoch(**kwargs)

    def train(
            self,
            experiences: Union[CLExperience, ExpSequence],
            eval_streams: Optional[Sequence[Union[CLExperience,
                                                ExpSequence]]] = None,
            **kwargs,
        ):
            """Training loop.

            If experiences is a single element trains on it.
            If it is a sequence, trains the model on each experience in order.
            This is different from joint training on the entire stream.
            It returns a dictionary with last recorded value for each metric.

            :param experiences: single Experience or sequence.
            :param eval_streams: sequence of streams for evaluation.
                If None: use training experiences for evaluation.
                Use [] if you do not want to evaluate during training.
            """
            self.is_training = True
            self._stop_training = False
                    
            

            self.model.train()
            self.model.to(self.device)

            # Normalize training and eval data.
            if not isinstance(experiences, Iterable):
                experiences = [experiences]
            if eval_streams is None:
                eval_streams = [experiences]
            self._eval_streams = eval_streams

            self._before_training(**kwargs)

            for self.experience in experiences:
                self._before_training_exp(**kwargs)
                self._train_exp(self.experience, eval_streams, **kwargs)
                self._after_training_exp(**kwargs)
            self._after_training(**kwargs)


    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)          

            if self.task_type=='classification':
                preds = self.classification_mean_std()

            self.mb_output = preds       
   

            
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    @torch.no_grad()
    def eval(
        self,
        exp_list: Union[CLExperience, CLStream],
        **kwargs,
    ):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()
        enable_dropout(self.model)

        if not isinstance(exp_list, Iterable):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            self._before_eval_exp(**kwargs)
            self._eval_exp(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)

        # restore previous shared state.
        self._load_train_state(prev_train_state)