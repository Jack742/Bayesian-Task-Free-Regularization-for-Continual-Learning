from typing import Iterable, Optional, Sequence, List, Union
import numpy as np
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Optimizer, SGD
from copy import deepcopy
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
ExpSequence = Iterable[CLExperience]

class TFCLBayesianCL(SupervisedTemplate):
    """Elastic Weight Consolidation (EWC) strategy.

    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        plugins: List[SupervisedPlugin] = None,
        task_type = 'classification',
        n_model_reruns = 5,
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
        self.num_test_repeats = n_model_reruns
        self.last_sample = None
        self.beta =1.0
        self.lower_threshold = 0.7
        
        
        self.recent_buffer = []
        self.hard_buffer = []

        self.recent_buffer_size = 30
        self.hard_buffer_size = 30
        self.use_hard_buffer = True
        self.gradient_steps = 5
    
        self.losses = []
        self.last_loss_window_mean = 0.0
        self.last_loss_window_var = 0.0
        self.new_peak_detected = False

        self.loss_window = []
        self.loss_window_means =[]
        self.loss_window_vars =[]
        self.loss_window_length=int(5),
        self.loss_window_mean_threshold=0.2,
        self.loss_window_variance_threshold=0.1, 

        self.update_tags = []
        self.MAS_Weight = 0.5
        self.star_vars = []
        self.omegas = []
 

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
            device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

            for self.mbatch in self.dataloader:
                if self._stop_training:
                    break
                self._unpack_minibatch()
                self._before_training_iteration(**kwargs)

                self.recent_buffer.append({'x':self.mb_x, 'y': self.mb_y, 'id':self.mb_task_id})
                if len(self.recent_buffer) > self.recent_buffer_size:
                    del(self.recent_buffer[0])
                
                if len(self.recent_buffer) == self.recent_buffer_size:
                    x = [_['x'] for _ in self.recent_buffer]
                    y = [_['y'] for _ in self.recent_buffer]
                    _id = [_['id'] for _ in self.recent_buffer]

                    if self.use_hard_buffer and len(self.hard_buffer) != 0:
                        xh = [_['x'] for _ in self.hard_buffer]
                        yh = [_['y'] for _ in self.hard_buffer]
                        
                    for gs in range(self.gradient_steps):
                        self.optimizer.zero_grad()
                        self.loss = 0
                        _x = torch.cat([_.detach().cpu() for _ in x]).reshape(-1,28,28).to(device)#.type(torch.float32)#torch.from_numpy(np.asarray([_.detach().cpu() for _ in x], dtype=np.float64)).to(device)#.reshape(-1,28,28)
                        _y = torch.zeros(len(y),10).scatter_(1,torch.cat([_.detach().cpu() for _ in y]).reshape(-1,1).type(torch.LongTensor),1.).type(torch.FloatTensor).to(device)#.type(torch.LongTensor),1.).type(torch.FloatTensor)
                        self.mbatch = [_x, _y, self.mb_task_id]

                        # Forward
                        self._before_forward(**kwargs)
                        preds = self.classification_mean_std()
                        self.mb_output = preds
                        self.recent_preds = preds
                        self.certainty = torch.max(preds).item()
                        self._after_forward(**kwargs)
                        recent_loss = self.criterion() 
                        self.loss += torch.sum(recent_loss)


                        if self.use_hard_buffer and len(self.hard_buffer) != 0:
                            assert len(xh) <= self.hard_buffer_size
                            _xh = torch.from_numpy(np.asarray(xh).reshape(-1,28,28)).type(torch.float32)#.type(torch.float32)#torch.from_numpy(np.asarray([_.detach().cpu() for _ in x], dtype=np.float64)).to(device)#.reshape(-1,28,28)
                            _yh = torch.zeros(len(yh),10).scatter_(1,torch.from_numpy(np.asarray(yh).reshape(-1,1)).type(torch.LongTensor),1.).type(torch.FloatTensor)
                            self.mbatch = [_xh, _yh,self.mb_task_id]
         
                            self._unpack_minibatch()
                            self._before_forward(**kwargs)
                            preds = self.classification_mean_std()
                            self.mb_output = preds
                            self.certainty = torch.max(preds).item()
                            self._after_forward(**kwargs)
                            hard_loss = self.criterion()
                            self.loss += sum(hard_loss)

                        if gs==0: first_train_loss = self.loss.detach().cpu().numpy()

                        #TODO add mas regularization
                        if len(self.star_vars)!=0 and len(self.omegas)!=0:
                            for pindex, p in enumerate(self.model.parameters()):
                                self.loss+=self.MAS_weight/2.*torch.sum(torch.from_numpy(self.omegas[pindex]).type(torch.float32)*(p-self.star_vars[pindex])**2)

                        # Loss & Backward
                        self.optimizer.zero_grad()
                        self._before_backward(**kwargs)
                        self.backward()
                        self._after_backward(**kwargs)

                        # Optimization step
                        self._before_update(**kwargs)
                        self.optimizer_step()
                        self._after_update(**kwargs)
                       
                    
                    if self.use_hard_buffer and len(self.hard_buffer) != 0: 
                        if xh[0].ndim == 3:
                            xt = x+[torch.tensor(_).unsqueeze(0).to(device) for _ in xh]
                        else:
                            xt = x+[torch.tensor(_).to(device) for _ in xh]
                        if yh[0].ndim == 0:
                            yt =y+[torch.tensor(_).unsqueeze(0).to(device) for _ in yh]
                        else:
                            yt =y+[torch.tensor(_).to(device) for _ in yh]
                        # for m,each in enumerate(xt):
                        #     assert isinstance(each,torch.Tensor),(m,each, type(each), len(x), len(xh))
                    else:
                        xt = x[:]
                        yt = y[:]

                    self.mbatch = [torch.cat(xt),  torch.cat(yt), self.mb_task_id]
                    

                    
                    yt_pred = self.classification_mean_std()
                    accuracy = np.mean(np.argmax(yt_pred.detach().cpu().numpy(), axis=1)==yt)
                    self.losses.append(accuracy)

                    self.loss_window.append(np.mean(first_train_loss))
                    if len(self.loss_window)>self.loss_window_length[0]:
                         del(self.loss_window[0])

                    loss_window_mean = np.mean(self.loss_window)
                    loss_window_var = np.var(self.loss_window)
                    if not self.new_peak_detected and loss_window_mean > self.last_loss_window_mean+np.sqrt(self.last_loss_window_var):
                        self.new_peak_detected = True
                    if loss_window_mean < self.loss_window_mean_threshold \
                        and loss_window_var < self.loss_window_variance_threshold and self.new_peak_detected:
                        count_updates+=1 
                        self.update_tags.append(0.01)
                        self.last_loss_window_mean=loss_window_mean
                        self.last_loss_window_var = loss_window_var
                        self.new_peak_detected = False

                        grads = [0 for p in self.models.parameters()]
                        for i,sx in enumerate([_['x'] for _ in self.hard_buffer]):
                            self.model.zero_grad()
                            self.mbatch = [sx,i['y'],self.mb_task_id]   
                                                    
                            y_pred = self.classification_mean_std()
                            torch.norm(y_pred, 2, dim=1).backward()
                            for pindex, p in enumerate(self.model.parameters()):
                                g=p.grad.data.clone().detach().numpy()
                                grads[pindex]+=np.abs(g)
                        omegas_old = self.omegas[:]
                        self.omegas=[]
                        self.star_vars=[]
                        for pindex, p in enumerate(self.model.parameters()):
                            if len(omegas_old)!=0:
                                self.omegas.append(1/count_updates*grads[pindex]+(1-1/count_updates)*omegas_old[pindex])
                            else:
                                self.omegas.append(grads[pindex])
                            self.star_vars.append(p.data.clone().detach())
                    else:
                        self.update_tags.append(0)
                    self.loss_window_means.append(loss_window_mean)
                    self.loss_window_vars.append(loss_window_var)

                    if self.use_hard_buffer:
                        if len(self.hard_buffer)==0:
                            loss=recent_loss.detach().cpu().numpy()
                        else:
                            loss = torch.cat((recent_loss, hard_loss))
                            loss = loss.detach().cpu().numpy()
                        
                        self.hard_buffer=[]

                        #loss=np.mean(loss)                        
                        sorted_inputs = [np.asarray(lx) for  _,lx in reversed(sorted(zip(loss.tolist(),torch.cat(xt).detach().cpu()),key=lambda f:f[0]))]
                        sorted_targets = [ly for _,ly in reversed(sorted(zip(loss.tolist(),torch.cat(yt).detach().cpu()),key=lambda f:f[0]))] 
                        
                        for i in range(min(self.hard_buffer_size,len(sorted_inputs))):
                            self.hard_buffer.append({'x':deepcopy(sorted_inputs[i]),\
                                                    'y':deepcopy(sorted_targets[i])})
                if hasattr(self,  'recent_preds'):
                    self.mb_output = self.recent_preds
                    self.mbatch = [_x, _y, self.mb_task_id]
                    self._after_training_iteration(**kwargs)





    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)          

            if self.task_type=='classification':
                preds = self.classification_mean_std()

            self.mb_output = preds          
            #self.mb_output = self.forward()
            
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