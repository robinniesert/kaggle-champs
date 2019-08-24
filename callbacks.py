import torch
from fastai.basic_train import Learner, LearnerCallback, Callback, add_metrics
from fastai.basic_data import DatasetType
from fastai.callbacks.general_sched import TrainingPhase, GeneralScheduler
from fastai.callback import annealing_cos
from losses_and_metrics import group_mean_log_mae, reshape_targs
import constants as C

class GradientClipping(LearnerCallback):
    "Gradient clipping during training."
    def __init__(self, learn:Learner, clip:float = 0., 
                 start_it:int = 100):
        super().__init__(learn)
        self.clip, self.start_it = clip, start_it

    def on_backward_end(self, iteration, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip and (iteration > self.start_it): 
            torch.nn.utils.clip_grad_norm_(
                self.learn.model.parameters(), self.clip)


class GroupMeanLogMAE(Callback):
    _order = -20 # Needs to run before the recorder

    def __init__(self, learn, snapshot_ensemble=False, **kwargs): 
        self.learn = learn
        self.snapshot_ensemble = snapshot_ensemble

    def on_train_begin(self, **kwargs): 
        metric_names = ['group_mean_log_mae']
        if self.snapshot_ensemble: metric_names += ['group_mean_log_mae_es']
        self.learn.recorder.add_metric_names(metric_names)
        if self.snapshot_ensemble: self.val_preds = []
        
    def on_epoch_begin(self, **kwargs): 
        self.sc_types, self.output, self.target = [], [], []
    
    def on_batch_end(self, last_target, last_output, last_input, train, 
                     **kwargs):
        if not train:
            sc_types = last_input[-1].view(-1)
            mask = sc_types != C.BATCH_PAD_VAL
            self.sc_types.append(sc_types[mask])
            self.output.append(last_output[:,-1])
            self.target.append(reshape_targs(last_target)[:,-1])
                
    def on_epoch_end(self, epoch, last_metrics, **kwargs):
        if (len(self.sc_types) > 0) and (len(self.output) > 0):
            sc_types = torch.cat(self.sc_types)
            preds = torch.cat(self.output)
            target = torch.cat(self.target)
            metrics = [group_mean_log_mae(
                preds, target, sc_types, C.SC_MEAN, C.SC_STD)]
                
            if self.snapshot_ensemble: 
                self.val_preds.append(preds.view(-1,1))
                preds_se = torch.cat(self.val_preds, dim=1).mean(dim=1)
                metrics += [group_mean_log_mae(
                    preds_se, target, sc_types, C.SC_MEAN, C.SC_STD)]
            return add_metrics(last_metrics, metrics)


class WarmRestartsLRScheduler(GeneralScheduler):
    def __init__(self, learn, n_cycles, lr, mom, cycle_len=1, cycle_mult=1):
        n = len(learn.data.train_dl)
        phases = [(TrainingPhase(n * (cycle_len * cycle_mult**i))
                 .schedule_hp('lr', lr, anneal=annealing_cos)
                 .schedule_hp('mom', mom, anneal=annealing_cos)) 
                 for i in range(n_cycles)]
        super().__init__(learn, phases)
