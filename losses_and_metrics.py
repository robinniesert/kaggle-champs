import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.basic_train import Callback, add_metrics
import constants as C

def _reshape_targs(targs, mask_val=C.BATCH_PAD_VAL):
    targs = targs.view(-1, targs.size(-1))
    return targs[targs[:,0]!=mask_val]

def group_mean_log_mae(y_true, y_pred, types, epoch, sc_mean=0, sc_std=1):
    def proc(x): 
        if isinstance(x, torch.Tensor): return x.cpu().numpy().ravel() 
    y_true, y_pred, types = proc(y_true), proc(y_pred), proc(types)
    y_true = sc_mean + y_true * sc_std
    y_pred = sc_mean + y_pred * sc_std
    maes = pd.Series(y_true - y_pred).abs().groupby(types).mean()
    gmlmae = np.log(maes).mean()
    return gmlmae

class GroupMeanLogMAE(Callback):
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn, **kwargs): 
        self.learn = learn
        
    def on_train_begin(self, **kwargs): 
        self.learn.recorder.add_metric_names(['group_mean_log_mae'])
        
    def on_epoch_begin(self, **kwargs): 
        self.input, self.output, self.target = [], [], []
    
    def on_batch_end(self, last_target, last_output, last_input, train, 
                     **kwargs):
        if not train:
            sc_types = last_input[-1].view(-1)
            mask = sc_types != C.BATCH_PAD_VAL
            self.input.append(sc_types[mask])
            self.output.append(last_output[:,-1])
            self.target.append(_reshape_targs(last_target)[:,-1])
                
    def on_epoch_end(self, epoch, last_metrics, **kwargs):
        if (len(self.input) > 0) and (len(self.output) > 0):
            inputs = torch.cat(self.input)
            preds = torch.cat(self.output)
            target = torch.cat(self.target)
            metric = group_mean_log_mae(
                preds, target, inputs, epoch, C.SC_MEAN, C.SC_STD)
            return add_metrics(last_metrics, [metric])
        
def contribs_rmse_loss(preds, targs):
    """
    Returns the sum of RMSEs for each sc contribution and total sc value.
    
    Args:
        - preds: tensor of shape (batch_size * n_sc, 5) containing 
            predictions. Last column is the total scalar coupling value.
        - targs: tensor of shape (batch_size * n_sc, 5) containing 
            true values. Last column is the total scalar coupling value.
    """
    targs = _reshape_targs(targs)
    return torch.mean((preds - targs) ** 2, dim=0).sqrt().sum()

def rmse(preds, targs):
    targs = _reshape_targs(targs)
    return torch.sqrt(F.mse_loss(preds[:,-1], targs[:,-1]))

def mae(preds, targs):
    targs = _reshape_targs(targs)
    return torch.abs(preds[:,-1] - targs[:,-1]).mean()
