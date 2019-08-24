import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import constants as C

def reshape_targs(targs, mask_val=C.BATCH_PAD_VAL):
    targs = targs.view(-1, targs.size(-1))
    return targs[targs[:,0]!=mask_val]

def group_mean_log_mae(y_true, y_pred, types, sc_mean=0, sc_std=1):
    def proc(x): 
        if isinstance(x, torch.Tensor): return x.cpu().numpy().ravel() 
    y_true, y_pred, types = proc(y_true), proc(y_pred), proc(types)
    y_true = sc_mean + y_true * sc_std
    y_pred = sc_mean + y_pred * sc_std
    maes = pd.Series(y_true - y_pred).abs().groupby(types).mean()
    gmlmae = np.log(maes).mean()
    return gmlmae
        
def contribs_rmse_loss(preds, targs):
    """
    Returns the sum of RMSEs for each sc contribution and total sc value.
    
    Args:
        - preds: tensor of shape (batch_size * n_sc, 5) containing 
            predictions. Last column is the total scalar coupling value.
        - targs: tensor of shape (batch_size * n_sc, 5) containing 
            true values. Last column is the total scalar coupling value.
    """
    targs = reshape_targs(targs)
    return torch.mean((preds - targs) ** 2, dim=0).sqrt().sum()

def rmse(preds, targs):
    targs = reshape_targs(targs)
    return torch.sqrt(F.mse_loss(preds[:,-1], targs[:,-1]))

def mae(preds, targs):
    targs = reshape_targs(targs)
    return torch.abs(preds[:,-1] - targs[:,-1]).mean()
