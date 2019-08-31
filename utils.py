import random
import copy
import numpy as np
import pandas as pd
import torch
from time import strftime, localtime

import constants as C


def set_seed(seed=100):
    """Set the seed for all relevant RNGs."""
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)


def print_progress(i, n, print_iter=10000):
    if (i%print_iter)==0:
        print(f'{strftime("%H:%M:%S", localtime())} - {(100 * i / n):.2f} %')


def store_submit(predictions, name, print_head=False):
    if not isinstance(predictions, pd.DataFrame):
        submit = pd.read_csv(C.RAW_DATA_PATH + 'sample_submission.csv')
        submit['scalar_coupling_constant'] = predictions
    else:
        submit = predictions
    submit.to_csv(f'{C.SUB_PATH}{name}-submission.csv', index=False)
    if print_head: print(submit.head())


def store_oof(predictions, name, print_head=False):
    if not isinstance(predictions, pd.DataFrame):
        oof = pd.DataFrame(predictions, columns=['scalar_coupling_constants'])
    else:
        oof = predictions
    oof.to_csv(f'{C.OOF_PATH}{name}-oof.csv')
    if print_head: print(oof.head())


def scale_features(df, features, train_mol_ids=None, means=None, stds=None,
                   return_mean_and_std=False):
    if ((df[features].mean().abs()>0.1).any()
        or ((df[features].std()-1.0).abs()>0.1).any()):
        if train_mol_ids is not None:
            idx = df['molecule_id'].isin(train_mol_ids)
            means = df.loc[idx, features].mean()
            stds = df.loc[idx, features].std()
        else:
            assert means is not None
            assert stds is not None
        df[features] = (df[features] - means) / stds
    if return_mean_and_std: return df, means, stds
    else: return df
