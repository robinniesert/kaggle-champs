import random
import copy
import numpy as np
import pandas as pd
import torch
import constants as C


def set_seed(seed=100):
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)

def store_submit(predictions, name, print_head=False):
    submit = pd.read_csv(C.RAW_DATA_PATH + 'sample_submission.csv') 
    submit['scalar_coupling_constant'] = predictions
    submit.to_csv(f'{name}-submission.csv', index=False)
    if print_head: print(submit.head())

def store_oof(predictions, name, print_head=False):
    oof = pd.DataFrame(predictions, columns=['scalar_coupling_constants'])
    oof.to_csv(f'{name}-oof.csv')
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


def get_dist_matrix(struct_df):
    locs = struct_df[['x','y','z']].values
    n_atoms = len(locs)
    loc_tile = np.tile(locs.T, (n_atoms,1,1))
    dist_mat = np.sqrt(((loc_tile - loc_tile.T)**2).sum(axis=1))
    return dist_mat


def clones(module, n):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
