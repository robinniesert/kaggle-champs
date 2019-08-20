
import gc
import argparse
import pandas as pd
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fastai.callbacks import SaveModelCallback
from fastai.basic_data import DataBunch, DeviceDataLoader, DatasetType
from fastai.basic_train import Learner
from fastai.train import *
from fastai.distributed import *

from moldataset import MoleculeDataset, collate_parallel_fn
from model import Transformer
from utils import scale_features, set_seed
from callbacks import GradientClipping
from losses_and_metrics import rmse, mae, contribs_rmse_loss, GroupMeanLogMAE
import constants as C


# constants
FOLD_ID      = 1
VERSION      = 1
MODEL_STRING = f'mol_transformer_parallel_v{VERSION}_fold{FOLD_ID}'


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--d_model', type=int, default=768, 
                    help='dimenstion of node state vector')
parser.add_argument('--N', type=int, default=12, 
                    help='number of encoding layers')
parser.add_argument('--h', type=int, default=12, help='number attention heads')
parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args()


# import data
train_df = pd.read_csv(C.PROC_DATA_PATH+'train_proc_df.csv', index_col=0)
test_df  = pd.read_csv(C.PROC_DATA_PATH+'test_proc_df.csv', index_col=0)
atom_df  = pd.read_csv(C.PROC_DATA_PATH+'atom_df.csv', index_col=0)
edge_df  = pd.read_csv(C.PROC_DATA_PATH+'edge_df.csv', index_col=0)
angle_in_df  = pd.read_csv(C.PROC_DATA_PATH+'angle_in_df.csv', index_col=0)
angle_out_df = pd.read_csv(C.PROC_DATA_PATH+'angle_out_df.csv', index_col=0)
graph_dist_df = pd.read_csv(
    C.PROC_DATA_PATH+'graph_dist_df.csv', index_col=0, dtype=np.int32)

structures_df = pd.read_csv(C.RAW_DATA_PATH+'structures.csv')
mol_id_map = {m_name: m_id for m_id, m_name 
              in enumerate(sorted(structures_df['molecule_name'].unique()))}
structures_df['molecule_id'] = structures_df['molecule_name'].map(mol_id_map)

train_mol_ids = pd.read_csv(C.PROC_DATA_PATH+'train_idxs_8_fold_cv.csv', 
                            usecols=[0, FOLD_ID], index_col=0
                            ).dropna().astype(int).iloc[:,0]
val_mol_ids   = pd.read_csv(C.PROC_DATA_PATH+'val_idxs_8_fold_cv.csv', 
                            usecols=[0, FOLD_ID], index_col=0
                            ).dropna().astype(int).iloc[:,0]
test_mol_ids  = pd.Series(test_df['molecule_id'].unique())

contribs_df = pd.read_csv(C.RAW_DATA_PATH+'scalar_coupling_contributions.csv')
train_df = pd.concat((train_df, contribs_df[C.CONTRIB_COLS]), axis=1)
del contribs_df
gc.collect()

train_df[[C.TARGET_COL, 'fc']] = ((train_df[[C.TARGET_COL, 'fc']] - C.SC_MEAN) 
                                  / C.SC_STD)
train_df[C.CONTRIB_COLS[1:]] = train_df[C.CONTRIB_COLS[1:]] / C.SC_STD

num_atom_cols = ['num_C_atoms', 'num_F_atoms', 'num_H_atoms', 'num_N_atoms', 
                 'num_O_atoms']
train_df['num_atoms'] = train_df[num_atom_cols].sum(axis=1)
test_df['num_atoms'] = test_df[num_atom_cols].sum(axis=1)
num_atom_cols += ['num_atoms']
train_df[num_atom_cols] /= 10
test_df[num_atom_cols] /= 10


# scale features
train_df, sc_feat_means, sc_feat_stds = scale_features(
    train_df, C.SC_FEATS_TO_SCALE, train_mol_ids, return_mean_and_std=True
)
test_df = scale_features(
    train_df, C.SC_FEATS_TO_SCALE, train_mol_ids, means=sc_feat_means, 
    stds=sc_feat_stds
)
atom_df = scale_features(atom_df, C.ATOM_FEATS_TO_SCALE, train_mol_ids)
edge_df = scale_features(edge_df, C.EDGE_FEATS_TO_SCALE, train_mol_ids)


# group data by molecule id
gb_mol_sc = train_df.groupby('molecule_id')
test_gb_mol_sc = test_df.groupby('molecule_id')
gb_mol_atom = atom_df.groupby('molecule_id')
gb_mol_edge = edge_df.groupby('molecule_id')
gb_mol_struct = structures_df.groupby('molecule_id')
gb_mol_angle_in = angle_in_df.groupby('molecule_id')
gb_mol_angle_out = angle_out_df.groupby('molecule_id')
gb_mol_graph_dist = graph_dist_df.groupby('molecule_id')


# create dataloaders
set_seed(100)
batch_size = args.batch_size

train_ds = MoleculeDataset(
    train_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_edge, gb_mol_struct, 
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist
)
val_ds   = MoleculeDataset(
    val_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_edge, gb_mol_struct, 
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist
)
test_ds  = MoleculeDataset(
    test_mol_ids, test_gb_mol_sc, gb_mol_atom, gb_mol_edge, gb_mol_struct, 
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist
)

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8)
val_dl   = DataLoader(val_ds, batch_size, num_workers=8)
test_dl  = DeviceDataLoader.create(
    test_ds, batch_size, num_workers=8, 
    collate_fn=partial(collate_parallel_fn, test=True)
)

db = DataBunch(train_dl, val_dl, collate_fn=collate_parallel_fn)
db.test_dl = test_dl


# set up model
set_seed(100)
wd, d_model = args.wd, args.d_model
enn_args = dict(layers=3*[d_model], dropout=3*[0.0], layer_norm=True)
ann_args = dict(layers=1*[d_model], dropout=1*[0.0], layer_norm=True, 
                out_act=nn.Tanh())
model = Transformer(
    C.N_ATOM_FEATURES, C.N_EDGE_FEATURES, C.N_SC_EDGE_FEATURES, 
    C.N_SC_MOL_FEATURES, N=args.N, d_model=d_model, d_ff=d_model*4, 
    d_ff_contrib=d_model//4, h=args.h, dropout=args.dropout, 
    kernel_sz=min(128, d_model), enn_args=enn_args, ann_args=ann_args
)


# train model
callback_fns = [partial(GradientClipping, clip=10), GroupMeanLogMAE]
learn = Learner(db, model, metrics=[rmse, mae], callback_fns=callback_fns, 
                wd=wd, loss_func=contribs_rmse_loss)
if torch.cuda.device_count() > 1: learn = learn.to_parallel()

learn.fit_one_cycle(args.epochs, max_lr=args.lr, callbacks=[
        SaveModelCallback(learn, every='improvement', mode='min',
                          monitor='group_mean_log_mae', name=MODEL_STRING)
    ])
learn.recorder.plot_losses(skip_start=500)


# make predictions
val_contrib_preds = learn.get_preds(DatasetType.Valid)
test_contrib_preds = learn.get_preds(DatasetType.Test)
val_preds = val_contrib_preds[0][:,-1].detach().numpy() * C.SC_STD + C.SC_MEAN
test_preds = test_contrib_preds[0][:,-1].detach().numpy() * C.SC_STD + C.SC_MEAN


# store results
def store_submit(predictions):
    submit = pd.read_csv(C.RAW_DATA_PATH + 'sample_submission.csv')
    print(len(submit), len(predictions))   
    submit['scalar_coupling_constant'] = predictions
    submit.to_csv(f'{MODEL_STRING}-submission.csv', index=False)

def store_oof(predictions, val_ids):
    oof = pd.DataFrame(predictions, columns=['scalar_coupling_constants'])
    print(oof.head())
    oof.to_csv(f'{MODEL_STRING}-oof.csv')

store_submit(test_preds)
store_oof(val_preds, val_mol_ids)