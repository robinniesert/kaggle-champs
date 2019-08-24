
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
from utils import scale_features, set_seed, store_submit, store_oof
from callbacks import GradientClipping, GroupMeanLogMAE, WarmRestartsLRScheduler
from losses_and_metrics import rmse, mae, contribs_rmse_loss
import constants as C


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--lr_div', type=int, default=5, help='lr division factor')
parser.add_argument('--mom', type=float, default=0.95, help='momentum')
parser.add_argument('--cycle_len', type=int, default=1)
parser.add_argument('--cycle_mult', type=int, default=1)
parser.add_argument('--d_model', type=int, default=650, 
                    help='dimenstion of node state vector')
parser.add_argument('--N', type=int, default=10, 
                    help='number of encoding layers')
parser.add_argument('--h', type=int, default=10, 
                    help='number of attention heads')
parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--fold_id', type=int, default=1)
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()


# check if distributed training is possible and set model description 
distributed_train = torch.cuda.device_count() > 1
model_str = f'mol_transformer_v{args.version}_fold{args.fold_id}'
model_se_str = f'mol_transformer_se_v{args.version}_fold{args.fold_id}'


# import data
train_df = pd.read_csv(C.PROC_DATA_PATH+'train_proc_df.csv', index_col=0)
test_df  = pd.read_csv(C.PROC_DATA_PATH+'test_proc_df.csv', index_col=0)
atom_df  = pd.read_csv(C.PROC_DATA_PATH+'atom_df.csv', index_col=0)
edge_df  = pd.read_csv(C.PROC_DATA_PATH+'edge_df.csv', index_col=0)
angle_in_df   = pd.read_csv(C.PROC_DATA_PATH+'angle_in_df.csv', index_col=0)
angle_out_df  = pd.read_csv(C.PROC_DATA_PATH+'angle_out_df.csv', index_col=0)
graph_dist_df = pd.read_csv(
    C.PROC_DATA_PATH+'graph_dist_df.csv', index_col=0, dtype=np.int32)

structures_df = pd.read_csv(C.RAW_DATA_PATH+'structures.csv')
mol_id_map = {m_name: m_id for m_id, m_name 
              in enumerate(sorted(structures_df['molecule_name'].unique()))}
structures_df['molecule_id'] = structures_df['molecule_name'].map(mol_id_map)

train_mol_ids = pd.read_csv(C.PROC_DATA_PATH+'train_idxs_8_fold_cv.csv', 
                            usecols=[0, args.fold_id], index_col=0
                            ).dropna().astype(int).iloc[:,0]
val_mol_ids   = pd.read_csv(C.PROC_DATA_PATH+'val_idxs_8_fold_cv.csv', 
                            usecols=[0, args.fold_id], index_col=0
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
    train_df, C.SC_FEATS_TO_SCALE, train_mol_ids, return_mean_and_std=True)
test_df = scale_features(
    test_df, C.SC_FEATS_TO_SCALE, means=sc_feat_means, stds=sc_feat_stds)
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


# initialize distributed
if distributed_train:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


# train model
callback_fns = [
    partial(GradientClipping, clip=10), 
    partial(GroupMeanLogMAE, snapshot_ensemble=True),
    partial(SaveModelCallback, every='epoch', mode='min',
            monitor='group_mean_log_mae', name=model_se_str),
    partial(WarmRestartsLRScheduler, n_cycles=args.epochs, 
            lr=(args.lr, args.lr/args.lr_div), mom=(args.mom, 0.95),
            cycle_len=args.cycle_len, cycle_mult=args.cycle_mult)
]
learn = Learner(db, model, metrics=[rmse, mae], callback_fns=callback_fns, 
                wd=wd, loss_func=contribs_rmse_loss)
learn.load(model_str)
torch.cuda.empty_cache()
if distributed_train: learn = learn.to_distributed(args.local_rank)

learn.fit(args.epochs)


# make predictions
n_val = len(train_df[train_df['molecule_id'].isin(val_mol_ids)])
val_preds, test_preds = np.zeros(n_val), np.zeros(len(test_df))
for m in range(args.epochs):
    print(f'Predicting for model {m}')
    learn.load(model_se_str+f'_{m}')
    val_contrib_preds = learn.get_preds(DatasetType.Valid)
    test_contrib_preds = learn.get_preds(DatasetType.Test)
    val_preds += val_contrib_preds[0][:,-1].detach().numpy()
    test_preds += test_contrib_preds[0][:,-1].detach().numpy()
val_preds = val_preds * C.SC_STD / args.epochs + C.SC_MEAN
test_preds = test_preds * C.SC_STD / args.epochs + C.SC_MEAN



# store results
store_submit(test_preds, model_se_str, print_head=True)
store_oof(val_preds, model_se_str, print_head=True)