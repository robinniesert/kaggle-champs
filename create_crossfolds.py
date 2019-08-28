
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import constants as C

train_df = pd.read_csv(C.PROC_DATA_PATH + 'train_proc_df.csv', index_col=0)
mol_ids = train_df['molecule_id'].unique()

folds = KFold(C.N_FOLDS, shuffle=True, random_state=100).split(mol_ids)
folds = [(pd.Series(mol_ids[f[0]]), pd.Series(mol_ids[f[1]])) for f in folds]
train_idxs = pd.concat([f[0] for f in folds], axis=1)#.dropna().astype(int)
val_idxs = pd.concat([f[1] for f in folds], axis=1)#.dropna().astype(int)

train_idxs.to_csv(f'{C.PROC_DATA_PATH}train_idxs_{C.N_FOLDS}_fold_cv.csv')
val_idxs.to_csv(f'{C.PROC_DATA_PATH}val_idxs_{C.N_FOLDS}_fold_cv.csv')