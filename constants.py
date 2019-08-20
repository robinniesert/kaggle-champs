import numpy as np

TYPES     = np.array(['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', 
                      '3JHN'])
TYPES_MAP = {t: i for i, t in enumerate(TYPES)}

SC_EDGE_FEATS = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 
                 'type_6', 'type_7', 'dist', 'dist_min_rad', 
                 'dist_electro_neg_adj', 'normed_dist', 'diangle', 'cos_angle', 
                 'cos_angle0', 'cos_angle1'#, 'inv_dist', 'normed_inv_dist'
                ]
SC_MOL_FEATS  = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 
                 'type_6', 'type_7', 'dist', 'dist_min_rad', 
                 'dist_electro_neg_adj', 'normed_dist', 'diangle', 'cos_angle', 
                 'cos_angle0', 'cos_angle1', 'num_atoms', 'num_C_atoms', 
                 'num_F_atoms', 'num_H_atoms', 'num_N_atoms', 'num_O_atoms', 
                 'std_bond_length', 'ave_bond_length', 'ave_atom_weight'
                 #, 'total_atom_weight', 'total_bond_length', 
                 # 'ave_inv_bond_length', 'total_inv_bond_length', inv_dist', 
                 # 'normed_inv_dist', 
                ]
ATOM_FEATS    = ['type_H', 'type_C', 'type_N', 'type_O', 'type_F', 'degree_1', 
                 'degree_2', 'degree_3', 'degree_4', 'degree_5', 'SP', 'SP2', 
                 'SP3', 'hybridization_unspecified', 'aromatic', 
                 'formal_charge', 'atomic_num', 'donor', 'acceptor', 
                 'ave_bond_length', 'ave_neighbor_weight'
                 #, 'ave_inv_bond_length',
                 ]
EDGE_FEATS    = ['single', 'double', 'triple', 'aromatic', 'conjugated', 
                 'in_ring', 'dist', 'normed_dist'#,'inv_dist', 'normed_inv_dist'
                ]
TARGET_COL    = 'scalar_coupling_constant'
CONTRIB_COLS  = ['fc', 'sd', 'pso', 'dso']

N_EDGE_FEATURES    = len(EDGE_FEATS)
N_SC_EDGE_FEATURES = len(SC_EDGE_FEATS)
N_SC_MOL_FEATURES  = len(SC_MOL_FEATS)
N_ATOM_FEATURES    = len(ATOM_FEATS)
N_TYPES            = len(TYPES)
N_MOLS             = 130775

MAX_N_ATOMS   = 29
MAX_N_SC      = 135
SC_MEAN       = 16
SC_STD        = 35
BATCH_PAD_VAL = -999

SC_FEATS_TO_SCALE   = ['dist', 'dist_min_rad', 'dist_electro_neg_adj', 
                       'num_atoms', 'num_C_atoms', 'num_F_atoms', 'num_H_atoms', 
                       'num_N_atoms', 'num_O_atoms', 'inv_dist', 
                       'ave_bond_length', 'std_bond_length', 
                       'total_bond_length',  'ave_inv_bond_length', 
                       'total_inv_bond_length', 'ave_atom_weight', 
                       'total_atom_weight']
ATOM_FEATS_TO_SCALE = ['atomic_num', 'ave_bond_length', 'ave_inv_bond_length', 
                       'ave_neighbor_weight']
EDGE_FEATS_TO_SCALE = ['dist', 'inv_dist']

RAW_DATA_PATH  = 'data/'
PATH           = 'tmp/'
PROC_DATA_PATH = 'tmp/'