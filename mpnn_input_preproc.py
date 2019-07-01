import os
import numpy as np
import pandas as pd
import deepchem as dc
import gc
from tqdm import tqdm
from scipy.spatial.distance import norm
from glob import glob
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from xyz2mol import xyz2mol, read_xyz_file

## Constants
# scalar coupling types
TYPES     = np.array(['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC',
                      '3JHN'])
TYPES_MAP = {t: i for i, t in enumerate(TYPES)}

# feature definition file
FDEF = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')

# feature collections
SYMBOLS        = ['H', 'C', 'N', 'O', 'F']
DEGREES        = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.UNSPECIFIED
]

# feature maps
ATOMIC_RADIUS = {'H': 0.43, 'C': 0.82, 'N': 0.8, 'O': 0.78, 'F': 0.76} # With fudge factor
ELECTRO_NEG   = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

# feature dimensions
N_EDGE_FEATURES        = 8
N_ATOM_FEATURES        = 21
MAX_N_ATOMS            = 29
MAX_N_BONDS            = 58
N_TYPES                = len(TYPES)

# paths
DATA_PATH = 'data/'
PATH      = 'tmp/'

## Helper functions
def array_to_csv(arr, f_name, n, fmt='%.10f'):
    "Writes numpy array 'arr' to csv file."
    f = PATH + f_name + '.csv'
    np.savetxt(f, arr.reshape(n, -1), delimiter=',', fmt=fmt)

def print_progress(i):
    if (i%10000)==0: print(i)

def clear_memory(var_strs):
    for var_str in var_strs: del globals()[var_str]
    gc.collect()

## import data
train_df = pd.read_csv(DATA_PATH+'train.csv')
test_df = pd.read_csv(DATA_PATH+'test.csv')
structures_df = pd.read_csv(DATA_PATH+'structures.csv')

## Process dataframes
def map_atom_info(df, atom_idx, struct_df):
    df = pd.merge(df, struct_df, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

def add_dist(df, struct_df):
    df = map_atom_info(df, 0, struct_df)
    df = map_atom_info(df, 1, struct_df)
    p_0 = df[['x_0', 'y_0', 'z_0']].values
    p_1 = df[['x_1', 'y_1', 'z_1']].values
    df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)
    df.drop(
        columns=['atom_0', 'atom_1', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'],
        inplace=True
    )
    return df

def add_atom_counts(df, struct_df):
    pd.options.mode.chained_assignment = None
    atoms_per_mol_df = struct_df.groupby(['molecule_name', 'atom']).count()
    atoms_per_mol_map = atoms_per_mol_df['atom_index'].unstack().fillna(0)
    atoms_per_mol_map = atoms_per_mol_map.astype(int).to_dict()
    df['num_atoms'] = 0
    for a in atoms_per_mol_map:
        df[f'num_{a}_atoms'] = df['molecule_name'].map(atoms_per_mol_map[a])
        df['num_atoms'] += df[f'num_{a}_atoms']
    return df

train_df = add_dist(train_df, structures_df)
test_df = add_dist(test_df, structures_df)
train_df = add_atom_counts(train_df, structures_df)
test_df = add_atom_counts(test_df, structures_df)
train_df.drop(columns='id', inplace=True)
test_df.drop(columns='id', inplace=True)
train_df['type'] = train_df['type'].map(TYPES_MAP)
test_df['type'] = test_df['type'].map(TYPES_MAP)

## Create molecules
def mol_from_xyz(filepath, add_hs=True, compute_dist_centre=False):
    """Wrapper function for calling xyz2mol function."""
    charged_fragments = True  # alternatively radicals are made

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    mol, dMat = xyz2mol(atomicNumList, charge, xyz_coordinates,
                        charged_fragments, quick, check_chiral_stereo=False)

    # Compute distance from centroid
    xyz_arr = np.array(xyz_coordinates)
    if compute_dist_centre:
        centroid = xyz_arr.mean(axis=0)
        dFromCentroid = norm(xyz_arr - centroid, axis=1)
    else:
        dFromCentroid = None

    return mol, xyz_arr, dMat, dFromCentroid

# get xyx files and number of molecules
xyz_filepath_list = list(glob(DATA_PATH+'structures/*.xyz'))
xyz_filepath_list.sort()
n_mols = len(xyz_filepath_list)
print('total xyz filepath # ', n_mols)

# transform .xyz to .mol files and store distance matrices
dist_matrices, mols, mol_ids = {}, {}, {}
for i in tqdm(range(n_mols)):
    filepath = xyz_filepath_list[i]
    mol_name = filepath.split('/')[-1][:-4]
    mol, _, dist_matrix, _ = mol_from_xyz(filepath)
    mols[mol_name] = mol
    dist_matrices[mol_name] = dist_matrix
    mol_ids[mol_name] = i

## Store processed dfs
# add molecule ids to dataframes
train_df['molecule_id'] = train_df['molecule_name'].map(mol_ids)
test_df['molecule_id'] = test_df['molecule_name'].map(mol_ids)
train_df.drop(columns=['molecule_name'], inplace=True)
test_df.drop(columns=['molecule_name'], inplace=True)
train_df.to_csv(PATH + 'train_proc_df.csv')
test_df.to_csv(PATH + 'test_proc_df.csv')
clear_memory(['train_df', 'test_df'])

## Engineer features

# functions partially sourced from:
# https://deepchem.io/docs/_modules/deepchem/feat/graph_features.html
def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    if 0:
        if sum(one_hot)==0: print('one_hot_encoding() return NULL!', x, set)
    return one_hot

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_edge_features(mol, eucl_dist):
    """
    Compute the following features for each bond in 'mol':
        - bond type: categorical {1: single, 2: double, 3: triple,
            4: aromatic} (one-hot)
        - is conjugated: bool {0, 1}
        - is in ring: bool {0, 1}
        - euclidean distance: float
        - normalized eucl distance: float
    """
    n_edges = 2 * mol.GetNumBonds()
    features = np.zeros((n_edges, N_EDGE_FEATURES))
    pairs_idx = np.zeros((n_edges, 2))
    for n, e in enumerate(mol.GetBonds()):
        ix1 = 2 * n
        ix2 = (2 * n) + 1
        i = e.GetBeginAtomIdx()
        j = e.GetEndAtomIdx()
        dc_e_feats = dc.feat.graph_features.bond_features(e).astype(int)
        for ix in [ix1, ix2]:
            features[ix, :6] = dc_e_feats
            features[ix, 6] = eucl_dist[i, j]
        pairs_idx[ix1] = i, j
        pairs_idx[ix2] = j, i
    sorted_idx = pairs_idx[:,0].argsort()
    dists = features[:, 6]
    features[:, 7] = (dists - dists.mean()) / dists.std()
    return features[sorted_idx], pairs_idx[sorted_idx]

def get_atom_features(mol):
    """
    Compute the following features for each atom in 'mol':
        - atom type: H, C, N, O, F (one-hot)
        - degree: 1, 2, 3, 4, 5 (one-hot)
        - Hybridization: SP, SP2, SP3, UNSPECIFIED (one-hot)
        - is aromatic: bool {0, 1}
        - formal charge: int
        - atomic number: float
        - atomic radius: float
        - electro negativity: float
        - donor: bool {0, 1}
        - acceptor: bool {0, 1}
    """
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, N_ATOM_FEATURES))
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        a_feats = one_hot_encoding(sym, SYMBOLS) \
            + one_hot_encoding(a.GetDegree(), DEGREES) \
            + one_hot_encoding(a.GetHybridization(), HYBRIDIZATIONS) \
            + [a.GetIsAromatic(), a.GetFormalCharge(), a.GetAtomicNum() / 10,
               ATOMIC_RADIUS[sym], ELECTRO_NEG[sym]]
        features[a.GetIdx(), :len(a_feats)] = np.array(a_feats)

    feat_factory = ChemicalFeatures.BuildFeatureFactory(FDEF)
    chem_feats = feat_factory.GetFeaturesForMol(mol)
    for t in range(len(chem_feats)):
        if chem_feats[t].GetFamily() == 'Donor':
            for i in chem_feats[t].GetAtomIds():
                features[i, -2] = 1
        elif chem_feats[t].GetFamily() == 'Acceptor':
            for i in chem_feats[t].GetAtomIds():
                features[i, -1] = 1

    return features

# create features
atomic_features = np.zeros((n_mols, MAX_N_ATOMS, N_ATOM_FEATURES))
edge_features = np.zeros((n_mols, MAX_N_BONDS, N_EDGE_FEATURES))
pairs_idx = np.zeros((n_mols, MAX_N_BONDS, 2), dtype=np.int16) - 1
mask = np.zeros((n_mols, MAX_N_ATOMS), dtype=np.int16)
edge_mask = np.zeros((n_mols, MAX_N_BONDS), dtype=np.int16)
for i, m_name in enumerate(mols):
    print_progress(i)
    m_id, mol, = mol_ids[m_name], mols[m_name]
    xyz, dist_matrix = xyzs[mol_name], dist_matrices[m_name]
    n_atoms, n_edges = mol.GetNumAtoms(), 2 * mol.GetNumBonds()
    atomic_features[m_id, :n_atoms, :] = get_atom_features(mol)
    edge_features[m_id, :n_edges, :], pairs_idx[m_id, :n_edges, :] = \
        get_edge_features(mol, dist_matrix)
    mask[m_id, :n_atoms], edge_mask[m_id, pairs_idx[m_id,:,0] != -1] = 1, 1

# store arrays
array_to_csv(atomic_features, 'atomic_features', n_mols, fmt='%i')
array_to_csv(edge_features, 'edge_features', n_mols, fmt='%.10f')
array_to_csv(pairs_idx, 'pairs_idx', n_mols, fmt='%i')
array_to_csv(mask, 'mask', n_mols, fmt='%i')
array_to_csv(edge_mask, 'edge_mask', n_mols, fmt='%i')