##
# Many of my features are taken from or inspired by public kernels. The
# following is a probably incomplete list of these kernels:
#   - https://www.kaggle.com/ggeo79/j-coupling-lightbgm-gpu-dihedral-angle for
#   the idea to use dihedral angles on 3J couplings.
#   - https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb
#   mostly for distance features.
#   - https://www.kaggle.com/kmat2019/effective-feature provides the idea to
#   compute cosine angles between scalar coupling atoms and their nearest
#   neighbors.
#   - https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
#   for an efficient distance calculation between scalar coupling atoms.
#
# Running this script will give some warnings related to the
# 'explicit valance..' rdkit error. The problem is dicussed here
# https://www.kaggle.com/c/champs-scalar-coupling/discussion/94274#latest-572435
# I hadn't gotten around to implementing the proper solutions discussed there.


import gc
import numpy as np
import pandas as pd
from itertools import combinations
from glob import glob

import deepchem as dc
from rdkit.Chem import rdmolops, ChemicalFeatures

from xyz2mol import read_xyz_file, xyz2mol
from utils import print_progress
import constants as C


mol_feat_columns = ['ave_bond_length', 'std_bond_length', 'ave_atom_weight']
xyz_filepath_list = list(glob(C.RAW_DATA_PATH + 'structures/*.xyz'))
xyz_filepath_list.sort()


## Functions to create the RDKit mol objects
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

    return mol, np.array(xyz_coordinates), dMat

def get_molecules():
    """
    Constructs rdkit mol objects derrived from the .xyz files. Also returns:
        - mol ids (unique numerical ids)
        - set of molecule level features
        - arrays of xyz coordinates
        - euclidean distance matrices
        - graph distance matrices.
    All objects are returned in dictionaries with 'mol_name' as keys.
    """
    mols, mol_ids, mol_feats = {}, {}, {}
    xyzs, dist_matrices, graph_dist_matrices = {}, {}, {}
    print('Create molecules and distance matrices.')
    for i in range(C.N_MOLS):
        print_progress(i, C.N_MOLS)
        filepath = xyz_filepath_list[i]
        mol_name = filepath.split('/')[-1][:-4]
        mol, xyz, dist_matrix = mol_from_xyz(filepath)
        mols[mol_name] = mol
        xyzs[mol_name] = xyz
        dist_matrices[mol_name] = dist_matrix
        mol_ids[mol_name] = i

        # make padded graph distance matrix dataframes
        n_atoms = len(xyz)
        graph_dist_matrix = pd.DataFrame(np.pad(
            rdmolops.GetDistanceMatrix(mol),
            [(0, 0), (0, C.MAX_N_ATOMS - n_atoms)], 'constant'
        ))
        graph_dist_matrix['molecule_id'] = n_atoms * [i]
        graph_dist_matrices[mol_name] = graph_dist_matrix

        # compute molecule level features
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        atomic_num_list, _, _ = read_xyz_file(filepath)
        dists = dist_matrix.ravel()[np.tril(adj_matrix).ravel()==1]
        mol_feats[mol_name] = pd.Series(
            [np.mean(dists), np.std(dists), np.mean(atomic_num_list)],
            index=mol_feat_columns
        )

    return mols, mol_ids, mol_feats, xyzs, dist_matrices, graph_dist_matrices


## Functions to create features at the scalar coupling level.
def map_atom_info(df, atom_idx, struct_df):
    """Adds xyz-coordinates of atom_{atom_idx} to 'df'."""
    df = pd.merge(df, struct_df, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name', 'atom_index'])
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

def add_dist(df, struct_df):
    """Adds euclidean distance between scalar coupling atoms to 'df'."""
    df = map_atom_info(df, 0, struct_df)
    df = map_atom_info(df, 1, struct_df)
    p_0 = df[['x_0', 'y_0', 'z_0']].values
    p_1 = df[['x_1', 'y_1', 'z_1']].values
    df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)
    df.drop(columns=['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], inplace=True)
    return df

def transform_per_atom_group(df, a_idx, col='dist', trans='mean'):
    """Apply transformation 'trans' on feature in 'col' to scalar coupling
    constants grouped at the atom level."""
    return df.groupby(
        ['molecule_name', f'atom_index_{a_idx}'])[col].transform(trans)

def inv_dist_per_atom(df, a_idx, d_col='dist', power=3):
    """Compute sum of inverse distances of scalar coupling constants grouped at
    the atom level."""
    trans = lambda x: 1 / sum(x ** -power)
    return transform_per_atom_group(df, a_idx, d_col, trans=trans)

def inv_dist_harmonic_mean(df, postfix=''):
    """Compute the harmonic mean of inverse distances of atom_0 and atom_1."""
    c0, c1 = 'inv_dist0' + postfix, 'inv_dist1' + postfix
    return (df[c0] * df[c1]) / (df[c0] + df[c1])

def add_atom_counts(df, struct_df):
    """Add atom counts (total and per type) to 'df'."""
    pd.options.mode.chained_assignment = None
    atoms_per_mol_df = struct_df.groupby(['molecule_name', 'atom']).count()
    atoms_per_mol_map = atoms_per_mol_df['atom_index'].unstack().fillna(0)
    atoms_per_mol_map = atoms_per_mol_map.astype(int).to_dict()
    df['num_atoms'] = 0
    for a in atoms_per_mol_map:
        df[f'num_{a}_atoms'] = df['molecule_name'].map(atoms_per_mol_map[a])
        df['num_atoms'] += df[f'num_{a}_atoms']
    return df

# source: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def dihedral(p):
    """Praxeolitic formula: 1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def cosine_angle(p):
    p0, p1, p2 = p[0], p[1], p[2]
    v1, v2 = p0 - p1, p2 - p1
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

def add_sc_angle_features(df, xyzs, dist_matrices):
    """
    Adds the following angle features to 'df':
    - diangle: for 3J couplings
    - cos_angle: for 2J couplings, angle between sc atom 0, atom in between sc
        atoms and sc atom 1
    - cos_angle0: for all coupling types, cos angle between sc atoms and atom
        closest to atom 0 (except for 1J coupling)
    - cos_angle1: for all coupling types, cos angle between sc atoms and atom
        closest to atom 1
    """
    df['diangle'] = 0.0
    df['cos_angle'] = 0.0
    df['cos_angle0'] = 0.0
    df['cos_angle1'] = 0.0

    diangles, cos_angles, cos_angles0, cos_angles1 = {}, {}, {}, {}
    print('Add scalar coupling angle based features.')
    n = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        print_progress(i, n, 500000)
        mol_name = row['molecule_name']
        mol, xyz = mols[mol_name], xyzs[mol_name]
        dist_matrix = dist_matrices[mol_name]
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        idx0, idx1 = row['atom_index_0'], row['atom_index_1']
        atom_ids = rdmolops.GetShortestPath(mol, idx0, idx1)

        if len(atom_ids)==4:
            diangles[idx] = dihedral(xyz[atom_ids,:])
        elif len(atom_ids)==3:
            cos_angles[idx] = cosine_angle(xyz[atom_ids,:])

        if row['type'] not in [0, 2]:
            neighbors0 = np.where(adj_matrix[idx0]==1)[0]
            if len(neighbors0) > 0:
                idx0_closest = neighbors0[
                    dist_matrix[idx0][neighbors0].argmin()]
                cos_angles0[idx] = cosine_angle(
                    xyz[[idx0_closest, idx0, idx1],:])
        neighbors1 = np.setdiff1d(np.where(adj_matrix[idx1]==1)[0], [idx0])
        if len(neighbors1) > 0:
            idx1_closest = neighbors1[
                dist_matrix[idx1][neighbors1].argmin()]
            cos_angles1[idx] = cosine_angle(
                xyz[[idx0, idx1, idx1_closest],:])

    df['diangle'] = pd.Series(diangles).abs()
    df['cos_angle'] = pd.Series(cos_angles)
    df['cos_angle0'] = pd.Series(cos_angles0)
    df['cos_angle1'] = pd.Series(cos_angles1)
    df.fillna(0., inplace=True)
    return df

def add_sc_features(df, structures_df, mol_feats, xyzs, dist_matrices, mol_ids):
    """Add scalar coupling edge and molecule level features to 'df'."""
    # add euclidean distance between scalar coupling atoms
    df = add_dist(df, structures_df)

    # compute distance normalized by scalar coupling type mean and std
    gb_type_dist = df.groupby('type')['dist']
    df['normed_dist'] = ((df['dist'] - gb_type_dist.transform('mean'))
                         / gb_type_dist.transform('std'))

    # add distance features adjusted for atom radii and electronegativity
    df['R0'] = df['atom_0'].map(C.ATOMIC_RADIUS)
    df['R1'] = df['atom_1'].map(C.ATOMIC_RADIUS)
    df['E0'] = df['atom_0'].map(C.ELECTRO_NEG)
    df['E1'] = df['atom_1'].map(C.ELECTRO_NEG)
    df['dist_min_rad'] = df['dist'] - df['R0'] - df['R1']
    df['dist_electro_neg_adj'] = df['dist'] * (df['E0'] + df['E1']) / 2
    df.drop(columns=['R0','R1','E0','E1'], inplace=True)

    # map scalar coupling types to integers and add dummy variables
    df['type'] = df['type'].map(C.TYPES_MAP)
    df = pd.concat((df, pd.get_dummies(df['type'], prefix='type')), axis=1)

    # add angle related features
    df = add_sc_angle_features(df, xyzs, dist_matrices)

    # add molecule level features
    mol_feat_df = pd.concat(mol_feats, axis=1).T
    mol_feat_dict = mol_feat_df.to_dict()
    for f in mol_feat_columns:
        df[f] = df['molecule_name'].map(mol_feat_dict[f])

    # add atom counts per molecule
    df = add_atom_counts(df, structures_df)

    # add molecule ids
    df['molecule_id'] = df['molecule_name'].map(mol_ids)

    return df

def store_train_and_test(all_df):
    """Split 'all_df' back to train and test and store the resulting dfs."""
    train_df = all_df.iloc[:C.N_SC_TRAIN]
    test_df = all_df.iloc[C.N_SC_TRAIN:]
    train_df.drop(columns='molecule_name', inplace=True)
    test_df.drop(columns='molecule_name', inplace=True)
    test_df.drop(columns='scalar_coupling_constant', inplace=True)

    # Add scalar coupling contributions to train and normalize
    contribs_df = pd.read_csv(
        C.RAW_DATA_PATH + 'scalar_coupling_contributions.csv')
    train_df = pd.concat((train_df, contribs_df[C.CONTRIB_COLS]), axis=1)
    train_df[[C.TARGET_COL, 'fc']] = \
        (train_df[[C.TARGET_COL, 'fc']] - C.SC_MEAN) / C.SC_STD
    train_df[C.CONTRIB_COLS[1:]] = train_df[C.CONTRIB_COLS[1:]] / C.SC_STD

    train_df.to_csv(C.PROC_DATA_PATH + 'train_proc_df.csv')
    test_df.to_csv(C.PROC_DATA_PATH + 'test_proc_df.csv')


## Functions to create atom and bond level features
def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot

def get_bond_features(mol, eucl_dist):
    """
    Compute the following features for each bond in 'mol':
        - bond type: categorical {1: single, 2: double, 3: triple,
            4: aromatic} (one-hot)
        - is conjugated: bool {0, 1}
        - is in ring: bool {0, 1}
        - euclidean distance: float
        - normalized eucl distance: float
    """
    n_bonds = mol.GetNumBonds()
    features = np.zeros((n_bonds, C.N_BOND_FEATURES))
    bond_idx = np.zeros((n_bonds, 2))
    for n, e in enumerate(mol.GetBonds()):
        i = e.GetBeginAtomIdx()
        j = e.GetEndAtomIdx()
        dc_e_feats = dc.feat.graph_features.bond_features(e).astype(int)
        features[n, :6] = dc_e_feats
        features[n, 6] = eucl_dist[i, j]
        bond_idx[n] = i, j
    sorted_idx = bond_idx[:,0].argsort()
    dists = features[:, 6]
    features[:, 7] = (dists - dists.mean()) / dists.std() # normed_dist
    return features[sorted_idx], bond_idx[sorted_idx]

def get_atom_features(mol, dist_matrix):
    """
    Compute the following features for each atom in 'mol':
        - atom type: H, C, N, O, F (one-hot)
        - degree: 1, 2, 3, 4, 5 (one-hot)
        - Hybridization: SP, SP2, SP3, UNSPECIFIED (one-hot)
        - is aromatic: bool {0, 1}
        - formal charge: int
        - atomic number: float
        - average bond length: float
        - average weight of neigboring atoms: float
        - donor: bool {0, 1}
        - acceptor: bool {0, 1}
    """
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, C.N_ATOM_FEATURES))
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        if sum(adj_matrix[idx]) > 0:
            ave_bond_length = np.mean(dist_matrix[idx][adj_matrix[idx]==1])
            ave_neighbor_wt = np.mean(
                [n.GetAtomicNum() for n in a.GetNeighbors()])
        else:
            ave_bond_length, ave_neighbor_wt = 0.0, 0.0

        sym = a.GetSymbol()
        a_feats = one_hot_encoding(sym, C.SYMBOLS) \
            + one_hot_encoding(a.GetDegree(), C.DEGREES) \
            + one_hot_encoding(a.GetHybridization(), C.HYBRIDIZATIONS) \
            + [a.GetIsAromatic(), a.GetFormalCharge(), a.GetAtomicNum(),
               ave_bond_length, ave_neighbor_wt]
        features[idx, :len(a_feats)] = np.array(a_feats)

    feat_factory = ChemicalFeatures.BuildFeatureFactory(C.FDEF)
    try:
        chem_feats = feat_factory.GetFeaturesForMol(mol)
        for t in range(len(chem_feats)):
            if chem_feats[t].GetFamily() == 'Donor':
                for i in chem_feats[t].GetAtomIds():
                    features[i, -2] = 1
            elif chem_feats[t].GetFamily() == 'Acceptor':
                for i in chem_feats[t].GetAtomIds():
                    features[i, -1] = 1
    except RuntimeError as e:
        print(e)

    return features

def get_atom_and_bond_features(mols, mol_ids, dist_matrices):
    atom_features, bond_features = [], []
    bond_idx, atom_to_m_id, bond_to_m_id = [], [], []

    print('Get atom and bond features.')
    for it, m_name in enumerate(mols):
        print_progress(it, C.N_MOLS)
        m_id, mol = mol_ids[m_name], mols[m_name]
        dist_matrix = dist_matrices[m_name]
        n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()

        atom_features.append(get_atom_features(mol, dist_matrix))

        e_feats, b_idx = get_bond_features(mol, dist_matrix)
        bond_features.append(e_feats)
        bond_idx.append(b_idx)

        atom_to_m_id.append(np.repeat(m_id, n_atoms))
        bond_to_m_id.append(np.repeat(m_id, n_bonds))

    atom_features = pd.DataFrame(
        np.concatenate(atom_features), columns=C.ATOM_FEATS)
    bond_features = pd.DataFrame(
        np.concatenate(bond_features), columns=C.BOND_FEATS)
    bond_idx = np.concatenate(bond_idx)
    bond_features['idx_0'] = bond_idx[:,0]
    bond_features['idx_1'] = bond_idx[:,1]
    atom_features['molecule_id'] = np.concatenate(atom_to_m_id)
    bond_features['molecule_id'] = np.concatenate(bond_to_m_id)

    return atom_features, bond_features

def store_atom_and_bond_features(atom_df, bond_df):
    atom_df.to_csv(C.PROC_DATA_PATH + 'atom_df.csv')
    bond_df.to_csv(C.PROC_DATA_PATH + 'bond_df.csv')


## Functions to store distance matrices
def store_graph_distances(graph_dist_matrices):
    graph_dist_df = pd.concat(graph_dist_matrices)
    graph_dist_df.reset_index(drop=True, inplace=True)
    graph_dist_df.replace(1e8, 10, inplace=True) # fix for one erroneous atom
    graph_dist_df = graph_dist_df.astype(int)
    graph_dist_df.to_csv(C.PROC_DATA_PATH + 'graph_dist_df.csv')

def store_eucl_distances(dist_matrices, atom_df):
    dist_df = pd.DataFrame(np.concatenate(
        [np.pad(dm, [(0,0), (0, C.MAX_N_ATOMS-dm.shape[1])], mode='constant')
        for dm in dist_matrices.values()]
    ))
    dist_df['molecule_id'] = atom_df['molecule_id']
    dist_df.to_csv(C.PROC_DATA_PATH + 'dist_df.csv')


## Functions to compute cosine angles for all bonds
def _get_combinations(idx_0_group):
    s = list(idx_0_group['idx_1'])[1:]
    return [list(combinations(s, r))[-1] for r in range(len(s), 0, -1)]

def get_all_cosine_angles(bond_df, structures_df, mol_ids, store=True):
    """Compute cosine angles between all bonds. Grouped at the bond level."""
    bond_idx = bond_df[['molecule_id', 'idx_0', 'idx_1']].astype(int)

    in_out_idx = pd.concat((
        bond_idx,
        bond_idx.rename(columns={'idx_0': 'idx_1', 'idx_1': 'idx_0'})
    ), sort=False)
    gb_mol_0_bond_idx = in_out_idx.groupby(['molecule_id', 'idx_0'])

    angle_idxs = []
    print('Get cosine angle indices.')
    for it, (mol_id, idx_0) in enumerate(gb_mol_0_bond_idx.groups):
        # iterate over all atoms (atom_{idx0})
        print_progress(it, gb_mol_0_bond_idx.ngroups, print_iter=500000)
        idx_0_group = gb_mol_0_bond_idx.get_group((mol_id, idx_0))
        combs = _get_combinations(idx_0_group)
        for i, comb in enumerate(combs):
            # iterate over all bonds of the atom_{idx0} (bond_{idx_0, idx_1})
            idx_1 = idx_0_group['idx_1'].iloc[i]
            for idx_2 in comb:
                # iterate over all angles between bonds with bond_{idx_0, idx_1}
                # as base
                angle_idxs.append((mol_id, idx_0, idx_1, idx_2))

    angle_cols = ['molecule_id', 'atom_index_0', 'atom_index_1', 'atom_index_2']
    angle_df = pd.DataFrame(angle_idxs, columns=angle_cols)
    angle_df['molecule_name'] = angle_df['molecule_id'].map(
        {v:k for k,v in mol_ids.items()})
    angle_df.drop(columns='molecule_id', inplace=True)

    for i in range(3): angle_df = map_atom_info(angle_df, i, structures_df)
    drop_cols = ['atom_0', 'atom_1', 'atom_2', 'molecule_id_x', 'molecule_id_y']
    angle_df.drop(columns=drop_cols, inplace=True)

    for c in ['x', 'y', 'z']:
        angle_df[f'{c}_0_1'] = \
            angle_df[f'{c}_0'].values - angle_df[f'{c}_1'].values
        angle_df[f'{c}_0_2'] = \
            angle_df[f'{c}_0'].values - angle_df[f'{c}_2'].values

    def cos_angles(v1, v2):
        return (v1 * v2).sum(1) / np.sqrt((v1 ** 2).sum(1) * (v2 ** 2).sum(1))

    angle_df['cos_angle'] = cos_angles(
        angle_df[['x_0_1', 'y_0_1', 'z_0_1']].values,
        angle_df[['x_0_2', 'y_0_2', 'z_0_2']].values
    )
    angle_df = angle_df[['molecule_id', 'atom_index_0', 'atom_index_1',
                         'atom_index_2', 'cos_angle']]

    gb_mol_angle = angle_df.groupby('molecule_id')
    gb_mol_bond_idx = bond_idx.groupby('molecule_id')

    angle_to_in_bond, angle_to_out_bond = [], []
    print('Get cosine angles.')
    for i, mol_id in enumerate(mol_ids.values()):
        print_progress(i, C.N_MOLS)
        b_df = gb_mol_bond_idx.get_group(mol_id)
        a_df = gb_mol_angle.get_group(mol_id)
        b_in_idxs = b_df[['idx_0', 'idx_1']].values
        b_out_idxs = b_df[['idx_1', 'idx_0']].values
        a1 = a_df[['atom_index_0', 'atom_index_1', 'cos_angle']].values
        a2 = a_df[['atom_index_0', 'atom_index_2', 'cos_angle']].values
        for a in np.concatenate((a1, a2)):
            if any(np.all(b_in_idxs==a[:2], axis=1)):
                a_to_in_idx = np.where(np.all(b_in_idxs==a[:2], axis=1))[0][0]
                angle_to_in_bond.append((mol_id, a_to_in_idx, a[-1]))
            if any(np.all(b_out_idxs==a[:2], axis=1)):
                a_to_out_idx = np.where(np.all(b_out_idxs==a[:2], axis=1))[0][0]
                angle_to_out_bond.append((mol_id, a_to_out_idx, a[-1]))

    angle_in_df = pd.DataFrame(
        angle_to_in_bond, columns=['molecule_id', 'b_idx', 'cos_angle'])
    angle_out_df = pd.DataFrame(
        angle_to_out_bond, columns=['molecule_id', 'b_idx', 'cos_angle'])

    if store: store_angles(angle_in_df, angle_out_df)
    return angle_in_df, angle_out_df

def store_angles(angle_in_df, angle_out_df):
    angle_in_df.to_csv(C.PROC_DATA_PATH + 'angle_in_df.csv')
    angle_out_df.to_csv(C.PROC_DATA_PATH + 'angle_out_df.csv')


def process_and_store_structures(structures_df, mol_ids):
    structures_df['molecule_id'] = structures_df['molecule_name'].map(mol_ids)
    structures_df.to_csv(C.PROC_DATA_PATH + 'structures_proc_df.csv')
    return structures_df

def _clear_memory(var_strs):
    for var_str in var_strs: del globals()[var_str]
    gc.collect()


if __name__=='__main__':

    # import data
    train_df = pd.read_csv(C.RAW_DATA_PATH + 'train.csv', index_col=0)
    test_df = pd.read_csv(C.RAW_DATA_PATH + 'test.csv', index_col=0)
    structures_df = pd.read_csv(C.RAW_DATA_PATH + 'structures.csv')

    # concatenate train and test into one dataframe
    all_df = pd.concat((train_df, test_df), sort=True)
    if 'id' in all_df.columns: all_df.drop(columns='id', inplace=True)
    _clear_memory(['train_df', 'test_df'])

    # create molecules
    mols, mol_ids, mol_feats, xyzs, dist_matrices, graph_dist_matrices = \
        get_molecules()

    # create and store features
    all_df = add_sc_features(
        all_df, structures_df, mol_feats, xyzs, dist_matrices, mol_ids)
    store_train_and_test(all_df)
    _clear_memory(['all_df'])

    atom_df, bond_df = get_atom_and_bond_features(mols, mol_ids, dist_matrices)
    store_atom_and_bond_features(atom_df, bond_df)

    store_graph_distances(graph_dist_matrices)
    store_eucl_distances(dist_matrices, atom_df) # only used for MPNN model

    structures_df = process_and_store_structures(structures_df, mol_ids)
    _, _ = get_all_cosine_angles(bond_df, structures_df, mol_ids, store=True)