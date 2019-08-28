import numpy as np
import torch
from torch.utils.data import Dataset
import constants as C


def _get_existing_group(gb, i):
    try: group_df = gb.get_group(i)
    except KeyError: group_df = None
    return group_df


def get_dist_matrix(struct_df):
    locs = struct_df[['x','y','z']].values
    n_atoms = len(locs)
    loc_tile = np.tile(locs.T, (n_atoms,1,1))
    dist_mat = np.sqrt(((loc_tile - loc_tile.T)**2).sum(axis=1))
    return dist_mat


class MoleculeDataset(Dataset):
    def __init__(self, mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, 
                 gb_mol_struct, gb_mol_angle_in, gb_mol_angle_out, 
                 gb_mol_graph_dist):
        self.n = len(mol_ids)
        self.mol_ids = mol_ids
        self.gb_mol_sc = gb_mol_sc
        self.gb_mol_atom = gb_mol_atom
        self.gb_mol_bond = gb_mol_bond
        self.gb_mol_struct = gb_mol_struct
        self.gb_mol_angle_in = gb_mol_angle_in
        self.gb_mol_angle_out = gb_mol_angle_out
        self.gb_mol_graph_dist = gb_mol_graph_dist

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.gb_mol_sc.get_group(self.mol_ids[idx]),
                self.gb_mol_atom.get_group(self.mol_ids[idx]), 
                self.gb_mol_bond.get_group(self.mol_ids[idx]), 
                self.gb_mol_struct.get_group(self.mol_ids[idx]), 
                self.gb_mol_angle_in.get_group(self.mol_ids[idx]), 
                _get_existing_group(self.gb_mol_angle_out, self.mol_ids[idx]),
                self.gb_mol_graph_dist.get_group(self.mol_ids[idx]))

def arr_lst_to_padded_batch(arr_lst, dtype=torch.float, 
                            pad_val=C.BATCH_PAD_VAL):
    tensor_list = [torch.Tensor(arr).type(dtype) for arr in arr_lst]
    batch = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=pad_val)
    return batch.contiguous()
                   
def collate_parallel_fn(batch, test=False):
    batch_size, n_atom_sum, n_pairs_sum = len(batch), 0, 0
    x, bond_x, sc_x, sc_m_x = [], [], [], []
    eucl_dists, graph_dists = [], []
    angles_in, angles_out = [], []
    mask, bond_idx, sc_idx = [], [], []
    angles_in_idx, angles_out_idx = [], []
    sc_types, sc_vals = [], []

    for b in range(batch_size):
        (sc_df, atom_df, bond_df, struct_df, angle_in_df, angle_out_df, 
         graph_dist_df) = batch[b]
        n_atoms, n_pairs, n_sc = len(atom_df), len(bond_df), len(sc_df)
        n_pad = C.MAX_N_ATOMS - n_atoms
        eucl_dists_ = get_dist_matrix(struct_df)
        eucl_dists_ = np.pad(eucl_dists_, [(0, 0), (0, n_pad)], 'constant', 
                             constant_values=999)
        
        x.append(atom_df[C.ATOM_FEATS].values)
        bond_x.append(bond_df[C.BOND_FEATS].values)
        sc_x.append(sc_df[C.SC_EDGE_FEATS].values)
        sc_m_x.append(sc_df[C.SC_MOL_FEATS].values)
        sc_types.append(sc_df['type'].values)
        if not test: 
            n_sc_pad = C.MAX_N_SC - n_sc
            sc_vals_ = sc_df[C.CONTRIB_COLS+[C.TARGET_COL]].values
            sc_vals.append(np.pad(sc_vals_, [(0, n_sc_pad), (0, 0)], 'constant', 
                                  constant_values=-999))
        eucl_dists.append(eucl_dists_)
        graph_dists.append(graph_dist_df.values[:,:-1])
        angles_in.append(angle_in_df['cos_angle'].values)
        if angle_out_df is not None: 
            angles_out.append(angle_out_df['cos_angle'].values)
        else: 
            angles_out.append(np.array([C.BATCH_PAD_VAL]))
        
        mask.append(np.pad(np.ones(2 * [n_atoms]), [(0, 0), (0, n_pad)], 
                           'constant'))
        bond_idx.append(bond_df[['idx_0', 'idx_1']].values)
        sc_idx.append(sc_df[['atom_index_0', 'atom_index_1']].values)
        angles_in_idx.append(angle_in_df['b_idx'].values)
        if angle_out_df is not None: 
            angles_out_idx.append(angle_out_df['b_idx'].values)
        else:
            angles_out_idx.append(np.array([0.]))
        
        n_atom_sum += n_atoms
        n_pairs_sum += n_pairs
        
    x = arr_lst_to_padded_batch(x, pad_val=0.)
    bond_x = arr_lst_to_padded_batch(bond_x)
    max_n_atoms = x.size(1)
    max_n_bonds = bond_x.size(1)
    angles_out_idx = [a + max_n_bonds for a in angles_out_idx]
    
    sc_x = arr_lst_to_padded_batch(sc_x)
    sc_m_x =arr_lst_to_padded_batch(sc_m_x)
    if not test: sc_vals = arr_lst_to_padded_batch(sc_vals)
    else: sc_vals = torch.tensor([0.] * batch_size)
    sc_types = arr_lst_to_padded_batch(sc_types, torch.long)
    mask = arr_lst_to_padded_batch(mask, torch.uint8, 0)
    mask = mask[:,:,:max_n_atoms].contiguous()
    bond_idx = arr_lst_to_padded_batch(bond_idx, torch.long, 0)
    sc_idx = arr_lst_to_padded_batch(sc_idx, torch.long, 0)
    angles_in_idx = arr_lst_to_padded_batch(angles_in_idx, torch.long, 0)
    angles_out_idx = arr_lst_to_padded_batch(angles_out_idx, torch.long, 0)
    angles_idx = torch.cat((angles_in_idx, angles_out_idx), dim=-1).contiguous()
    eucl_dists = arr_lst_to_padded_batch(eucl_dists, pad_val=999)
    eucl_dists = eucl_dists[:,:,:max_n_atoms].contiguous()
    graph_dists = arr_lst_to_padded_batch(graph_dists, torch.long, 10)
    graph_dists = graph_dists[:,:,:max_n_atoms].contiguous()
    angles_in = arr_lst_to_padded_batch(angles_in)
    angles_out = arr_lst_to_padded_batch(angles_out)
    angles = torch.cat((angles_in, angles_out), dim=-1).contiguous()
    
    return (x, bond_x, sc_x, sc_m_x, eucl_dists, graph_dists,  angles, mask, 
            bond_idx, sc_idx, angles_idx, sc_types), sc_vals
