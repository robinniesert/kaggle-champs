# 

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fcnet import FullyConnectedNet, hidden_layer
from scatter import scatter_mean
from utils import clones
from layernorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
   
def _gather_nodes(x, idx, sz_last_dim):
    idx = idx.unsqueeze(-1).expand(-1, -1, sz_last_dim)
    return x.gather(1, idx)

class ENNMessage(nn.Module):
    PAD_VAL = -999
    
    def __init__(self, d_model, d_edge, kernel_sz, enn_args={}, ann_args=None):
        super().__init__()
        assert kernel_sz <= d_model
        self.d_model, self.kernel_sz = d_model, kernel_sz
        self.enn = FullyConnectedNet(d_edge, d_model*kernel_sz, **enn_args)
        if ann_args: self.ann = FullyConnectedNet(1, d_model, **ann_args)
        else: self.ann = None
    
    def forward(self, x, edges, pairs_idx, angles=None, 
                angles_idx=None, t=0): 
        if t==0: 
            self.set_a_mat(edges)
            if self.ann: self.set_att(angles)
            self.pairs_idx = torch.cat([pairs_idx, pairs_idx[:,:,[1,0]]], dim=1)
        
        return self.add_message(torch.zeros_like(x), x, angles_idx)
    
    def set_a_mat(self, edges):
        n_edges = edges.size(1)
        a_vect = self.enn(edges) 
        a_vect = a_vect / (self.kernel_sz ** .5) # rescale
        mask = edges[:,:,0,None].expand(a_vect.size())==self.PAD_VAL
        a_vect = a_vect.masked_fill(mask, 0.0)
        self.a_mat = a_vect.view(-1, n_edges, self.d_model, self.kernel_sz)
        self.a_mat = torch.cat([self.a_mat, self.a_mat], dim=1)
    
    def set_att(self, angles):
        angles = angles.unsqueeze(-1)
        self.att = self.ann(angles)
        mask = angles.expand(self.att.size())==self.PAD_VAL
        self.att = self.att.masked_fill(mask, 0.0)
    
    def add_message(self, m, x, angles_idx=None):
        # select the 'x_{j}' feeding into the 'm_{i}'
        x_in = _gather_nodes(x, self.pairs_idx[:,:,1], self.d_model)
        
        # do the matrix multiplication 'A_{ij}x_{j}'
        if self.kernel_sz==self.d_model:
            ax = (x_in.unsqueeze(-2) @ self.a_mat).squeeze(-2)
        else:
            x_padded = F.pad(x_in, self.n_pad)
            x_unfolded = x_padded.unfold(-1, self.kernel_sz, 1)
            ax = (x_unfolded * self.a_mat).sum(-1)
        
        # apply atttention
        if self.ann:
            n_pairs = self.pairs_idx.size(1)
            ave_att = scatter_mean(self.att, angles_idx, num=n_pairs, dim=1, 
                                   out=torch.ones_like(ax))
            ax = ave_att * ax
        
        # Sum up all 'A_{ij}h_{j}' per node 'i'
        idx_0 = self.pairs_idx[:,:,0,None].expand(-1, -1, self.d_model)
        return m.scatter_add(1, idx_0, ax)
    
    @property
    def n_pad(self):
        k = self.kernel_sz
        return (k // 2, k // 2 - int(k % 2 == 0))

class MultiHeadedDistAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        super().__init__()
        self.d_model, self.d_k, self.h = d_model, d_model // h, h
        self.attn = None
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
        
    def forward(self, dists, x, mask):
        batch_size = x.size(0)
        x = self.linears[0](x).view(batch_size, -1, self.h, self.d_k)
        x, self.attn = self.apply_attn(dists, x, mask)
        x = x.view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    def apply_attn(self, dists, x, mask):
        attn = self.create_raw_attn(dists, mask)
        attn = attn.transpose(-2,-1).transpose(1, 2)
        if self.dropout: attn = self.dropout(attn)
        
        x = x.transpose(1, 2)
        x = torch.matmul(attn, x)
        x = x.transpose(1, 2).contiguous()
        return x, attn
    
    def create_raw_attn(self, dists, mask):
        pass

class MultiHeadedGraphDistAttention(MultiHeadedDistAttention):
    def __init__(self, h, d_model, dropout=0.0):
        super().__init__(h, d_model, dropout)
        self.embedding = nn.Embedding(11, h)
    
    def create_raw_attn(self, dists, mask):
        emb_dists = self.embedding(dists)
        mask = mask.unsqueeze(-1).expand(emb_dists.size())
        emb_dists = emb_dists.masked_fill(mask==0, -1e9)
        return F.softmax(emb_dists, dim=-2).masked_fill(mask==0, 0)

class MultiHeadedEuclDistAttention(MultiHeadedDistAttention):
    def __init__(self, h, d_model, dropout=0.0):
        super().__init__(h, d_model, dropout)
        self.log_prec = nn.Parameter(torch.Tensor(1, 1, 1, h))
        self.locs = nn.Parameter(torch.Tensor(1, 1, 1, h))
        nn.init.normal_(self.log_prec, mean=0.0, std=0.1)
        nn.init.normal_(self.locs, mean=0.0, std=1.0)
    
    def create_raw_attn(self, dists, mask):
        dists = dists.unsqueeze(-1).expand(-1, -1, -1, self.h)
        z = torch.exp(self.log_prec) * (dists - self.locs)
        pdf = torch.exp(-0.5 * z ** 2)
        return pdf / pdf.sum(dim=-2, keepdim=True).clamp(1e-9)      

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None: scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1).masked_fill(mask==0, 0)
    if dropout is not None: p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super().__init__()
        # We assume d_v always equals d_k
        self.d_model, self.d_k, self.h = d_model, d_model // h, h
        self.attn = None
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
        
    def forward(self, query, key, value, mask):
        # Same mask applied to all h heads.
        mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask, self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        return self.linears[-1](x)

class AttendingLayer(nn.Module):
    def __init__(self, size, eucl_dist_attn, graph_dist_attn, 
                 self_attn, ff, dropout):
        super().__init__()
        self.eucl_dist_attn = eucl_dist_attn
        self.graph_dist_attn = graph_dist_attn
        self.self_attn = self_attn
        self.ff = ff
        self.subconns = clones(SublayerConnection(size, dropout), 4)
        self.size = size

    def forward(self, x, eucl_dists, graph_dists, mask):
        eucl_dist_sub = lambda x: self.eucl_dist_attn(eucl_dists, x, mask)
        x = self.subconns[0](x, eucl_dist_sub)
        graph_dist_sub = lambda x: self.graph_dist_attn(graph_dists, x, mask)
        x = self.subconns[1](x, graph_dist_sub)
        self_sub = lambda x: self.self_attn(x, x, x, mask)
        x = self.subconns[2](x, self_sub)
        return self.subconns[3](x, self.ff)

class MessagePassingLayer(nn.Module):
    def __init__(self, size, bond_mess, sc_mess, dropout, N):
        super().__init__()
        self.bond_mess = bond_mess
        self.sc_mess = sc_mess
        self.linears = clones(nn.Linear(size, size), 2*N)
        self.subconns = clones(SublayerConnection(size, dropout), 2*N)

    def forward(self, x, bond_x, sc_pair_x, angles, mask, bond_idx, 
                sc_idx, angles_idx, t=0):
        bond_sub = lambda x: self.linears[2*t](
            self.bond_mess(x, bond_x, bond_idx, angles, angles_idx, t))
        x = self.subconns[2*t](x, bond_sub)
        sc_sub = lambda x: self.linears[(2*t)+1](
            self.sc_mess(x, sc_pair_x, sc_idx, t=t))
        return self.subconns[(2*t)+1](x, sc_sub)


class Encoder(nn.Module):
    """
    Encoder is a stack of N attention layers and one message passing 
    layer.
    """
    def __init__(self, mess_pass_layer, attn_layer, N):
        super().__init__()
        self.mess_pass_layer = mess_pass_layer
        self.attn_layers = clones(attn_layer, N)
        self.norm = LayerNorm(attn_layer.size)
        
    def forward(self, x, bond_x, sc_pair_x, eucl_dists, graph_dists, angles, 
                mask, bond_idx, sc_idx, angles_idx):
        """Pass the inputs (and mask) through each layer in turn. At each layer
        the same message passing layer is used."""
        for t, attn_layer in enumerate(self.attn_layers):
            x = self.mess_pass_layer(
                x, bond_x, sc_pair_x, angles, mask, bond_idx, sc_idx, 
                angles_idx, t)
            x = attn_layer(x, eucl_dists, graph_dists, mask)
        return self.norm(x)


def create_contrib_head(d_in, d_ff, act, dropout=0.0, layer_norm=True):
    layers = hidden_layer(d_in, d_ff, False, dropout, layer_norm, act)
    layers += hidden_layer(d_ff, 1, False, 0.0) # output layer
    return nn.Sequential(*layers)

class ContribsNet(nn.Module):
    N_CONTRIBS = 5
    CONTIB_SCALES = [1, 250, 45, 35, 500]
    
    def __init__(self, d_in, d_ff, vec_in, act, dropout=0.0, 
                 layer_norm=True):
        super().__init__()
        contrib_head = create_contrib_head(d_in, d_ff, act, dropout, layer_norm) 
        self.blocks = clones(contrib_head, self.N_CONTRIBS)
        
    def forward(self, x):
        ys = torch.cat(
            [b(x)/s for b,s in zip(self.blocks, self.CONTIB_SCALES)], dim=-1)
        return torch.cat([ys[:,:-1], ys.sum(dim=-1, keepdim=True)], dim=-1)
    
class MyCustomHead(nn.Module):
    PAD_VAL = -999
    N_TYPES = 8
    
    def __init__(self, d_input, d_ff, d_ff_contribs, pre_layers=[], 
                 post_layers=[], act=nn.ReLU(True), 
                 dropout=3*[0.0], norm=False):
        super().__init__()
        fc_pre = hidden_layer(d_input, d_ff, False, dropout[0], norm, act)
        self.preproc = nn.Sequential(*fc_pre)
        fc_type = hidden_layer(d_ff, d_input, False, dropout[1], norm, act)
        self.types_net = clones(nn.Sequential(*fc_type), self.N_TYPES)
        self.contribs_net = ContribsNet(
            d_input, d_ff_contribs, d_ff, act, dropout[2], layer_norm=norm)
        
    def forward(self, x, sc_types):
        # reshape inputs for easier processing
        x, sc_types = x.view(-1, x.size(-1)), sc_types.view(-1)
        mask =  sc_types != self.PAD_VAL
        x, sc_types = x[mask], sc_types[mask]
        
        x_ = self.preproc(x)
        x_types = torch.zeros_like(x)
        for i in range(self.N_TYPES):
            t_idx = sc_types==i
            if torch.any(t_idx): x_types[t_idx] = self.types_net[i](x_[t_idx])
            else: x_types = x_types + 0.0 * self.types_net[i](x_) # fake call (only necessary for distributed training - to make sure all proceesses have gradients for all parameters)
        x = x + x_types 
        return self.contribs_net(x)

class Transformer(nn.Module):
    def __init__(self, d_atom, d_bond, d_sc_pair, d_sc_mol, N=6, 
                 d_model=512, d_ff=2048, d_ff_contrib=128, h=8, 
                 dropout=0.1, kernel_sz=128, enn_args={}, ann_args={}):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        c = copy.deepcopy
        bond_mess = ENNMessage(d_model, d_bond, kernel_sz, enn_args, ann_args)
        sc_mess = ENNMessage(d_model, d_sc_pair, kernel_sz, enn_args)
        eucl_dist_attn = MultiHeadedEuclDistAttention(h, d_model)
        graph_dist_attn = MultiHeadedGraphDistAttention(h, d_model)
        self_attn = MultiHeadedSelfAttention(h, d_model, dropout)
        ff = FullyConnectedNet(d_model, d_model, [d_ff], dropout=[dropout])
        
        message_passing_layer = MessagePassingLayer(
            d_model, bond_mess, sc_mess, dropout, N)
        attending_layer = AttendingLayer(
            d_model, c(eucl_dist_attn), c(graph_dist_attn), c(self_attn), c(ff), 
            dropout
        )
        
        self.projection = nn.Linear(d_atom, d_model)
        self.encoder = Encoder(message_passing_layer, attending_layer, N)
        self.write_head = MyCustomHead(
            2 * d_model + d_sc_mol, d_ff, d_ff_contrib, norm=True)
        
    def forward(self, atom_x, bond_x, sc_pair_x, sc_mol_x, eucl_dists, 
                graph_dists, angles, mask, bond_idx, sc_idx, 
                angles_idx, sc_types):
        x = self.encoder(
            self.projection(atom_x), bond_x, sc_pair_x, eucl_dists, graph_dists, 
            angles, mask, bond_idx, sc_idx, angles_idx
        )
        x = torch.cat(
            [_gather_nodes(x, sc_idx[:,:,0], self.d_model), 
             _gather_nodes(x, sc_idx[:,:,1], self.d_model), 
             sc_mol_x], 
            dim=-1
        )
        return self.write_head(x, sc_types)
