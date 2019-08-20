import torch

def scatter_add(src, idx, num=None, dim=0, out=None):
    if not num: num = idx.max().item() + 1
    sz, expanded_idx_sz = src.size(), src.size()
    sz = sz[:dim] + torch.Size((num,)) + sz[(dim+1):]
    expanded_idx = idx.unsqueeze(-1).expand(expanded_idx_sz)
    if out is None: out = torch.zeros(sz, dtype=src.dtype, device=src.device)
    return out.scatter_add(dim, expanded_idx, src)

def scatter_mean(src, idx, num=None, dim=0, out=None):
    return (scatter_add(src, idx, num, dim, out) 
            / scatter_add(torch.ones_like(src), idx, num, dim).clamp(1.0))