import torch.nn as nn

class LayerNorm(nn.LayerNorm):
    """Class overriding pytorch default layernorm intitialization"""
    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)