import torch.nn as nn

# The below layernorm class initializes parameters according to the default 
# initialization of bacthnorm layers in pytorch v1.1 and below. Somehow this 
# initialization seemed to work significantly beter.
class LayerNorm(nn.LayerNorm):
    """Class overriding pytorch default layernorm intitialization."""
    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)