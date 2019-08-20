import torch.nn as nn
from layernorm import LayerNorm

def hidden_layer(n_in, n_out, batch_norm, dropout, layer_norm=False, act=None):
    layers = []
    layers.append(nn.Linear(n_in, n_out))
    if act: layers.append(act)
    if batch_norm: layers.append(nn.BatchNorm1d(n_out))
    if layer_norm: layers.append(LayerNorm(n_out))
    if dropout != 0: layers.append(nn.Dropout(dropout))
    return layers

class FullyConnectedNet(nn.Module):
    def __init__(self, n_input, n_output=None, layers=[], act=nn.ReLU(True), 
                 dropout=[], batch_norm=False, out_act=None, final_bn=False, 
                 layer_norm=False, final_ln=False):
        super().__init__()
        sizes = [n_input] + layers
        if n_output: 
            sizes += [n_output]
            dropout += [0.0]
        layers_ = []
        for i, (n_in, n_out, dr) in enumerate(zip(sizes[:-1], sizes[1:], 
                                                  dropout)):
            act_ = act if i < len(layers) else out_act
            batch_norm_ = batch_norm if i < len(layers) else final_bn
            layer_norm_ = layer_norm if i < len(layers) else final_ln
            layers_ += hidden_layer(
                n_in, n_out, batch_norm_, dr, layer_norm_, act_)      
        self.layers = nn.Sequential(*layers_)
        
    def forward(self, x):
        return self.layers(x)