import torch
from torch_geometric.nn.conv import TransformerConv, GCNConv
from torch_geometric.nn import BatchNorm, LayerNorm
from ..torch_utils import activation_functions

def GNN(input_dim, hidden_layer_size, n_layers, encoder_act=None, output_dim=None, device="cuda", norm:str=""):
    # Encoder layers
    encoder_layer = []

    #calculate number of layer
    n_layers = n_layers-1 if output_dim is None else n_layers-2
    # input size
    inputs_size = [input_dim]+[hidden_layer_size]*(n_layers)
    #create layers
    for in_size in inputs_size:
        encoder_layer.append(GCNConv(in_size,
                                    hidden_layer_size,
                                    ))
        if isinstance(encoder_act, str):
            encoder_layer.append(activation_functions(encoder_act))
        encoder_layer.append(get_norm(hidden_layer_size, norm))

    if output_dim is not None:
        encoder_layer.append(GCNConv(hidden_layer_size,
                                    output_dim,
                                    ))
    # modulelist
    encoder_layer = torch.nn.ModuleList(encoder_layer)

    return encoder_layer.to(device)

def get_norm(hidden_layer_size, norm):
    "get pytorch geometric norm"
    if norm == "layer":
        return LayerNorm(hidden_layer_size)
    elif norm == "batch":
        return BatchNorm(hidden_layer_size)
    else:
        return torch.nn.Identity(hidden_layer_size)