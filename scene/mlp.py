
import torch
from torch import nn
from torch.func import vmap, functional_call, stack_module_state

class MLP(nn.Module):
    def __init__(self, layers_size_list=None):
        super().__init__()
        self.layers = self.get_mlp_layers_sizes(layers_size_list)

    @staticmethod
    def get_mlp_layers_sizes(layers_size_list):
        layers = nn.Sequential()
        for i in range(len(layers_size_list)-2):
            layers.append(nn.Linear(layers_size_list[i], layers_size_list[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layers_size_list[-2], layers_size_list[-1]))
        return layers

    def forward(self, x):
        x = self.layers(x)
        return x

USE_VMAP = False

def vmap_mlp(params, x, meta=None):

    if USE_VMAP:
        def wrapper(params, data):
            return functional_call(meta, params, data)
        x = vmap(wrapper)(params, x)
    else:
        # This manual implementation is a little little bit faster that vmap version
        i = 0
        while True:
            if not f'layers.{i}.weight' in params:
                break
            x = torch.einsum('kdc,kc->kd', params[f'layers.{i}.weight'], x) + params[f'layers.{i}.bias']
            if f'layers.{i+2}.weight' in params:
                x = torch.relu(x)
            i = i+2

    return x
