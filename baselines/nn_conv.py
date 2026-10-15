from torch_geometric.nn import MessagePassing, NNConv
from torch import nn
import torch
from torchdiffeq import odeint, odeint_adjoint
from functools import partial
import torch.nn.functional as F

class NNConvWrapper(nn.Module):

    def __init__(self, cT, Tau, hidden_channels, num_layers, aggr="sum", **kwargs):
        super().__init__()
        n_steps = cT + Tau

        self.node_encoder = nn.Linear(n_steps, hidden_channels)
        self.edge_encoder = nn.Linear(n_steps * 3, hidden_channels)

        self.mpnn_layers = nn.ModuleList([])
        
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels**2),
            )
            self.mpnn_layers.append(
                NNConv(hidden_channels, hidden_channels, nn=mlp, aggr=aggr)
            )
        
        self.node_decoder = nn.Linear(hidden_channels, n_steps)

    def prepare_inputs(self, cT, Tau, **model_inputs):
        n_steps = cT + Tau
        
        node_features = F.pad(model_inputs['x'].squeeze(-1), (0, Tau))
        node_features[model_inputs['boundary_index'].squeeze(-1)] = model_inputs['boundary_values'].squeeze(-1)[:,:n_steps]

        edge_capacity = model_inputs['edge_capacity_scaled'].view(-1, 1)
        et_shape = model_inputs['flows'].shape

        edge_index = torch.cat((model_inputs['edge_index'], model_inputs['edge_index'].flip(0)), dim=-1)
        edge_attr = torch.stack((
            edge_capacity.expand(et_shape).unsqueeze(-1),
            model_inputs['delay_steps_scaled'].unsqueeze(-1) * 2, 
            model_inputs['flows_scaled'].unsqueeze(-1),
        ), dim=-1)
        
        edge_attr = torch.cat((edge_attr, edge_attr * torch.tensor([1., -1., -1.], device=edge_attr.device)), dim=0)
        edge_attr = edge_attr[:,:n_steps].reshape(et_shape[0] * 2, -1)
        
        return {
            'x' : node_features,
            'edge_index' : edge_index,
            'edge_attr' : edge_attr,
        }   

    def forward(self, cT, Tau, **model_inputs):
        inputs = self.prepare_inputs(cT, Tau, **model_inputs)

        x = self.node_encoder(inputs['x'])
        e = self.edge_encoder(inputs['edge_attr'])

        edge_index = inputs['edge_index']

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr=e)

        pred = self.node_decoder(x)
        
        return pred,