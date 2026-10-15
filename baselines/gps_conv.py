from torch_geometric.nn import NNConv, GPSConv, ResGatedGraphConv, GINEConv
from torch import nn
import torch
import torch.nn.functional as F

def create_mpnn(mpnn, hidden_channels, **kwargs):
    if mpnn == 'nnconv':
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels**2),
        )
        return NNConv(hidden_channels, hidden_channels, nn=mlp, aggr='sum', **kwargs)
    elif mpnn == 'gine':
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        return GINEConv(mlp, **kwargs)
    elif mpnn == 'gated_gnn':
        return ResGatedGraphConv(
            hidden_channels, hidden_channels, edge_dim=hidden_channels, **kwargs
        )
    else:
        raise NotImplementedError()

class GPSConvWrapper(nn.Module):

    def __init__(self, cT, Tau, hidden_dim, num_layers, pe_dim=8, conv='gated_gnn',**kwargs):
        super().__init__()
        n_steps = cT + Tau

        self.node_encoder = nn.Linear(n_steps, hidden_dim - pe_dim)
        self.edge_encoder = nn.Linear(n_steps * 3, hidden_dim)
        self.pe_lin = nn.Linear(30, pe_dim)
        self.pe_norm = nn.BatchNorm1d(30)

        self.mpnn_layers = nn.ModuleList([])
        attn_kwargs = {
            'dropout' : kwargs.pop('attn_dropout', 0.0)
        }

        for i in range(num_layers):
            mpnn = create_mpnn(conv, hidden_dim)
            self.mpnn_layers.append(
                GPSConv(hidden_dim, mpnn, attn_kwargs=attn_kwargs, **kwargs)
            )
        
        self.node_decoder = nn.Linear(hidden_dim, n_steps)

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
        
        x_pe = self.pe_norm(model_inputs['pe'])
        x = torch.cat((x, self.pe_lin(x_pe)), dim=1)

        edge_index = inputs['edge_index']

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr=e)

        pred = self.node_decoder(x)
        
        return pred,