from torch_geometric.nn import MessagePassing
from torch import nn
import torch
from torchdiffeq import odeint, odeint_adjoint
from functools import partial
import torch.nn.functional as F


class PDEFunction(MessagePassing):

    def __init__(self, in_features, out_features, config, device):
        super().__init__(aggr='sum')
        self.linear1 = torch.nn.Linear(config['n_edge_features'] + config['hidden_dim'], config['hidden_dim'], bias=False)
        self.linear2 = torch.nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False)
        self.linear3 = torch.nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False)
        self.alpha_train = nn.Parameter(torch.tensor(1.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.config = config
        self.device = device

    def init(self, x, edge_index, edge_weight=None, boundary_condition=None, boundary_index=None):
        num_nodes = x.shape[0]
        self.boundary_condition = boundary_condition
        self.boundary_index = boundary_index
        ode_edge_index = edge_index
        return x, ode_edge_index, {
            'x0' : x,
        }
    
    def message(self, x_j, x_i, edge_features=None):
        return self.linear2(torch.nn.functional.relu(self.linear1(torch.cat((x_j - x_i, edge_features), dim=-1))))
   
    # the t param is needed by the ODE solver.
    def forward(self, t, x, edge_index, edge_features=None, x0=None):
        x, edge_features = self.boundary_condition(
            t, x, edge_features
        )
        
        ax = self.propagate(
            edge_index, x=x, edge_features=edge_features, size=None
        )

        if self.config.get('alpha_sigmoid'):
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train


        f = (alpha * (ax - x) + self.linear3(x)).clone()
        f[self.boundary_index] = 0.
        
        return f
    
class PDEModel(torch.nn.Module):
    '''
    This class implements everything thats required to integrate an ODE.
    These are mainly the settings for the odeint integrator.
    '''

    def __init__(self, ode_function, config):
        super().__init__()
        self.ode_function = ode_function
        self.method = config['method']
        self.step_size = config['step_size']
        self.atol = config.get('atol', 1e-9)
        self.rtol = config.get('rtol', 1e-7)
        self.adjoint = config.get('adjoint')
        self.adjoint_method = config.get('adjoint_method')
        self.integrator = odeint_adjoint if self.adjoint else odeint
        self.device = self.ode_function.device # config['device']
        self.config = config
        #self.set_eval_times(config['time'])
        self.grand_setup_ode(config)
        # Input transformation
        self.m1 = torch.nn.Linear(config['input_dim'], config['hidden_dim'], bias=False)
        # Output transformation
        self.m2 = torch.nn.Linear(config['hidden_dim'], config['output_dim'], bias=False)
        
    @property
    def ode_opt(self):
        opt = {
            'method' : self.method,
            'options' : { 'step_size': self.step_size },
            'atol' : self.atol,
            'rtol' : self.rtol
        }
        if self.adjoint:
            opt.update({
                'adjoint_method' : self.config['adjoint_method'],
                'adjoint_options' : { 'step_size': self.config['adjoint_step_size'] },
                'adjoint_atol' : self.atol_adjoint,
                'adjoint_rtol' : self.rtol_adjoint
            })
        return opt
    
    def grand_setup_ode(self, config):
        self.atol = self.config['tol_scale'] * 1e-7
        self.rtol = self.config['tol_scale'] * 1e-9
        if self.config.get('adjoint'):
            self.atol_adjoint = self.config['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.config['tol_scale_adjoint'] * 1e-9

    def integrate(self, x, edge_index, **kwargs):
        x0, edge_index, int_kwargs = self.ode_function.init(x, edge_index, **kwargs)
        
        self.ode_function.forward = partial(
            self.ode_function.forward, edge_index=edge_index, **int_kwargs
        )
        
        state_dt = self.integrator(
            self.ode_function, x0, self.eval_times,
            **self.ode_opt
        )
        return state_dt
    
    def forward(self, x, edge_index, edge_weight=None, batch_index=None, boundary_condition=None, boundary_index=None):
        # Only use the first time step as the initial value
        # (Make sure everything except sensor nodes are masked).
        x = x[:,:1]
        x = self.m1(x)
        
        state_dt = self.integrate(
            x, edge_index, edge_weight=edge_weight, boundary_condition=boundary_condition, boundary_index=boundary_index
        )
        
        z = state_dt.transpose(0, 1)
        
        z = F.relu(z)
        
        if self.config.get('fc_out'):
            z = self.fc(z)
            z = F.relu(z)
            
        z = F.dropout(z, self.config['dropout'], training=self.training)
            
        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z.squeeze(-1)

    def set_eval_times(self, time):
        time = torch.tensor(time + 1, dtype=torch.float32, device=self.device)
        self.eval_times = torch.arange(0, time, dtype=torch.float32, device=self.device)


class PDEModelInputWrapper(PDEModel):
    
    def __init__(self, ode_function, config):
        super().__init__(ode_function, config)
    
    def __call__(self, cT, Tau, **model_inputs):
        model_inputs['x'] = model_inputs['x']#.unsqueeze(-1)
        model_inputs['boundary_values'] = model_inputs['boundary_values']#.unsqueeze(-1)
        self.set_eval_times(cT + Tau)
        
        et_shape = model_inputs['flows'].shape
        edge_capacity = model_inputs['edge_capacity_scaled'].view(-1, 1)
        #inv_capacity = min_edge_capacity / edge_capacity.expand(et_shape)
        
        edge_features = torch.cat((
            #inv_capacity.unsqueeze(-1),
            edge_capacity.expand(et_shape).unsqueeze(-1),
            model_inputs['delay_steps_scaled'].unsqueeze(-1) * 2, 
            model_inputs['flows_scaled'].unsqueeze(-1),
            #model_inputs['edge_lengths'].view(-1, 1).expand(et_shape).unsqueeze(-1),
            #model_inputs['flow_field'].unsqueeze(-1)
        ), dim=-1)
        edge_index = torch.cat((model_inputs['edge_index'], model_inputs['edge_index'].flip(0)), dim=-1)
        edge_features = torch.cat((edge_features, edge_features * torch.tensor([1., -1., -1.], device=edge_features.device)), dim=0)
        
        boundary_condition = partial(
            self.dirchlet_bc_and_flow,
            boundary_values=model_inputs['boundary_values'][...,:Tau + 1],
            boundary_index=model_inputs['boundary_index'],
            edge_values=edge_features[:,:Tau + 1],
            T=Tau
        )
        
        self_inputs = {
            'x' : model_inputs['x'].squeeze(1),
            'edge_index' : edge_index,
            'edge_weight' : edge_features,
            'boundary_condition' : boundary_condition,
            'boundary_index' : model_inputs['boundary_index'],
        }
        
        pred = super().__call__(**self_inputs)
        
        return pred[:,:-1], 
        
    def dirchlet_bc_and_flow(self, time, x, edge_features, boundary_values, edge_values, boundary_index, T):
        # inplace manipulation of the current solution
        t = time
        t_lo = torch.floor(t).int().clip(None, T)
        t_hi = torch.ceil(t).int().clip(None, T)
        boundary = boundary_values[:, t_lo] + (time % 1) * (boundary_values[:, t_hi] - boundary_values[:, t_lo])
        edge_features = edge_values[:, t_lo] + (time % 1) * (edge_values[:, t_hi] - edge_values[:, t_lo])
        x = x.clone()
        x[boundary_index] = self.m1(boundary)
        return x, edge_features