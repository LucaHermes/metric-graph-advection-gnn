import torch
import numpy as np
import modules
from model.advection_model_mp import AdvectionModelMP
from model.advection_layer import AdvectionLayer
import networkx as nx
from itertools import count

model_inputs = None



def semi_lagrangian_mpnn_1d(
    initial_state, flow_field, dx, L, output_times, dt, control_inputs=None, 
    control_indices=None, interpolation='bilinear', progress=True, mixing_at_nodes=True, 
    iterate=False, adaptive_steps=False, device='cpu'
):
    N = int((L+dx) // dx) # int(L / dx)
    N = initial_state.shape[0]
    n_edges = N - 1
    nsteps = flow_field.shape[1]

    edge_index = torch.stack((
        torch.arange(0, n_edges), torch.arange(1, N)
    ))

    
    # edge_index = torch.stack((
    #     torch.cat([
    #         torch.arange(0, N-20), torch.tensor([N-40]), 
    #         torch.arange(N-20, N)
    #     ]), 
    #     torch.cat([
    #         torch.arange(1, N-20), torch.zeros(1, dtype=torch.long),
    #         torch.arange(N-20, N), torch.zeros(1, dtype=torch.long)
    #     ])
    # ))


    n_edges = edge_index.shape[1]
    flow_field = flow_field.repeat(2, 1)[:n_edges]
    edge_lengths = torch.ones(n_edges) * dx
    cT = 1 if initial_state.ndim == 1 else initial_state.shape[1]
    
    result, _, _, _, _ = semi_lagrangian_mpnn(
        initial_state, flow_field, edge_index, edge_lengths, output_times, dt, nsteps-cT,
        control_inputs=control_inputs, control_indices=control_indices, device=device,
        interpolation=interpolation, max_msg_passing_rounds=5000, progress=progress,
        mixing_at_nodes=mixing_at_nodes, iterate=iterate, adaptive_steps=adaptive_steps
    )
    return result.numpy()


def semi_lagrangian_mpnn(
    initial_state, flow_field, edge_index, edge_lengths, output_times, dt, nsteps, model=None,
    edge_capacities=1., control_inputs=None, control_indices=None, interpolation='bilinear',
    max_msg_passing_rounds=300, progress=True, mixing_at_nodes=True, iterate=False, adaptive_steps=False,
    device='cpu'
):
    global model_inputs
    model_inputs = prepare_model_inputs(
        initial_state, flow_field, edge_index, edge_lengths, output_times, dt, 
        nsteps, edge_capacities, control_inputs, control_indices, iterate, device
    )
    
    if model is None:
        # Create Solver object
        advection_op = modules.AdvectionModuleGridSampleDynamic(interpolation_mode=interpolation)

        layer = AdvectionLayer(advection_op, mixing_at_nodes)
        model = AdvectionModelMP(
            layer, max_msg_passing_rounds=max_msg_passing_rounds,
            progress=progress, adaptive_steps=adaptive_steps
        )
    
    model.to(device)

    pred, edge_passes, _, agg_time, aggs_all = model(**model_inputs)
    
    pred[:,:model_inputs['x'].shape[1]] = model_inputs['x']
    
    return pred.cpu(), edge_passes, _, agg_time, aggs_all


def prepare_model_inputs(
    initial_state, flow_field, edge_index, edge_lengths, output_times, dt, 
    nsteps, edge_capacities=1., control_inputs=None, control_indices=None, 
    iterate=False, device='cpu'
):
    _, n_edges = edge_index.shape
    flow_field_graph = flow_field[:n_edges]

    traversal_times, selfloop_mask, xs_map = modules.compute_backward_transit_times_fast(
        edge_lengths, flow_field_graph, dt
    )
    traversal_times = np.nan_to_num(traversal_times, nan=max(output_times)+1)
    flow_field_graph = torch.as_tensor(flow_field_graph, dtype=torch.get_default_dtype(), device=device)

    edge_capacities = torch.as_tensor(edge_capacities, dtype=torch.get_default_dtype(), device=device)
    initial_state = torch.as_tensor(initial_state, dtype=torch.get_default_dtype(), device=device)
    traversal_times = torch.as_tensor(traversal_times, dtype=torch.get_default_dtype(), device=device)
    xs_map = torch.as_tensor(xs_map, dtype=torch.get_default_dtype(), device=device)
    selfloop_mask = torch.as_tensor(selfloop_mask, dtype=torch.get_default_dtype(), device=device)
    edge_index = torch.as_tensor(edge_index, device=device)

    if control_indices is not None:
        boundary_index = torch.tensor(control_indices, device=device)
        boundary_input = torch.tensor(control_inputs, dtype=torch.get_default_dtype(), device=device)
        if boundary_input.ndim < 2:
            boundary_input = boundary_input.unsqueeze(0)
        if boundary_input.ndim < 3:
            boundary_input = boundary_input.unsqueeze(-1)
    else:
        boundary_index = boundary_input = None
    
    if edge_capacities.ndim < 2 and edge_capacities.ndim > 0:
        edge_capacities = edge_capacities.unsqueeze(1)
    if initial_state.ndim < 2:
        initial_state = initial_state.unsqueeze(-1)
    if initial_state.ndim < 3:
        initial_state = initial_state.unsqueeze(-1)

    delay_steps = traversal_times.clamp(max=max(output_times))
    delay_steps = delay_steps / dt
    
    if iterate:
        n_steps = nsteps
        nsteps = 1
    else:
        n_steps = 1
    
    return {
        'x' : initial_state,
        'edge_index' : edge_index,
        'flows' : flow_field_graph * edge_capacities,
        'delay_steps' : delay_steps,
        'xs_map' : xs_map,
        'sl_mask' : selfloop_mask,
        'Tau' : nsteps,
        'n_steps' : n_steps,
        'boundary_values' : boundary_input,
        'boundary_index' : boundary_index,
    }