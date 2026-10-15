import wntr
import numpy as np
import simulate
import tempfile
import os
from contextlib import contextmanager
import uuid
import warnings

@contextmanager
def working_directory(path):
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def epanet_advection_1d(
    initial_state, flow_field, dx, L, output_times, dt, control_inputs=None, 
    control_indices=None, interpolation='bilinear', progress=True
):
    if control_inputs is not None:
        assert flow_field.shape[1] == control_inputs.shape[1]
    # Build the inp file of a 1-d advection system
    N = int(L / dx) + 1
    n_edges = N - 1
    nsteps = flow_field.shape[1]
    
    # For EPANET dt must be at least 1
    if dt % 1 != 0 or dt < 1:
        dt_new = max(np.round(dt), 1)
        print(f'Time step is {dt} which is invalid, mapping to nearest valid step size {dt_new}.')
        dt = dt_new

    _flow_field = flow_field[:n_edges]

    # spatial axis is axis 0
    inflows = np.maximum(_flow_field[:-1], 0) + np.maximum(-_flow_field[1:], 0)
    outflows = np.maximum(-_flow_field[:-1], 0) + np.maximum(_flow_field[1:], 0)

    inflows = np.concatenate((np.maximum(-_flow_field[:1], 0), inflows, np.maximum(_flow_field[-1:], 0)), axis=0) #np.pad(inflows, ((0,1),(0,0)))
    outflows = np.concatenate((np.maximum(_flow_field[:1], 0), outflows, np.maximum(-_flow_field[-1:], 0)), axis=0) #np.pad(outflows, ((1,0),(0,0)))
    demands = (inflows - outflows)

    # This diameter ensures that crosssectional area is 1.0, thus flowrate = velocity
    diameter = 2. / np.sqrt(np.pi)

    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.demand_model = 'DD'

    wn.add_reservoir('-1', base_head=100.)

    for n in range(N):
        wn.add_pattern(f'pattern_{n}', demands[n].tolist())
        wn.add_junction(str(n), base_demand=1., demand_pattern=f'pattern_{n}')
        wn.add_pipe(f'p{n-1}', f'{n-1}', f'{n}', length=dx, diameter=diameter, roughness=10000.)

    wn.options.time.hydraulic_timestep = int(dt)
    wn.options.time.pattern_timestep = int(dt)
    wn.options.time.report_timestep = int(dt)
    wn.options.time.duration = int(nsteps * dt)
    #wn.options.quality.parameter = 'CHEMICAL'

    #for source_idx, source_pattern in zip(control_indices, control_inputs):
    #    wn.add_pattern(f'Injection{source_idx}', source_pattern.tolist())
    #    wn.add_source(f'S{source_idx}', f'{source_idx}', 'MASS', 1000., f'Injection{source_idx}')

    assert control_inputs.shape[0] == 1, 'Currently only supports 1.'

    sim_kwargs = {
        'duration' : int(nsteps * dt),
        'hydraulic_step' : int(dt),
        'quality_step' : int(dt),
        'pattern' : control_inputs[0],
        'pattern_step' : int(dt),
        'f_msx_in' : os.path.abspath('networks/ltown.msx'),
        'progress' : progress,
        'sim_setup_fns' : []
    }
    with tempfile.TemporaryDirectory() as tempdir:
        with working_directory(tempdir):
            with tempfile.NamedTemporaryFile(suffix='.inp') as tmp_file:
                wntr.network.io.write_inpfile(wn, tmp_file.name, 'CMH')
                topo, res, inj_pattern = simulate.run_quality_simulation_spike(tmp_file.name, **sim_kwargs)

    data = res.get_data_bulk_species_node_concentration() # [time, nodes]
    edge_flows = res.get_data_flows() # [time, nodes]

    # edge flows are in m³/h, convert to m³/s
    edge_flows = edge_flows.T / 60 / 60
    # Make sure the hydraulics reflect the expected flow field
    assert np.isclose(flow_field[1:], edge_flows[1:,:nsteps], atol=1e-5).all(), \
        'The flows from epanet are not similar to the provided flow field.'
    
    return data.T[:-1, 1:nsteps+1] #, edge_flows

