from epyt_flow.simulation import ScenarioSimulator, EpanetConstants
import numpy as np
import utils as functions
from wds_utils import InpSetup
import time as timer
import os
import shutil
import tempfile

def run_quality_simulation_spike(
        f_inp_in, duration, hydraulic_step, f_msx_in=None, quality_step=None, 
        pattern_step=None, sim_setup_fns=[], pattern=None, progress=True, injection_nodes=[],
        source_type=EpanetConstants.EN_CONCEN, inp_setup_fns=[], **kwargs
    ):
    if quality_step is None:
        quality_step = hydraulic_step
    if pattern_step is None:
        pattern_step = hydraulic_step * 6

    with InpSetup(f_inp_in, inp_setup_fns) as inp_file:
        
        with ScenarioSimulator(f_inp_in=inp_file, f_msx_in=f_msx_in) as sim:
            # Set general parameters
            sim.set_general_parameters(
                simulation_duration=duration,
                hydraulic_time_step=hydraulic_step,
                reporting_time_step=hydraulic_step,
                quality_time_step=quality_step,
                flow_units_id=EpanetConstants.EN_CMH,
                **kwargs
            )
            # Set initial concentration and simple (constant) reactions
            #for node_idx in sim.epanet_api.get_all_nodes_idx():
            #    sim.epanet_api.set_node_init_quality(node_idx, 0)
            #for link_idx in sim.epanet_api.get_all_links_idx():
            #    sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_BULKORDER, 0.)
            #    sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_WALLORDER, 0.)
            #sim.set_general_parameters(flow_units_id=EpanetConstants.CMH)
            # Add chlorine injection at the reservoir
            sim.enable_chemical_analysis()
            #sim.enable_chemical_analysis(chemical_name='CL2')
            # sim.enable_chemical_analysis(chemical_name='THM')
            sim.epanet_api.settimeparam(EpanetConstants.EN_PATTERNSTEP, pattern_step)

            if f_msx_in is not None:
                sim.epanet_api.set_msx_time_step(quality_step)

            for fn in sim_setup_fns:
                fn(sim)

            if pattern is None:
                pattern_length = duration // pattern_step
                pattern = np.eye(pattern_length)[2] + np.eye(pattern_length)[3]*1e-10
                pattern += np.eye(pattern_length)[13] + np.eye(pattern_length)[14]*1e-10
                pattern += np.eye(pattern_length)[23] + np.eye(pattern_length)[24]*1e-10

            for reservoir_id in injection_nodes: # sim.sensor_config.nodes:# sim.epanet_api.get_all_reservoirs_id() + sim.epanet_api.get_all_tanks_id():
                if f_msx_in is not None:
                    # if reservoir_id != 'Injection':
                    #     continue
                    sim.add_species_injection_source(
                        'CL2', 
                        node_id=reservoir_id,
                        pattern=pattern,
                        source_type=source_type#EpanetConstants.EN_CONCEN#EN_MASS
                    )
                else:
                    sim.add_quality_source(
                        node_id=reservoir_id,
                        pattern=pattern,
                        source_type=source_type#EpanetConstants.EN_CONCEN#EN_MASS
                    )

            # Place quality and flow sensor everywhere
            sim.set_flow_sensors(sensor_locations=sim.sensor_config.links)
            sim.set_node_quality_sensors(sensor_locations=sim.sensor_config.nodes)
            sim.set_pressure_sensors(sensor_locations=sim.sensor_config.nodes)

            if f_msx_in is not None:
                sim.set_bulk_species_node_sensors(sensor_info={
                        "CL2": sim.sensor_config.nodes,
                #        "THM": sim.sensor_config.nodes
                })
            #sim.set_link_quality_sensors(sensor_locations=sim.sensor_config.links)

            # Run simulation and store results
            res = sim.run_simulation(verbose=progress)
            topo = sim.get_topology()
            
    return topo, res, pattern

def inp_to_graph_data(
        inp_file, injection_pattern, injection_nodes, n_seconds, dt, quality_dt=None, 
        pattern_dt=None, f_msx_in='networks/ltown.msx', sim_setup_fns=[], 
        inp_setup_fns=[], out_file=None, **kwargs
    ):

    sim_kwargs = {
        'duration' : n_seconds,
        'hydraulic_step' : dt,
        'quality_step' : quality_dt,
        'pattern' : injection_pattern,
        'pattern_step' : pattern_dt,
        'f_msx_in' : f_msx_in,
        'sim_setup_fns' : sim_setup_fns,
        'inp_setup_fns' : inp_setup_fns,
        'injection_nodes' : injection_nodes
    }
    
    sim_kwargs.update(kwargs)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        
        filename = os.path.basename(inp_file)
        local_inp = os.path.join(tmpdir, filename)
        shutil.copy(inp_file, local_inp)
        
        if f_msx_in is not None:
            filename = os.path.basename(f_msx_in)
            local_msx = os.path.join(tmpdir, filename)
            shutil.copy(f_msx_in, local_msx)
            sim_kwargs['f_msx_in'] = local_msx
            
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        t0 = timer.time()
        topo, res, inj_pattern = run_quality_simulation_spike(local_inp, **sim_kwargs)
        runtime = timer.time() - t0
        
        os.chdir(old_cwd)
        

    if f_msx_in is not None:
        data = res.get_data_bulk_species_node_concentration() # [time, nodes]
    else:
        data = res.get_data_nodes_quality()
    edge_flows = res.get_data_flows() # [time, nodes]
    
    edge_index = functions.make_edge_index(topo)
    #tensor_data = torch.tensor(data[:].T).float()
    true_flows = functions.flow_to_velocity(topo, edge_flows).T
    edge_index_t = edge_index
    lengths = functions.get_edge_attribute(topo, 'length')
    diameter = functions.get_edge_attribute(topo, 'diameter')/10/100
    boundary_index = functions.get_node_index(topo, injection_nodes)
    
    if out_file is not None:
        res.save_to_file(out_file)
        
    return {
        'edge_index' : edge_index_t,
        'epanet_result' : data[:].T,
        'flow_field' : true_flows,
        'edge_lengths' : lengths,
        'edge_diameter' : diameter,
        'boundary_index' : boundary_index,
        'boundary_values' : data[:, boundary_index].T,
        'topology' : topo,
        'runtime': runtime,
        'res' : res
    }