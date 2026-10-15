import utils as functions
import numpy as np
import simulate
import torch
import torch_geometric
import os
import pickle
import modules
import wds_utils
import time as timer

# LTOWN_EDGES = 908
# LTOWN_RESERVOIRS = 2
HANOI_EDGES = 34
HANOI_RESERVOIRS = 1

def create_random_demand_profiles(avg_demand_min, avg_demand_max, n_nodes, n_times):
    '''
    Make demand pattern for a WDS with average demand across nodes (ADN)
    ADN >= avg_demand_min
    ADN <= avg_demand_max
    Within the number of timesteps (n_times), both of these bounds will be 
    reached at least one time.
    '''
    z = np.random.normal(0, 30., size=(int(np.ceil(n_nodes/10)), 1)).repeat(10, axis=0)[:n_nodes]
    dp = np.abs((np.random.normal(size=(1, 100 + n_times)) + z).cumsum(1))[:,100:]
    #dp = dp / dp.sum(0) * avg_demand_max
    dp = (dp - dp.min()) / (dp.max() - dp.min())
    dp = np.abs(dp )#.clip(0)
    dp = dp * (avg_demand_max - avg_demand_min) + avg_demand_min
    return dp


def create_random_demand_profiles_clustered(avg_demand_min, avg_demand_max, n_nodes, n_times, cluster_ids, cov):
    '''
    Make demand pattern for a WDS with average demand across nodes (ADN)
    ADN >= avg_demand_min
    ADN <= avg_demand_max
    Within the number of timesteps (n_times), both of these bounds will be 
    reached at least one time.
    '''
    n_clusters = len(set(cluster_ids))
    z = np.random.normal(0, 0.1, size=(n_nodes, 1))
    # ts = np.random.normal(size=(n_clusters, 100 + n_times)).cumsum(1)[:,100:]
    ts = np.random.multivariate_normal(np.zeros(n_times), cov, size=n_clusters).clip(-3, 3)
    dp = z + ts[cluster_ids]
    #dp = dp / dp.sum(0) * avg_demand_max
    dp = (dp - dp.min()) / 5 # (dp.max() - dp.min())
    #dp = np.abs(dp )#.clip(0)
    dp = dp * (avg_demand_max - avg_demand_min) + avg_demand_min
    return dp


def generate_injection_pattern(num_points=1000, min_modes=3, max_modes=30, seed=None):
    rs = np.random.RandomState(seed)
    
    t = np.linspace(0, 2 * np.pi, num_points)
    signal = np.zeros(num_points)
    num_modes = rs.randint(min_modes, max_modes)
    decay_power = rs.uniform(0.5, 1.5)
    
    for k in range(1, num_modes + 1):
        amplitude_scale = 1.0 / (k**decay_power)
        a_k = rs.normal() * amplitude_scale
        b_k = rs.normal() * amplitude_scale
        signal += a_k * np.cos(k * t) + b_k * np.sin(k * t)
        
    return (signal - signal.min()) / (signal.max() - signal.min())

class RandomHydraulicsDataset:

    def __init__(self, inp_file, n_seconds, cT, Tau, dt, quality_dt, pattern_dt,
                 diameter_bounds, length_bounds, normalized_demand_bounds, pattern,
                 sources_at, overwrite=False, data_path='./dataset', equal_demand_at_nodes=False,
                 **sim_kwargs):
        '''
        inp_file: The inp file used for the simulation
        n_seconds: The number of seconds that the simulation runs 
            (n_seconds // Tau + 1 is the number of steps returned)
        cT: (s. Tau)
        Tau: cT corresponds to our (known) history and Tau to the number of
            time steps that we want to predict. cT and Tau parameters have no 
            effect on the data, they only effect the size of the x and y 
            output variables. So x will have shape [num_nodes, cT], 
            y will have shape [num_nodes, cT+Tau].
            (Note: Thus the whole simulation output is stored in y.
            cT and Tau are a little bit uneccessary to have, we should think 
            about removing them and only keeping n_seconds))
        dt: This is the hydraulics time step that is also the reporting time step.
        quality_dt: Quality time step used by EPANET
        pattern_dt: Pattern time step used by EPANET
        diameter_bounds: Used as distributional bounds to randomize pipe diameters
        length_bounds: Used as distributional bounds to randomize pipe lengths
        normalized_demand_bounds: Used as distributional bounds to randomize normalized demands
        pattern: The injection pattern (potentially also allow to randomize this)
        sources_at: The chlorine injection nodes
        data_path: The directory where data will be stored.
        sim_kwargs: Other simulation setup parameters.
        '''
        self.topology = functions.read_inp(inp_file)
        self.diameter_bounds = diameter_bounds 
        self.length_bounds = length_bounds
        # Different number of nodes affect the flow velocity, scale such that flow distributions remain similar
        upper_demand_bound = normalized_demand_bounds[1] * (HANOI_EDGES / self.n_links * self.n_reservoirs / HANOI_RESERVOIRS)
        self.average_demand_bounds = (normalized_demand_bounds[0], upper_demand_bound)
        self.inp_file = inp_file
        self.n_seconds = n_seconds
        self.dt = dt
        self.quality_dt = quality_dt
        self.pattern_dt = pattern_dt
        self.sim_kwargs = sim_kwargs
        self.pattern = pattern
        self.overwrite = overwrite
        # print(self.pattern.shape)
        # self.pattern = generate_injection_pattern(self.pattern_length, seed=0)
        # print(self.pattern.shape)
        self.sources_at = sources_at
        self.cT = cT
        self.Tau = Tau
        self.wds_name = os.path.basename(self.inp_file)[:-4] # remove .inp
        self.data_path = os.path.join(data_path, self.wds_name)
        self.diameter_order = self.topology.query_link_attribute('diameter').values.argsort()
        self.equal_demand_at_nodes = equal_demand_at_nodes
        if self.equal_demand_at_nodes:
            self.data_path = os.path.join(self.data_path, 'equal_demands')
        os.makedirs(self.data_path, exist_ok=True)
        self.initialize_variable_maxima()
    
    def initialize_variable_maxima(self):
        # convert from mm to m
        self.min_diameter, self.max_diameter = np.divide(self.diameter_bounds, 1000)
        # convert from L to m^3
        min_demand, max_demand = np.divide(self.average_demand_bounds, 1000)
        self.max_edge_capacity = (self.max_diameter / 2.)**2*np.pi
        self.min_edge_capacity = (self.min_diameter / 2.)**2*np.pi
        self.max_edge_length = self.length_bounds[1]
        self.max_flow = max_demand * self.topology.num_nodes
        self.max_vel = self.max_flow / self.min_edge_capacity
        self.max_delay_steps = (self.cT + self.Tau) * 2
        self.max_travel_time = self.max_delay_steps * self.dt

    @property
    def n_junctions(self): 
        return self.topology.num_junctions
    
    @property
    def n_nodes(self): 
        return self.topology.num_nodes
    
    @property
    def n_links(self): 
        return self.topology.num_links
    
    @property
    def n_reservoirs(self): 
        return len(self.topology.reservoir_name_list)
    
    @property
    def pattern_length(self):
        return int(self.n_seconds // self.pattern_dt) + 1
    
    def _generate_next(self):
        # dp = create_random_demand_profiles(
        #     *self.average_demand_bounds, self.n_junctions, self.pattern_length
        # )

        if not hasattr(self, 'junction_cluster_ids'):
            from scipy.linalg import toeplitz
            self.cluster_wn()
            r = np.linspace(0, 1, self.pattern_length)
            first_row = np.exp(-(r ** 2) / (2 * 0.1 ** 2))
            self.demand_cov = toeplitz(first_row)

        dp = create_random_demand_profiles_clustered(
            *self.average_demand_bounds, self.n_junctions, self.pattern_length, self.junction_cluster_ids, self.demand_cov
        )

        if self.equal_demand_at_nodes:
            dp = dp[:1].repeat(self.n_junctions, axis=0)
        lengths = np.random.uniform(*self.length_bounds, size=self.n_links)
        diameters = np.random.uniform(*self.diameter_bounds, size=self.n_links)
        # To generate more plausible networks, keep larger pipes large 
        # and smaller pipes small (large at reservoir, small at leaf nodes)
        ordered_dias = np.zeros_like(diameters)
        ordered_dias[self.diameter_order] = sorted(diameters)
        return dp, lengths, ordered_dias
    
    def cluster_wn(self):
        rel_n_clusters=0.01
        junction_mask = [ n not in self.topology.reservoir_name_list for n in self.topology.nodes ]
        n_clusters = max(4, int(self.topology.num_nodes * rel_n_clusters))
        self.node_cluster_ids = wds_utils.cluster_nodes(self.inp_file, n_clusters=n_clusters)
        self.junction_cluster_ids = self.node_cluster_ids[junction_mask]
    
    def get_sim_fn(self, sources_at=None):
        dp, lengths, diameters = self._generate_next()
        def _set_params(sim):
            functions.set_sim_demands(
                sim, base=[1.0] * self.n_junctions, pattern=dp, 
                nodelist=self.topology.junction_name_list
            ) 
            # TODO: (Fix) This overwrites the injection demands!
            #    If the injection mode is SETPOINT and injections are made 
            #    at a reservoir, then this works, otherwise
            # if sources_at is not None:
            #     functions.set_sim_demands(
            #     sim, base=[-1.0] * self.n_junctions, pattern=dp.sum(0), 
            #     nodelist=sources_at
            # )
            return
        return [_set_params]
    
    def get_inp_setup_fn(self):
        '''
        So far pipe lengths and diameters cannot be changes when EPANET already
        loaded the inp file. Here we create a function to modify the `in_inp`
        file and save it to the `out_inp` file.
        '''
        _, lengths, diameters = self._generate_next()

        def _set_params(in_inp, out_inp):
            wds_utils.set_pipe_lengths(in_inp, lengths, out_inp)
            wds_utils.set_pipe_diameters(out_inp, diameters, out_inp)
            
        return [_set_params]

    def add_normalization_values(self, sample):
        sample['max_diameter'] = self.max_diameter
        sample['max_edge_capacity'] = self.max_edge_capacity
        sample['max_edge_length'] = self.max_edge_length
        sample['max_vel'] = self.max_vel
        sample['max_delay_steps'] = self.max_delay_steps 
        sample['max_flow'] = self.max_flow
        return sample
    
    def get(self, idx):
        sample_path = os.path.join(self.data_path, f'{idx}.pkl')
        
        # Load a pickle file with the data if it has been simulated before.
        if not self.overwrite and os.path.isfile(sample_path):
            with open(sample_path, 'rb') as f:
                try:
                    return self.add_normalization_values(pickle.load(f))
                except Exception as e:
                    print(f"Error loading {sample_path}: {e}. Regenerating data.")
        
        self.pattern = generate_injection_pattern(self.pattern_length, seed=idx)
        
        sim_setup_fns = self.get_sim_fn(self.sources_at)
        sim_setup_fns = sim_setup_fns + self.sim_kwargs.get('sim_setup_fns', [])
        inp_setup_fns = self.get_inp_setup_fn() + self.sim_kwargs.get('inp_setup_fns', [])
        graph_data = simulate.inp_to_graph_data(
            self.inp_file, self.pattern, self.sources_at, self.n_seconds, self.dt, 
            pattern_dt=self.pattern_dt, progress=True, sim_setup_fns=sim_setup_fns, 
            inp_setup_fns=inp_setup_fns, **self.sim_kwargs
        )
        crossection = graph_data['edge_diameter']**2 * np.pi #* 60 * 60

        _, n_edges = graph_data['edge_index'].shape
        flow_field_graph = graph_data.pop('flow_field')[:n_edges] #/ 60 / 60

        t0 = timer.time()
        traversal_times, selfloop_mask, xs_map = modules.compute_backward_transit_times_fast(
            graph_data['edge_lengths'], flow_field_graph, self.dt
        )
        graph_data['flow_preprocessing_time'] = timer.time() - t0

        graph_data.pop('topology')
        graph_data.pop('res')
        
        sample = dict(
            **graph_data,
            flow_field_graph=flow_field_graph,
            edge_capacity=crossection,
            traversal_times=traversal_times,
            selfloop_mask=selfloop_mask,
            xs_map=xs_map,
            Tau=self.Tau,
            cT=self.cT,
            dt=self.dt,
        )

        # Save data as pickle file
        with open(sample_path, 'wb') as f:
            pickle.dump(sample, f)
            
        return self.add_normalization_values(sample)
    
class TorchRandomHydraulicsDataset(torch_geometric.data.Dataset):

    def __init__(self, size, hydraulics_ds, device='cpu', idx_offset=0, **sim_kwargs):
        super(TorchRandomHydraulicsDataset, self).__init__()
        self.hydraulics_ds = hydraulics_ds
        self.device = device
        self.data_path = './dataset'
        self.size = size
        self.idx_offset = idx_offset
        self.add_positional_encoding = False
        self.positional_encoding = None
        os.makedirs(self.data_path, exist_ok=True)
    
    def len(self):
        return self.size

    @property
    def dt(self):
        return self.hydraulics_ds.dt
    @property
    def cT(self):
        return self.hydraulics_ds.cT
    @property
    def Tau(self):
        return self.hydraulics_ds.Tau
    
    def get(self, idx):
        idx = idx + self.idx_offset
        sample = self.hydraulics_ds.get(idx)
        edge_capacities = sample['edge_capacity']

        if edge_capacities.ndim < 2 and edge_capacities.ndim > 0:
            edge_capacities = np.expand_dims(edge_capacities, axis=1)
        if sample['boundary_values'].ndim < 2:
            sample['boundary_values'] = np.expand_dims(sample['boundary_values'], axis=0)
            
        flows = sample['flow_field_graph'] * edge_capacities
        
        x = sample['epanet_result'][:, :self.cT]
        y = sample['epanet_result'][:, :self.cT+self.Tau]
        
        traversal_times = np.nan_to_num(sample['traversal_times'], nan=self.Tau*self.dt)
        xs_map = sample['xs_map']
        delay_steps = traversal_times / self.dt
        
        delay_steps = delay_steps.clip(max=self.cT+self.Tau)
        data = torch_geometric.data.Data(
            x=torch.as_tensor(x, dtype=torch.get_default_dtype()), 
            y=torch.as_tensor(y, dtype=torch.get_default_dtype()), 
            edge_index=torch.as_tensor(sample['edge_index']),
            edge_lengths=torch.as_tensor(sample['edge_lengths'], dtype=torch.get_default_dtype()),
            edge_diameter=torch.as_tensor(sample['edge_diameter'], dtype=torch.get_default_dtype()),
            edge_capacity=torch.as_tensor(sample['edge_capacity'], dtype=torch.get_default_dtype()),
            traversal_times=torch.as_tensor(traversal_times, dtype=torch.get_default_dtype()),
            #xs_map=torch.as_tensor(xs_map, dtype=torch.get_default_dtype()),
            flows=torch.as_tensor(flows, dtype=torch.get_default_dtype()),
            sl_mask=torch.as_tensor(sample['selfloop_mask'], dtype=torch.get_default_dtype()),
            boundary_values=torch.as_tensor(sample['boundary_values'], dtype=torch.get_default_dtype()),
            boundary_index=torch.as_tensor(sample['boundary_index']),
            delay_steps=torch.as_tensor(delay_steps, dtype=torch.get_default_dtype()),
            flow_field=torch.as_tensor(sample['flow_field_graph'], dtype=torch.get_default_dtype()),
            
            max_diameter=torch.as_tensor(sample['max_diameter'], dtype=torch.get_default_dtype()),
            max_edge_capacity=torch.as_tensor(sample['max_edge_capacity'], dtype=torch.get_default_dtype()),
            max_edge_length=torch.as_tensor(sample['max_edge_length'], dtype=torch.get_default_dtype()),
            max_vel=torch.as_tensor(sample['max_vel'], dtype=torch.get_default_dtype()),
            max_delay_steps=torch.as_tensor(sample['max_delay_steps'], dtype=torch.get_default_dtype()),
            max_flow=torch.as_tensor(sample['max_flow'], dtype=torch.get_default_dtype()),
        )
        if 'runtime' in sample:
            data['epanet_runtime'] = torch.as_tensor(sample['runtime'], dtype=torch.get_default_dtype())
            data['flow_preprocessing_time'] = torch.as_tensor(sample['flow_preprocessing_time'], dtype=torch.get_default_dtype())
        
        if self.add_positional_encoding:
            if self.positional_encoding is None:
                pe_encoder = torch_geometric.transforms.AddLaplacianEigenvectorPE(
                    k=30, attr_name='pe', is_undirected=True
                )
                self.positional_encoding = pe_encoder(data)['pe']
            data['pe'] = self.positional_encoding
        return data