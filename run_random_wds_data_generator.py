import networkx as nx
from itertools import count
import numpy as np
import os
import wntr
import utils as functions
import sys
from tqdm.auto import tqdm
from run_experiments_pde_gnn import _prepare_dataset_split, TIME_CONFIG, DEFAULT_CT, DEFAULT_TAU, EXP_SETUPS
from run_experiments_random_wds import RANDOM_WDS_CONFIG
from experiment_shared import init_dataset
from torch_geometric.loader import DataLoader

def synthetic_water_network(n=10, remove_prob=0.25, add_diag_prob=0.05, seed=None):
    random = np.random.RandomState(seed)
    G = nx.grid_2d_graph(n, n)
    
    # remove random edges
    for e in list(G.edges()):
        if random.rand() < remove_prob:
            G.remove_edge(*e)
    
    # add diagonal shortcuts
    for i in range(n):
        for j in range(n):
            for d1, d2 in [(1,1),(-1,1)]:
                if i+d1 < n and j+d2 < n and i+d1 >= 0 and j+d2 >= 0:
                    if random.rand() < add_diag_prob:
                        G.add_edge((i,j), (i+d1, j+d2))

    node_set = max(nx.connected_components(G), key=len)

    if len(node_set) < n:
        return synthetic_water_network(n, remove_prob, add_diag_prob)
    
    G = nx.subgraph(G, node_set)
    pos = [ [node[0], node[1]] for node in G.nodes() ]
    G = nx.relabel_nodes(G, dict(zip(G.nodes, count())))

    return G, pos

def get_network(n, remove_prob, add_diag_prob, fraction_of_source_nodes, diameter_range, seed=None):
    random = np.random.RandomState(seed)
    G, pos = synthetic_water_network(n=n, remove_prob=remove_prob, add_diag_prob=add_diag_prob, seed=random.randint(0, 10000))
    
    edge_index = np.stack(G.edges)
    
    n_sources = random.randint(1, 1+max(2, int(G.number_of_nodes() * fraction_of_source_nodes)))
    sources = random.choice(G.nodes, size=n_sources, replace=False)

    dists = [1e10] * len(G)
    for i, n in enumerate(G.nodes):
        for si, s in enumerate(sources):
            dists[i] = min(dists[i], nx.shortest_path_length(G, s, n))

    dists = np.stack(dists)
    amin, amax = 1, dists.max()

    # Diameter placement (large at reservoirs small at leafs)
    amin, amax = 1, dists.max()
    ps = (amax / np.linspace(1, amax, 100))
    ps = (ps) / (ps).sum()
    a = random.choice(np.linspace(amin, amax, 100), p=ps)

    dists = (dists / a).round() * a # + (_a-1) * (amax/_a)#**0.1
    dists = dists / dists.max() * amax

    edge_dist_to_source = dists[edge_index].min(1)
    edge_dist_to_source = 1 - edge_dist_to_source / edge_dist_to_source.max()
    d_range = diameter_range[1] - diameter_range[0]
    d_range = d_range # / len(sources)
    distance_scale = d_range#np.random.normal(d_range, (d_range/2)*0.78, n_edges)
    diameters = diameter_range[0] + edge_dist_to_source * distance_scale
    
    return G, sources, diameters

def to_wntr(G, sources, diameters):
    pos = nx.kamada_kawai_layout(G)

    n_nodes = G.number_of_nodes()

    wn = wntr.network.WaterNetworkModel()
    for i in range(n_nodes):
        if i in sources:
            wn.add_reservoir(f'{i}', base_head=100, coordinates=tuple(pos[i]))
        else:
            wn.add_junction(f'{i}', coordinates=tuple(pos[i]))
    for e, (i, j) in enumerate(G.edges()):
        wn.add_pipe(f'{i}->{j}', f'{i}', f'{j}', diameter=diameters[e], length=1000, roughness=100)

    wn.options.hydraulic.demand_model = 'DD'
    wn.options.hydraulic.inpfile_units = 'CMH'
    return wn

def generate_synthetic_inp_file(n, p, seed=None, inp_output_dir='networks/random_networks', overwrite_data=False):
    inp_file = os.path.join(inp_output_dir, f'synthetic_network_{seed}.inp')
    if not overwrite_data and os.path.exists(inp_file): 
        return inp_file
    G, sources, diameters = get_network(n=n, remove_prob=(1-p), add_diag_prob=p, fraction_of_source_nodes=0.03, diameter_range=(0.06, 0.25), seed=seed)
    wn = to_wntr(G, sources, diameters)
    wntr.network.write_inpfile(wn, inp_file)
    return inp_file

def generate_random_wds(inp_files, device='cpu', overwrite_data=False):
    config = RANDOM_WDS_CONFIG.copy()

    for inp_file in inp_files:
        config["inp_file"] = str(inp_file)
        wn = functions.read_inp(inp_file)
        config["sources_at"] = wn.reservoir_name_list

        ds = init_dataset(
            config, DEFAULT_CT, DEFAULT_TAU, device=device, 
            time_config=TIME_CONFIG, overwrite=overwrite_data
        )
        data_loader = DataLoader(ds, batch_size=10)
        [ _ for _ in tqdm(data_loader, desc="Generating data ...") ]

if __name__ == "__main__":
    from multiprocessing import Pool
    from functools import partial
    
    start_idx = int(sys.argv[1])
    step = int(sys.argv[2])
    overwrite_data = True

    inp_output_dir = 'networks/random_networks'
    os.makedirs(inp_output_dir, exist_ok=True)
    ns, ps = np.reshape(np.meshgrid(
        # Determines the number of nodes in the network (minimum sqrt(n))
        range(10, 32+1), 
        # Determines the connectivity of the network (0.1 = very sparse, 0.9 = very dense)
        np.linspace(0.1, 0.9, 10)
    ), (2, -1))
    ns = ns.astype(int)
    random = np.random.RandomState(42)
    params = [(int(n), p, random.randint(0, 1e8)) for n, p in zip(ns, ps)][start_idx::step]
    
    # generate_synthetic_inp_file(ns, ps, 4)
    jobs = Pool(processes=10).starmap(
        partial(generate_synthetic_inp_file, inp_output_dir=inp_output_dir, overwrite_data=overwrite_data), 
        tqdm(params, total=len(params)), 
        chunksize=5
    )
    inp_files = [ f for f in jobs ]
    
    generate_random_wds(
        inp_files, overwrite_data=overwrite_data
    )