from collections import defaultdict
from wntr.epanet import io
from utils import read_inp
from wntr.network import TimeSeries
import networkx as nx
import numpy as np
import os
from tempfile import NamedTemporaryFile

from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_array
from wntr.network.model import WaterNetworkModel

def merge_pipe_data_seq(topology, pipes):
    pipe_info = defaultdict(lambda: 0)

    for p in pipes:
        pipe = topology.get_link(p)
        pipe_info['crossection'] += (pipe.diameter/2)**2 * np.pi
        pipe_info['roughness'] += pipe.roughness
        pipe_info['minor_loss'] += pipe.minor_loss
        pipe_info['length'] += pipe.length

    pipe_info['roughness'] /= len(pipes)
    pipe_info['minor_loss'] /= len(pipes)
    pipe_info['crossection'] /= len(pipes)

    pipe_info['diameter'] = np.sqrt(pipe_info['crossection'] / np.pi) * 2
    pipe_info.pop('crossection')
    return dict(pipe_info)

def merge_demands(timeseriesA, timeseriesB, scaleA, scaleB):
    vA = timeseriesA.base_value * scaleA
    vB = timeseriesB.base_value * scaleB
    mA = timeseriesA.pattern.multipliers
    mB = timeseriesB.pattern.multipliers
    if vA != 0:
        new_base_value = vA
        new_multipliers = (vA * mA + vB * mB) / vA
    elif vB != 0:
        new_base_value = vB
        new_multipliers = (vB * mB + vA * mA) / vB
    else:
        new_base_value = 0.
        new_multipliers = mA * 0.
        
    return new_base_value, new_multipliers

def remove_inline_nodes(inp_file, out_file):
    topology = read_inp(inp_file)

    G = topology.to_graph().to_undirected()

    remove_nodes = [ n for n in G.nodes() if G.degree(n) == 2 ]

    for n in remove_nodes:
        add_edge = list(G.neighbors(n))
        remove_pipes = list(topology.get_links_for_node(n))
        remove_edges = list(G.edges(n))
        is_pipe = [ topology.get_link(p).link_type == 'Pipe' for p in remove_pipes ]
        if not all(is_pipe):
            continue
        G.add_edge(*add_edge, type='Pipe')
        G.remove_edges_from(remove_edges)
        pipe_info = merge_pipe_data_seq(topology, remove_pipes)

        for e, p in zip(remove_edges, remove_pipes):
            topology.remove_link(p)
            
        topology.add_pipe(
            f'rep_{n}-{add_edge[0]}-{add_edge[1]}', *add_edge, **pipe_info
        )

        node = topology.get_node(n)
        
        if node.node_type == 'Junction':
            u = topology.get_node(add_edge[0])
            v = topology.get_node(add_edge[1])
            is_junction_u = u.node_type == 'Junction'
            is_junction_v = v.node_type == 'Junction'

            for _n in [u, v]:
                for i, dta in enumerate(topology.get_node(n).demand_timeseries_list):
                    if _n.node_type == 'Junction':
                        pname = pnames = f'R-{_n.name}_{n}_{i}'
                        dtb = _n.demand_timeseries_list[i]
                        new_base_value, new_multipliers = merge_demands(dtb, dta, scaleA=1., scaleB=0.5)
                        topology.add_pattern(pname, new_multipliers)
                        new_demand = TimeSeries(topology.patterns, new_base_value, pname)
                        _n.demand_timeseries_list[i] = new_demand

        topology.remove_node(n)
        topology.add_pipe(
            f'rep_{n}-{add_edge[0]}-{add_edge[1]}', *add_edge, **pipe_info
        )
    
    io.InpFile().write(out_file, topology)

def convert_to_injection_nodes(inp_file, nodes, out_file):
    topology = read_inp(inp_file)
    source_nodes = []

    for node in nodes:
        n = topology.get_node(node)

        if hasattr(n, 'demand_timeseries_list'):
            source_nodes.append(n)
        else:
            for l in topology.get_links_for_node(node):
                sn = topology.get_link(l).start_node_name
                en = topology.get_link(l).end_node_name
                _node = sn if sn != node else en
                n = topology.get_node(_node)
                source_nodes.append(n)
        
    for n in source_nodes:
        supply_ts = TimeSeries(topology.patterns, -1.0)
        n.demand_timeseries_list.clear()
        n.demand_timeseries_list.append(supply_ts)
        
    io.InpFile().write(out_file, topology)

def set_pipe_lengths(inp_file, lengths, out_file):
    topology = read_inp(inp_file)

    for (_, pipe), length in zip(topology.pipes(), lengths):
        pipe.length = length
        
    io.InpFile().write(out_file, topology)

def set_pipe_diameters(inp_file, diameters, out_file):
    topology = read_inp(inp_file)

    for (_, pipe), diameter in zip(topology.pipes(), diameters):
        # wntr uses meters instead of mm -> divide by 1000
        pipe.diameter = diameter / 1000
    
    io.InpFile().write(out_file, topology)

class InpSetup:

    def __init__(self, inp_file, file_setup_fns):
        self.inp_file = inp_file
        self.file_setup_fns = file_setup_fns
        self.inactive = not self.file_setup_fns

    def __enter__(self):
        if self.inactive:
            return self.inp_file
        self.temp_inp_file = NamedTemporaryFile(mode='w', suffix='.inp')
        temp_inp_file_name = self.temp_inp_file.name
        inp_file = self.inp_file

        for fn in self.file_setup_fns:
            fn(inp_file, temp_inp_file_name)
            inp_file = self.temp_inp_file
            
        return temp_inp_file_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.inactive:
            return
        self.temp_inp_file.close()

def get_edge_index(topology):
    return np.stack(list(map(
        lambda l: (
            topology.node_name_list.index(l[1].start_node_name),
            topology.node_name_list.index(l[1].end_node_name)
        ),
        topology.links()
    ))).astype(np.int32).T

def merge_pipe_data(topology, pipes):
    pipe_info = defaultdict(lambda: 0)
    #print(pipes)
    for p in pipes:
        pipe = topology.get_link(p)
        pipe_info['hw_coeff_total'] += (10.67 * pipe.length / (
            pipe.roughness**1.852 * pipe.diameter**4.87
        ))#**(-1/1.852)

    pipe_info['hw_coeff_total'] = pipe_info['hw_coeff_total']#**(-1.852)

    for p in pipes:
        pipe = topology.get_link(p)
        hw = 10.67 * pipe.length / (
            pipe.roughness**1.852 * pipe.diameter**4.87
        )
        s = hw/pipe_info['hw_coeff_total']
        pipe_info['crossection'] += (pipe.diameter/2)**2 * np.pi #* s
        pipe_info['roughness'] += pipe.roughness #* s
        pipe_info['minor_loss'] += pipe.minor_loss #* s
        pipe_info['length'] += pipe.length #* s

    pipe_info['roughness'] /= len(pipes)
    pipe_info['minor_loss'] /= len(pipes)
    pipe_info['length'] /= len(pipes)
    # pipe_info['hw_coeff'] /= len(pipes)
    pipe_info['diameter'] = np.sqrt(pipe_info['crossection'] / np.pi) * 2
    
    # pipe_info['roughness'] = (
    #     10.67 * pipe_info['length'] / (
    #         pipe_info['hw_coeff_total'] * pipe_info['diameter']**4.87
    #     )
    # )#**(1/1.852)

    pipe_info.pop('crossection')
    # pipe_info.pop('hw_coeff')
    pipe_info.pop('hw_coeff_total')

    return dict(pipe_info)

def cluster_wds(inp_file, out_file, n_clusters=10):
    topology = read_inp(inp_file)

    edge_index = get_edge_index(topology)
    # Use diameters for clustering (add as argument)
    diameters = topology.query_link_attribute('diameter').values

    sc = SpectralClustering(n_clusters, affinity='precomputed')
    adj = csr_array((diameters, edge_index), shape=[topology.num_nodes] * 2)
    sc.fit(adj)
    clusters = sc.labels_

    junction_mask = topology.query_node_attribute('node_type') == 'Junction'
    reservoir_mask = topology.query_node_attribute('node_type') == 'Reservoir'
    n_res = sum(reservoir_mask)
    clusters[reservoir_mask] = np.arange(n_clusters, n_clusters+n_res)
    nodes_per_cluster = np.bincount(clusters)

    snd_cluster, rec_cluster = clusters[edge_index]
    cluster_nodes = np.arange(n_clusters + n_res)
    ic_edge_mask = snd_cluster != rec_cluster
    ic_edges_orig = edge_index[:,ic_edge_mask]
    ic_edge_names = np.array(topology.link_name_list)[ic_edge_mask]
    inter_cluster_edges = clusters[ic_edges_orig]
    # sort sender and receiver to find unique edges
    inter_cluster_edges = np.sort(inter_cluster_edges, axis=0)
    ic_edges, idx, inv, counts = np.unique(
        inter_cluster_edges, return_inverse=True, return_index=True, 
        return_counts=True, axis=1
    )
    cluster_pos = np.zeros((n_clusters + n_res, 2))
    pos = np.stack(topology.query_node_attribute('coordinates').values)
    cluster_pos = { n : np.mean(pos[clusters == n], axis=0) for n in cluster_nodes }

    get_ts = lambda n: topology.get_node(n).demand_timeseries_list[0]
    get_demand = lambda n: get_ts(n).base_value * get_ts(n).pattern.multipliers

    ic_pipe_infos = {}
    for i, e in enumerate(ic_edges.T):
        #print(i, [f'{topology.get_link(l).start_node_name} -> {topology.get_link(l).end_node_name}' for l in ic_edge_names[inv == i]])
        #print(ic_edge_names[inv == i], inv.shape, ic_edges.shape)
        ic_pipe_infos[(*e,)] = merge_pipe_data(topology, ic_edge_names[inv == i])

    demands = np.vectorize(get_demand, signature='()->(m)')(topology.junction_name_list)
    elevations = topology.query_node_attribute('elevation')

    c_demands = np.zeros((n_clusters + n_res, demands.shape[1]))
    np.add.at(c_demands, clusters[junction_mask], demands)

    c_elevations = np.zeros(n_clusters + n_res)
    np.add.at(c_elevations, clusters[junction_mask], elevations)
    c_elevations = c_elevations / nodes_per_cluster

    wn = WaterNetworkModel()

    for n, demand, elevation in zip(cluster_nodes, c_demands, c_elevations):
        if n >= n_clusters:
            continue
        wn.add_pattern(f'p{n}', demand)
        wn.add_junction(
            str(n), base_demand=1., demand_pattern=f'p{n}', elevation=elevation,
            coordinates=(*cluster_pos[n],)
        )

    for (r, reservoir), c in zip(topology.reservoirs(), clusters[reservoir_mask]):
        wn.add_reservoir(str(c), reservoir.base_head, coordinates=reservoir.coordinates)

    for i, (s, r) in enumerate(ic_edges.T):
        wn.add_pipe(str(i), str(s), str(r), **ic_pipe_infos[(s, r)])

    wn._options = topology.options
    io.InpFile().write(out_file, wn)
    return ic_edge_mask, ic_edges, inv, clusters, nodes_per_cluster

def cluster_nodes(inp_file, n_clusters=None):
    wn = read_inp(inp_file)
    G = wn.get_graph()

    adj = nx.adjacency_matrix(G)
    adj.indices = adj.indices.astype(np.int32)
    adj.indptr = adj.indptr.astype(np.int32)

    clusters = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit_predict(adj)
    return clusters