from wntr.epanet import io
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from wntr.network import TimeSeries
# from matplotlib.animation import FuncAnimation
# from matplotlib import colors
# from collections import defaultdict
from itertools import repeat
from sklearn.cluster import SpectralClustering
# import os

PI = float(np.pi)


COLOR_THEME = [
    "#332288",
    "#7593DD",
    "#ABEAFF",
    '#FC9432',
    "#2BB50C",
    "#CB4335",
    "#C676F5",
    "#A43FAD",
    "#760964",

    
    # "#0B5394",  # Model 1: Deep Sapphire Blue
    # "#1D87D9",  # Model 2: Bright Slate Blue
    # "#317070",  # Model 3: Dark Teal / Emerald
    
    # # --- THE BASELINES (Vibrant, distinct warm tones) ---
    # "#E65C00",  # Baseline 1: Deep Orange
    # "#CB4335",  # Baseline 2: Crimson Red
    # "#D4AC0D",  # Baseline 3: Dark Mustard Gold
    # "#884EA0"   # Baseline 4: Amethyst Purple
    # "#225EA8",  # Deep Blue (Strong baseline, replaces #3580D6)
    # "#41B6C4",  # Teal/Light Blue (High contrast, replaces #AFB7F5)
    # "#5E3C99",  # Deep Royal Purple (Replaces #64309A/#760964)
    # "#E66101",  # Crisp Orange (Pops perfectly, replaces #FC9432)
    # "#2CA25F",  # Forest Green (More professional than the neon #2BB50C)
    # "#B2ABD2",  # Muted Lavender (Excellent contrast spacer, replaces #9F5ECE)
    # "#D01C8B",  # Vibrant Magenta (For a sharp accent, replaces #C177BA)
    # "#FDB863"   # Soft Amber (For complex plots needing an 8th color)
]
# convert HEX to RGB: https://stackoverflow.com/a/29643643
COLOR_THEME = [ tuple(int(c.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) + (1.0,) for c in COLOR_THEME ]

ALL_MODELS = [
    "MeGA-MP",
    "MeGA-MP+MLP",
    "MeGA-MP-learned",
    "NNConv",
    "PDE-GNN",
    "GPSConv",
    "AntiSymmetricConv",
    "AntiSymmetricConv-Dia",
]

ZORDER = [
    "NNConv",
    "PDE-GNN",
    "GPSConv",
    "AntiSymmetricConv",
    "AntiSymmetricConv-Dia",
    "MeGA-MP-learned",
    "MeGA-MP",
    "MeGA-MP+MLP",
    "Simulator"
]

MODEL_COLOR = dict(zip(ALL_MODELS, COLOR_THEME))
MODEL_COLOR['RK4'] = COLOR_THEME[2]
MODEL_COLOR['Semi-Lagrangian'] = COLOR_THEME[4]

def read_inp(inp_file):
    return io.InpFile().read(inp_files=inp_file)

def get_edge_attribute(topology, name):
    return np.array(
        [ topology.get_link_info(l)[name] for l, _ in topology.get_all_links() ]
    )
def get_edge_attribute_dict(topology, name):
    return { (n1, n2) : topology.get_link_info(l)[name] for l, (n1, n2) in topology.get_all_links() }

def query_node_attribute(topology, name):
    return { n : topology.get_node_info(n)[name] for n in topology }

def compute_edge_time_coeffs(diameters, lengths):
    crosssection = (diameters / 2)**2 * PI
    crosssection[crosssection == 0] = 1e-10 # flow velocity through 0-diameter pipes are infinite
    return lengths * crosssection

def delay_to_flows(diameters, lengths, delay, dt):
    edge_time_coeffs = compute_edge_time_coeffs(diameters, lengths)
    flows = edge_time_coeffs / ((delay) * dt)
    return flows

def create_wavy_pattern(pattern_length, n, dt, seed=None):
    rs = np.random.RandomState(seed)
    pattern = np.sum([ 
        n/(n-i+1) * (1 + np.sin(np.linspace(0, 1, pattern_length) * dt * 2 * np.pi * rs.normal(i*0.01)))
        for i in range(1, n)
    ], axis=0)
    pattern /= pattern.max()
    return pattern

def spectral_sensor_select(topo, N, weight='diameter', seed=None):
    G = nx.Graph(topo)
    rs = np.random.RandomState(seed)
    if weight is not None:
        nx.set_edge_attributes(G, get_edge_attribute_dict(topo, weight), weight)
    adj = nx.adjacency_matrix(G, weight=weight, dtype=np.float32).todense()
    adj /= adj.max()
    sc = SpectralClustering(n_clusters=N, affinity='precomputed')
    node_cluster_ids = sc.fit_predict(adj)
    nt = zip(node_cluster_ids, nx.get_node_attributes(G, 'info').items())
    junction_cluster_ids, junctions = zip(*[ (c, n) for c, (n, i) in nt if i['type'] == 'JUNCTION' ])
    junction_cluster_ids, junctions = np.array(junction_cluster_ids), np.array(junctions)
    sensors = [ rs.choice(junctions[junction_cluster_ids == c]) for c in range(N) ]
    return sensors, node_cluster_ids

def make_dag_edge_index(topo, flows, nodelist=None):
    edges, edge_index = zip(*map(
        lambda l: (l[0][0], get_node_index(topo, l[0][1])[::int(np.sign(l[1]+1e-8))]),
        filter(
            lambda l: l[0][1][0] in nodelist and l[0][1][1] in nodelist, 
            zip(topo.get_all_links(), flows)
        )
        if nodelist is not None else 
        zip(topo.get_all_links(), flows)
    ))
    edge_index = np.stack(edge_index, 1)
    return edge_index

def make_edge_index(topo, nodelist=None, bidirectional=False):
    edges, edge_index = zip(*map(
        lambda l: (l[0], get_node_index(topo, l[1])),
        filter(
            lambda l: l[1][0] in nodelist and l[1][1] in nodelist, 
            topo.get_all_links()
        )
        if nodelist is not None else 
        topo.get_all_links()
    ))
    edge_index = np.stack(edge_index, 1)
    if bidirectional:
        return np.concatenate((edge_index, edge_index[::-1]), axis=1)
    return edge_index

def wds_to_flow_tree(topology, flow_data, weighted=False):
    inv_mask = flow_data < 0
    edgelist = np.array(list(zip(*list(topology.get_all_links())))[1])
    edgelist = [ (v, u) if inv else (u, v)  for (_, (u, v)), inv in zip(topology.get_all_links(), inv_mask) ]
    tree_topo = topology.to_directed()
    tree_topo.remove_edges_from(np.array(tree_topo.edges))
    all_edges = np.concatenate(edgelist)
    tree_topo.add_edges_from(edgelist)
    if weighted:
        weights = dict(zip(map(tuple, all_edges), np.abs(flow_data)))
        nx.set_edge_attributes(tree_topo, weights, 'weight')
    return tree_topo

def build_path_lengths(G, target_node, edge_lengths, length=[], path=[], attrib=None):
    neighbors = list(G.predecessors(target_node))
    if len(neighbors) == 0:
        return [length], [path]
    lengths, paths = [], []
    for n in neighbors:
        if attrib is None:
            addlen = edge_lengths.get((target_node, n), edge_lengths.get((n, target_node)))
        else:
            addlen = attrib.get((target_node, n), attrib.get((n, target_node)))
            if addlen is None: addlen = 0.
        l, p = build_path_lengths(
            G, n, edge_lengths, length + [addlen], path + [(target_node, n)], attrib=attrib
        )
        lengths.extend(l)
        paths.extend(p)
    return lengths, paths

def trace_sensor_edges(edge_index, sensor_nodes, agg_time=None, stop_at_sensors=True):
        # if agg_time is None:
        #     node_active = sensor_nodes
        #     finished_edges = node_active[edge_mask[0]]
        #     edge_mask = (1 - finished_edges).bool()
        # else:
        #     agg_time, node_time, finished_edges = agg_time
        #     finished_nodes, _ = scatter_min(finished_edges, edge_index[1], dim=0, out=new_node_time)

        snd, rec = edge_index

        if agg_time is None:
            node_mask = sensor_nodes.int()
            edge_mask_new = node_mask[snd]
            return (node_mask, edge_mask_new, edge_mask_new.bool())

        node_mask, edge_mask_new, _ = agg_time

        edge_mask = edge_mask_new.clamp(0, 1)
        #node_mask, _ = scatter_min(edge_mask, rec, 0, node_mask.int() + 1)
        from torch_scatter import scatter_max
        node_mask_new, _ = scatter_max(edge_mask, rec, 0, node_mask)
        if stop_at_sensors:
            node_mask_new = node_mask_new * (~sensor_nodes)
        edge_mask_new = node_mask_new[edge_index[0]]
        edge_active = (edge_mask_new.clamp(0,1) - edge_mask).clamp(0,1)

        return node_mask_new - node_mask, edge_mask_new, edge_active.clamp(0, 1).bool()

def find_switching_edges(flows):
    return ((flows > 0).min(0) == (flows < 0).min(0))

def flow_to_velocity(topology, flow_data, unit='CMH'):
    assert unit == 'CMH'
    diameters = get_edge_attribute(topology, 'diameter')
    diameters = diameters / 10 / 100 # convert mm to meters
    diameters = np.broadcast_to(diameters[None], flow_data.shape)
    crosssection = (diameters / 2)**2 * np.pi
    crosssection[crosssection == 0] = 1e-10 # TODO: Changed to 1e-10 flow velocity through 0-diameter pipes are infinite
    flow_velocities = flow_data / crosssection
    flow_velocities = np.nan_to_num(flow_velocities) # [m/h]
    return flow_velocities / 60 / 60 # [m/s]

def get_edge_travel_times(topology, edge_flows=None, edge_velocities=None):
    if edge_velocities is None:
        assert edge_flows is not None
        edge_velocities = flow_to_velocity(topology, edge_flows)
    time_per_edge = get_edge_attribute(topology, 'length') / edge_velocities
    return np.nan_to_num(time_per_edge)

def query_node_attribute(topology, name):
    return { n : topology.get_node_info(n)[name] for n in topology }

def get_node_index(topology, nodes):
    if isinstance(nodes, str):
        nodes = [nodes]
    return [ topology.get_all_nodes().index(n) for n in nodes ]

def set_sim_demands(sim: ScenarioSimulator, base=None, pattern=None, nodelist=None, multiplier=1.):
    if nodelist is None:
        nodelist = sim.get_topology().get_all_junctions()
    if pattern is None:
        pattern = np.array([1.])
    if isinstance(base, list):
        if np.ndim(pattern) == 1:
            pattern = repeat(pattern)
        else:
            assert np.ndim(pattern) == 2, ('Pattern can either be shaped '
                '[n_times] or [n_nodes, n_times]'
            )

        for b, patt, node in zip(base, pattern, nodelist):
            sim.set_node_demand_pattern(node, b * multiplier, 'Constant', patt)
        return
    elif base is None:
        node_pattern_indices = sim.epanet_api.get_node_demand_patterns_idx()
        for node in nodelist:
            nodeidx = sim.epanet_api.get_node_idx(node)
            node_pattern_index = node_pattern_indices[nodeidx-1]
            base_demands = sim.epanet_api.get_node_base_demands(nodeidx)
            if len(base_demands) == 0:
                print('Not setting demand for node', node)
                continue
            base_demand = base_demands[0]
            node_pattern = sim.epanet_api.get_pattern(node_pattern_index)
            demand_mean = base_demand * node_pattern.mean()
            sim.set_node_demand_pattern(node, demand_mean * multiplier, 'Constant', pattern)
    else:
        for node in nodelist:
            sim.set_node_demand_pattern(node, base * multiplier, 'Constant', pattern)

def set_sim_injection_supplies(sim: ScenarioSimulator, supply=None, nodelist=None):
    for node in nodelist:
        sim.set_node_demand_pattern(node, supply, 'Constant', np.array([1.]))

def scale_diameters(sim: ScenarioSimulator, scale: float=1.0):
    for link, _ in sim.epanet_api.get_all_links_id():
        linkidx = sim.epanet_api.get_link_idx(link)
        diameter = sim.epanet_api.get_link_diameter(linkidx)
        sim.epanet_api.setlinkvalue(linkidx, EpanetConstants.EN_DIAMETER, diameter * scale)

def set_sim_diameters(sim: ScenarioSimulator, diameters):
    dias_it = iter(sorted(diameters, reverse=True))
    all_links = sim.epanet_api.get_all_links_id()
    lidx = lambda l: sim.epanet_api.get_link_idx(l)
    all_diameters = [ 
        (sim.epanet_api.get_link_diameter(lidx(l)), lidx(l)) for l, _ in all_links 
    ]
    for _, lidx in sorted(all_diameters, reverse=True):
        sim.epanet_api.setlinkvalue(lidx, EpanetConstants.EN_DIAMETER, next(dias_it))

def set_sim_lengths(sim: ScenarioSimulator, lengths: list[float]):
    length_it = iter(lengths)
    for link, _ in sim.epanet_api.get_all_links_id():
        linkidx = sim.epanet_api.get_link_idx(link)
        # check if link has length
        _ = sim.epanet_api.get_link_length(linkidx)
        sim.epanet_api.setlinkvalue(linkidx, EpanetConstants.EN_LENGTH, next(length_it))

def iswntr(topology):
    return not hasattr(topology, 'get_node_info')

def get_nodes_of_type(topology, type):
    if iswntr(topology):
        return list(map(lambda n: n[0], filter(lambda n: n[1].node_type == type.capitalize(), topology.nodes())))
    else: # wntr topology
        return [ n[0] for n in topology.nodes(data=True) if n[1]['info']['type'] == type ]

def get_all_links(topology):
    if iswntr(topology):
        return list(map(lambda p: (p[1].start_node.name, p[1].end_node.name), topology.links()))
    else:
        return list(zip(*topology.get_all_links()))[1]

def plot_graph_from_topology(
        topology, node_colors='#96a6d4', edge_colors='#2c436d', labels=False, width=3,
        show_colorbar=False, show_colorbar_pipes=False, ax=None, node_size=380, arrows=False,
        edge_labels=False, node_type_props=None
    ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    if iswntr(topology):
        G = topology.to_graph()
        allnodes = list(list(zip(*topology.nodes()))[0])
        #node_map = dict(zip(G.nodes, range(len(G.nodes))))
        #pos = nx.get_node_attributes(topology, 'coord')
        pos = nx.get_node_attributes(topology.to_graph(), 'pos')
        pumps = [ (v,u) for u, v, d in G.edges(data=True) if d['type'] == 'PUMP' ]
    else:
        allnodes = topology.get_all_nodes()
        Graph = nx.Graph if not arrows else nx.DiGraph
        G = Graph(list(zip(*topology.get_all_links()))[1])# nx.Graph(topology)#  # 
        #node_map = dict(zip(G.nodes, range(len(G.nodes))))
        pos = query_node_attribute(topology, 'coord') # 
        # pumps = [ (v,u) for u, v, d in G.edges(data=True) if d['info']['type'] == 'PUMP' ]
        pumps = [ topology.get_link_info(p)['nodes'] for p in topology.get_all_pumps() ]
    #pos = dict(zip(node_map.values(), pos.values()))
    #pos = { node_map[n] : topology.get_node_info(n)['coord'] for n in topology }
    # pos = { n : topology.get_node_info(n)['coord'] for n in topology }
    #G = nx.relabel_nodes(G, node_map)
    i = 0
    
    pipes_plot = nx.draw_networkx_edges(
        G, pos=pos, width=width, edge_color=edge_colors, ax=ax, node_size=node_size, 
        arrows=arrows, edgelist=get_all_links(topology)
        #edgelist=[ (u,v) for u, v, d in G.edges(data=True) if d['info']['type'] == 'PIPE' ]
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos=pos, edge_labels=dict(zip(G.edges(), topology.link_name_list))
        )
    if not arrows:
        pipes_plot.set_label('Pipe')
    pipes_plots = [pipes_plot]
    for pu, pv in pumps:
        pumps_plot = nx.draw_networkx_nodes(
            nx.Graph([[0,0]]), pos={ 0 : np.add(pos[pu], pos[pv])/2 }, node_color=node_colors, ax=ax, node_size=node_size/2,
            node_shape='D'
        )
    if len(pumps):
        pumps_plot.set_label('Pump')
        pipes_plot = pipes_plots.append(pumps_plot)
    #node_types = [ n[1]['info']['type'] for n in topology.nodes(data=True) ]
    node_plots = []
    for marker, node_type in zip('ovs', ['JUNCTION', 'TANK', 'RESERVOIR']):
        # subset = [ n[0] for n in G.nodes(data=True) if n[1].get('info', n[1])['type'] == node_type ]
        subset = get_nodes_of_type(topology, node_type)
        if not len(subset):
            continue
        idxs = [ allnodes.index(n) for n in subset ]
        nc = node_colors[idxs] if len(node_colors) == len(allnodes) else node_colors
        ns = node_size
        if node_type_props is not None and node_type.lower() in node_type_props:
            nc = node_type_props[node_type.lower()].get('color', nc)
            ns = node_type_props[node_type.lower()].get('size', ns)
        node_plot = nx.draw_networkx_nodes(
            nx.subgraph(G, subset), pos=pos, node_color=nc, ax=ax, node_size=ns, node_shape=marker, 
            nodelist=subset
        )
        if labels:
            nx.draw_networkx_labels(
                nx.subgraph(G, subset), pos=pos, ax=ax, #labels=dict(zip(G.nodes, node_map.keys()))
            )
        node_plot.set_label(node_type.capitalize())
        node_plots.append(node_plot)
    ax.axis('off')
    
    if show_colorbar:
        plt.colorbar(node_plot)

    return [ *pipes_plots, *node_plots, ]
