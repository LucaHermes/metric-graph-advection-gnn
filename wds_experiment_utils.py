from copy import deepcopy
import pickle

import matplotlib.pyplot as plt
import matplotlib.legend as mlegend
import numpy as np
import shapely
from matplotlib import colors

import utils as functions
import wds_utils
from experiment_shared import resolve_time_config


def init_inp_file(
    inp_file_source,
    inp_file_out,
    sources_at,
    remove_inline_nodes=True,
    n_clusters=None,
):
    # Remove nodes with exactly two neighbors to simplify the topology.
    inp_file = inp_file_source

    if remove_inline_nodes:
        wds_utils.remove_inline_nodes(inp_file, inp_file_out)
        inp_file = inp_file_out

    if n_clusters is not None:
        wds_utils.cluster_wds(inp_file, inp_file_out, n_clusters=n_clusters)
        inp_file = inp_file_out

    wds_utils.convert_to_injection_nodes(inp_file, sources_at, inp_file_out)


def _resolve_sources(sources_at):
    return list(sources_at) if sources_at is not None else []

def key_to_label(key, glossary=True):
    if glossary:
        key = key.replace("MeGA-MP", "\\gls*{MeGA-MP}")
        key = key.replace("GNN", "\\gls*{GNN}")
        key = key.replace("PDE", "\\gls*{PDE}")
    key = key.replace("+MLP", "$^+$")
    key = key.replace("-learned", "$^-$")
    key = key.replace("AntiSymmetricConv-Dia", "A-DGN$_{Dia}$")
    key = key.replace("AntiSymmetricConv", "A-DGN")
    return key

def plot_with_map(
    data,
    flow_field,
    topology,
    sources_at=None,
    plot_nodes=None,
    plt_kwargs=None,
    cT=200,
    Tau=500,
    time_config=None,
):
    time_config = resolve_time_config(time_config)
    sources_at = _resolve_sources(sources_at)
    plt_kwargs = plt_kwargs or []

    if plot_nodes is None:
            plot_nodes = topology.node_name_list

    n = len(plot_nodes)

    fig, ax = plt.subplot_mosaic(
        "WWW0\nWWW1\nWWW2\nWWW3\nWWW4\nWWW5\nWWW6\nWWW7",
        figsize=(14, 7),
        constrained_layout=True,
    )

    nsteps = data[0].shape[1]
    
    highlight_colors = [(0.89, 0.48, 0.15, 1.0)] * n
    lg = (0.9, 0.9, 0.9, 1.0)

    edge_data = np.abs(flow_field).mean(1)
    edge_norm = colors.LogNorm(edge_data.min(), edge_data.max())
    edge_cmap = plt.cm.Blues

    functions.plot_graph_from_topology(
        topology,
        node_colors=lg,
        edge_colors=edge_cmap(edge_norm(edge_data)),
        ax=ax["W"],
        node_size=15,
        width=3,
    )
    ax["W"].margins(-0.02)

    pos = topology.query_node_attribute("coordinates")
    ts = (
        np.linspace(0, (cT + Tau) * time_config["hydraulic_step"] + 0.1, nsteps)
        / 60
        / 60
    )

    if sources_at:
        ax["W"].scatter(
            *np.transpose([pos[s] for s in sources_at]),
            marker="*",
            zorder=2,
            s=400,
            label="Injection Nodes",
            color=(0.4, 0.6, 0.9, 1),
            ec="k",
        )

    for i, node in enumerate(plot_nodes):
        node_idx = topology.node_name_list.index(node)
        for d, kwargs in zip(data, plt_kwargs):
            ax[str(i)].plot(ts, d[node_idx, :nsteps], **kwargs)
        ax[str(i)].set_xlim(min(ts), max(ts))
        ax[str(i)].set_ylim(-0.05, 1.05)
        ax[str(i)].set_ylabel(
            i + 1,
            rotation=0,
            fontdict={
                "weight": "bold",
                "fontsize": 13,
                "color": highlight_colors[i],
            },
            va="center",
        )
        ax["W"].text(
            *pos[node],
            i + 1,
            fontdict={
                "weight": "bold",
                "fontsize": 13,
                "color": highlight_colors[i],
            },
            va="center",
            ha="center",
            zorder=3,
        )
        ax["W"].scatter(*pos[node], color=(1, 1, 1, 0.8), zorder=2, s=650)
        ax["W"].scatter(
            *pos[node], color=(*highlight_colors[i][:-1], 0.25), zorder=2, s=650
        )
        if i < n - 1:
            ax[str(i)].set_xticklabels([])
        else:
            ax[str(i)].set_xlabel("Time [hours]")

    handles, labels = ax["W"].get_legend_handles_labels()
    linehandles, linelabels = ax["0"].get_legend_handles_labels()

    if len(handles) > 2:
        del handles[2]
        del labels[2]

    x1, y1, w1, _h1 = ax["W"].bbox.bounds
    x, y = fig.transFigure.inverted().transform([x1 + 0.55 * w1, y1])
    w, _ = fig.transFigure.inverted().transform([w1 * 0.45, y1])
    cax = fig.add_axes((x, y, w, 0.03))

    pipe_h = plt.Rectangle([0, 0], 3, 1, color=plt.cm.Blues(0.9), label="Pipe")
    ax["W"].legend([pipe_h, *handles[1:], *linehandles], [*labels, *linelabels])

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_label(r"Avg. Flow Velocity [$\frac{m}{s}$]")

    return fig, ax


def plot_results_with_map(
    result_file,
    out_file,
    sources_at=None,
    cT=200,
    Tau=500,
    time_config=None,
):
    with open(result_file, "rb") as f:
        results, graph_data, topology = pickle.load(f)

    fig, _ax = plot_with_map(
        [results["pred"], graph_data["epanet_result"]],
        graph_data["flow_field"],
        topology,
        sources_at=sources_at,
        plt_kwargs=[
            dict(linestyle="-", linewidth=2, color="C0", label="MPNN"),
            dict(linestyle=":", linewidth=2, color="C1", label="Ground Truth"),
        ],
        cT=cT,
        Tau=Tau,
        time_config=time_config,
    )

    fig.savefig(out_file, bbox_inches="tight")


def plot_on_map(
    data,
    topology,
    sources_at=None,
    plt_kwargs=None,
    cT=200,
    Tau=500,
    time_config=None,
):
    time_config = resolve_time_config(time_config)
    sources_at = _resolve_sources(sources_at)
    plt_kwargs = plt_kwargs or []

    fig, ax = plt.subplots(figsize=(14, 8))

    plot_nodes = topology.node_name_list
    np.random.RandomState(5).shuffle(plot_nodes)
    lg = (0.9, 0.9, 0.9, 1.0)

    functions.plot_graph_from_topology(
        topology, node_colors=lg, edge_colors=lg, ax=ax, node_size=15, width=3
    )

    nsteps = data[0].shape[1]
    ts = (
        np.linspace(0, (cT + Tau) * time_config["hydraulic_step"], nsteps)
        / 60
        / 60
    )
    pos = topology.query_node_attribute("coordinates")

    axis_polygon = shapely.MultiPolygon()
    # Make sure nothing occludes source nodes
    if sources_at:
        axis_polygon = axis_polygon.union(
            shapely.MultiPoint(
                [
                    fig.transFigure.inverted().transform(ax.transData.transform(pos[s]))
                    for s in sources_at
                ]
            )
        )

    if sources_at:
        xys = np.transpose([pos[s] for s in sources_at])
        ax.scatter(*xys, marker="*", zorder=2, s=200, label="Injection Nodes", c="C0")

    line_handles = []
    line_labels = []

    for node in plot_nodes:
        node_idx = topology.node_name_list.index(node)
        x, y = fig.transFigure.inverted().transform(ax.transData.transform(pos[node]))
        x, y, w, h = (x, y, 0.18 * 0.7, 0.1 * 0.8)

        ax_poly = shapely.Polygon.from_bounds(x, y, x + w, y + h).buffer(0.03)
        if ax_poly.intersects(axis_polygon):
            continue
        axis_polygon = axis_polygon.union(shapely.Polygon.from_bounds(x, y, x + w, y + h))
        mini_ax = fig.add_axes((x, y, w, h))

        ax.scatter(*pos[node], s=100, color=(0.3, 0.3, 0.3, 1.0), zorder=2)

        for d, kwargs in zip(data, deepcopy(plt_kwargs)):
            line_labels.append(kwargs.pop("label"))
            line_plot, = mini_ax.plot(ts, d[node_idx, :nsteps], **kwargs)
            line_handles.append(line_plot)

        mini_ax.set_xlim(min(ts), max(ts))
        mini_ax.set_xticks([round(max(ts))])
        mini_ax.set_yticks([1])
        mini_ax.set_ylim(-0.05, 1.05)
        mini_ax.set_fc((1, 1, 1, 0.6))

    handles, labels, _ = mlegend._parse_legend_args([ax])
    ax.legend([*handles, *line_handles], [*labels, *line_labels])
    return fig, ax


def plot_results_on_map(
    result_file,
    out_file,
    sources_at=None,
    cT=200,
    Tau=500,
    time_config=None,
):
    with open(result_file, "rb") as f:
        results, graph_data, topology = pickle.load(f)

    fig, _ax = plot_on_map(
        [results["pred"], graph_data["epanet_result"]],
        topology,
        sources_at=sources_at,
        plt_kwargs=[
            dict(linestyle="-", linewidth=2, color="C0", label="MPNN"),
            dict(linestyle=":", linewidth=2, color="C1", label="Ground Truth"),
        ],
        cT=cT,
        Tau=Tau,
        time_config=time_config,
    )

    fig.savefig(out_file, bbox_inches="tight")
