from run_experiments_pde_gnn import _get_model_config, DEFAULT_CT, PDE_GNN_CONFIG, TIME_CONFIG, RESULTS_DIR, _configure_antisymmetric_dia, DEFAULT_EVAL_MODELS
from experiment_shared import evaluate, init_model, safe_torch_load
from pathlib import Path
import os
import torch
import argparse
import networkx as nx
import utils as functions
import pickle
import numpy as np
from wds_experiment_utils import key_to_label
from run_experiments_euclidean_1d import draw_discretization_axis
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import pearsonr

root = PDE_GNN_CONFIG["root"]
RANDOM_WDS_CONFIG = {
    "msx_file" : "networks/complicated_reactions.msx",
    "data_path": os.path.join(root, "dataset_random_wds"),
    "equal_demand_at_nodes" : False,
    "idx_offset" : 0,
    "size" : 5,
}
RANDOM_WDS_RESULTS_DIR = os.path.join(root, "random_wds_results")

def __inp_files():
    return list(Path("networks/random_networks").glob('*.inp'))

def run_evaluation(model_names=None, device=None):
    if model_names is None:
        model_names = DEFAULT_EVAL_MODELS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = RANDOM_WDS_CONFIG.copy()
    
    model_dir = os.path.join(RESULTS_DIR, "experiment_2")
    random_inp_files = __inp_files()
    if len(random_inp_files) == 0:
        raise ValueError("No random inp files found in networks/random_networks. Please generate them first via experiments/random_water_networks.py.")

    for model_name in model_names:
        base_config = _get_model_config(
            "AntiSymmetricConv" if model_name == "AntiSymmetricConv-Dia" else model_name,
            device,
        )
        model_config = base_config
        model = init_model(model_config, device=device)
        model_path = os.path.join(model_dir, model_name)
        model_path = model_path.replace("AntiSymmetricConv-Dia", "AntiSymmetricConv")
        
        if not (
            model_config["model_name"].startswith("MeGA-MP") and model_config["layer"] == "advect"
        ):
            model_file = os.path.join(model_path, "model.pt")
            model.load_state_dict(
                safe_torch_load(model_file, weights_only=True, map_location=device)
            )
        
        for inp_file in random_inp_files:
            config["inp_file"] = str(inp_file)
            wn = functions.read_inp(inp_file)
            config["sources_at"] = wn.reservoir_name_list
            diameter = nx.diameter(nx.Graph(wn.to_graph()))
            wds_name = inp_file.stem

            if model_name == "AntiSymmetricConv-Dia":
                model.anti_symm_conv.epsilon = model_config["epsilon"] * model_config["num_iters"] / diameter
                model.anti_symm_conv.num_iters = diameter
            

            results_path = os.path.join(RANDOM_WDS_RESULTS_DIR, model_name, "results")
            os.makedirs(results_path, exist_ok=True)
            
            result = evaluate(
                DEFAULT_CT,
                model_config["eval_Tau"],
                config,
                model_config,
                model_path,
                batch_size=1,
                model=model,
                device=device,
                time_config=TIME_CONFIG,
            )

            result["key"] = wds_name
            out_file = os.path.join(results_path, f"eval_{wds_name}.pkl")

            with open(out_file, "wb") as f:
                pickle.dump(result, f)

def draw_example_networks(seed=42):
    from run_random_wds_data_generator import get_network

    random = np.random.RandomState(seed)
    np.random.seed(seed) # Required for layout reproducibility of networkx

    fig, axs = plt.subplots(5, 5, figsize=(18, 15), constrained_layout=True)
    ax_iter = iter(axs.flatten())

    for p in np.linspace(0.1, 0.9, 5):
        for n in range(10, 32+1, 5):
            G, sources, diameters = get_network(n=n, remove_prob=(1-p), add_diag_prob=p, fraction_of_source_nodes=0.03, diameter_range=(0.06, 0.25), seed=random.randint(0, 10000))
            pos = nx.kamada_kawai_layout(G)
            
            _, node_degrees = zip(*nx.degree(G))
            avg_degree = np.mean(node_degrees)
            num_nodes = G.number_of_nodes()

            ax = next(ax_iter)
            nx.draw(G, pos=pos, with_labels=False, node_size=10, node_color="#4698da", edge_color='#77c1ff', width=diameters*10, ax=ax)
            nx.draw_networkx_nodes(nx.subgraph(G, sources), pos=pos, node_size=80, node_color="#125a96", label="Sources", ax=ax)
            ax.text(0, 0, rf"$|V|={num_nodes}$" + "\n" + rf"Avg. Deg.$={avg_degree:.1f}$", 
                    fontsize=12, ha='left', va='bottom', fontdict={'color':"#125a96"}, transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', boxstyle='round,pad=0.3')
            )

    plt.draw()
    supax = draw_discretization_axis(fig, axs, [], [])
    leg = axs[0][-1].legend(
        [plt.scatter([], [], color='#125a96', s=80), plt.plot([], [], color='#77c1ff')[0], plt.scatter([], [], color='#4698da')], 
        ['Reservoir', 'Pipe', 'Junction'], 
        loc='upper right', 
        bbox_to_anchor=(1.0, 1.0), 
        ncol=3, 
        fontsize=14
    )
    leg.set_in_layout(False)

    supax.set_xlabel('Graph Size ->', fontsize=16)
    supax.set_ylabel('<- Graph Density', fontsize=16, rotation=90)

    fig.savefig(os.path.join(RANDOM_WDS_RESULTS_DIR, 'random_wds_graphs.pdf'), bbox_inches='tight')

def read_results(model_names):
    random_inp_files = __inp_files()
    config = RANDOM_WDS_CONFIG.copy()

    # model_names = ["NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "MeGA-MP", "MeGA-MP+MLP", "MeGA-MP-learned"]

    bins = [10, 400, 750, 1100]
    density_bins = [1, 5, 8]
    wds_metrics = {
        'num_nodes' : {}, 'avg_degree' : {}, 'avg_transport' : {},
        'Preprocessing' : {}, 'Simulator' : {}, 'zmask' : {}, 'avg_flow' : {}, 'k_ups' : {}
    }
    metrics = { 'mae' : {}, 't_model' : {} }

    for inp_file in tqdm(random_inp_files):
        config["inp_file"] = str(inp_file)
        wn = functions.read_inp(inp_file)
        G = nx.Graph(wn.to_graph())
        config["sources_at"] = wn.reservoir_name_list
        wds_name = inp_file.stem


        _, node_degrees = zip(*nx.degree(G))
        avg_degree = np.mean(node_degrees)
        num_nodes = G.number_of_nodes()
        wds_size_binned = np.digitize(num_nodes, bins)
        density_binned = np.digitize(np.mean(node_degrees), density_bins)
        
        results_path = os.path.join(
            RANDOM_WDS_RESULTS_DIR, model_names[0], "results"
        )

        with open(os.path.join(results_path, f"eval_{wds_name}.pkl"), "rb") as f:
            res = pickle.load(f)
            y_true = res["ground_truths"]
        
        all_transporttimes = []
        for data_file in (Path(config['data_path']) / wds_name).glob('*.pkl'):
            with open(data_file, 'rb') as f:
                dat = pickle.load(f)
                all_transporttimes.append(np.mean(np.abs(dat['traversal_times']), where=~np.isnan(dat['traversal_times'])))

        for k in wds_metrics.keys():
            wds_metrics[k][wds_size_binned] = wds_metrics[k].get(wds_size_binned, {})
            wds_metrics[k][wds_size_binned][density_binned] = wds_metrics[k][wds_size_binned].get(density_binned, [])

        wds_metrics['Preprocessing'][wds_size_binned][density_binned].append(res['preprocessing_times'].cpu().numpy())
        wds_metrics['Simulator'][wds_size_binned][density_binned].append(res['epanet_runtimes'].cpu().numpy())
        
        wds_metrics['zmask'][wds_size_binned][density_binned].append(np.isclose(y_true, 0).mean(-1) >= 0.1)
        wds_metrics['avg_flow'][wds_size_binned][density_binned].append(res['flow_fields'].abs().mean())
        wds_metrics['k_ups'][wds_size_binned][density_binned].append(700/(res['edge_lengths'][:,None] / (res['flow_fields'].abs() * 60)).abs().min())
        wds_metrics['num_nodes'][wds_size_binned][density_binned].append(num_nodes)
        wds_metrics['avg_degree'][wds_size_binned][density_binned].append(avg_degree)
        wds_metrics['avg_transport'][wds_size_binned][density_binned].append(np.mean(all_transporttimes))

        for model_name in model_names:
            for _, m in metrics.items():
                m[model_name] = m.get(model_name, {})
                m[model_name][wds_size_binned] = m[model_name].get(wds_size_binned, {})
                m[model_name][wds_size_binned][density_binned] = m[model_name][wds_size_binned].get(density_binned, [])
            eval_path = os.path.join(
                RANDOM_WDS_RESULTS_DIR, model_name, "results"
            )
            with open(os.path.join(eval_path, f"eval_{wds_name}.pkl"), "rb") as f:
                res = pickle.load(f)
                pred = res["predictions"].clamp(0, 1)

            mae = torch.nn.L1Loss(reduction="none")(pred, y_true)
            # mse = torch.nn.MSELoss(reduction="none")(pred, y_true)
            metrics['mae'][model_name][wds_size_binned][density_binned].append(mae.cpu().numpy())
            metrics["t_model"][model_name][wds_size_binned][density_binned].append(res['model_runtimes'].cpu().numpy())
            
    return metrics, wds_metrics

def plot_error_results(metrics, wds_metrics, model_name):

    y_mean = flatten(map_structure(lambda m: m.mean(), metrics['mae'][model_name]))
    y_err = flatten(map_structure(lambda m: m.mean(-1).std(), metrics['mae'][model_name]))
    y_mean2 = flatten(map_structure(lambda m, zm: m[~zm].mean(), metrics['mae'][model_name], wds_metrics['zmask']))
    ns = flatten(wds_metrics['num_nodes'])
    ds = flatten(wds_metrics['avg_degree'])
    avg_flow = flatten(wds_metrics['avg_flow'])
    avg_trans = flatten(wds_metrics['avg_transport'])

    # 1. plot MAE
    fig, axes = plt.subplots(2, 2, figsize=(14, 5), sharey=True, constrained_layout=True)

    plots = [
        (axes[0, 0], ns, "Number of Nodes"),
        (axes[0, 1], avg_trans, "Average Transport Time"),
        (axes[1, 0], ds, "Avg. Node Degree"),
        (axes[1, 1], avg_flow, "Average Flow"),
    ]

    PRIMARY_COLOR = "#2b5c8f"
    ERROR_COLOR = "#8fa3ba"

    for ax, x_data, xlabel in plots:
        ax.errorbar(
            x_data, y_mean, yerr=y_err,
            fmt='o',
            markersize=3,
            color=PRIMARY_COLOR,
            ecolor=ERROR_COLOR,
            elinewidth=1.2,
            capsize=0,
            alpha=0.85,
            zorder=3, 
            label=r'MAE'
        )
        ax.scatter(x_data, y_mean2, color=PRIMARY_COLOR, fc='none', s=32, zorder=4, label='Masked MAE')
        r_log, p_log = pearsonr(np.log10(x_data), y_mean)
        r_log2, _ = pearsonr(np.log10(x_data), y_mean2)
        p_text = "p < 0.001" if p_log < 0.001 else f"p = {p_log:.3f}"
        stats_annotation = f"$r = {r_log:.2f}\;|\;r_{{masked}}={r_log2:.2f}$\n{p_text}"
        ax.text(0.95, 0.95, stats_annotation, transform=ax.transAxes, fontsize=10, ha='right', va='top', color=PRIMARY_COLOR, fontweight='medium')
        
        ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='medium', labelpad=6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        ax.grid(True, which='major', linestyle='--', alpha=0.5, color='#e0e0e0', zorder=1)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#e0e0e0', zorder=1)

    axes[0, 0].set_ylabel('MAE', fontsize=11, fontweight='medium')
    axes[1, 0].set_ylabel('MAE', fontsize=11, fontweight='medium')

    maps, labs = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend([maps[1], maps[0]], [labs[1], labs[0]])

    fig.savefig('additional_figures/random_wds_error_analysis.pdf', bbox_inches='tight')

def plot_runtime_analysis(metrics, wds_metrics, model_name):
    y_mean = flatten(map_structure(lambda m: m.mean(), metrics['mae'][model_name]))
    runtime_mean = flatten(map_structure(lambda m: m.mean(), metrics['t_model'][model_name]))
    runtime_std = flatten(map_structure(lambda m: m.std(), metrics['t_model'][model_name]))
    ns = flatten(wds_metrics['num_nodes'])
    ds = flatten(wds_metrics['avg_degree'])
    avg_flow = flatten(wds_metrics['avg_flow'])
    avg_trans = flatten(wds_metrics['avg_transport'])
    n_edges = np.array(ns) * np.array(ds) * 0.5

    fig, axes = plt.subplots(2, 2, figsize=(14, 5), sharey=False, constrained_layout=True)

    plots = [
        (axes[0, 0], n_edges, "Number of Edges", avg_trans, "Avg. Transport Time (sec)"),
        (axes[0, 1], avg_trans, "Avg. Transport Time (sec)", n_edges, "Number of Edges"),
        (axes[1, 0], ds, "Graph Density", avg_trans, "Avg. Transport Time (sec)"),
        (axes[1, 1], avg_flow, "Average Flow", n_edges, "Number of Edges"),
    ]

    PRIMARY_COLOR = "#2b5c8f"
    ERROR_COLOR = "#8fa3ba"

    for ax, x_data, xlabel, color, clabel in plots:
        
        for _x, _y, _yerr, c in zip(x_data, runtime_mean, runtime_std, color):
            ax.errorbar(
                _x, _y, yerr=_yerr,
                fmt='o',
                markersize=3,
                color=plt.cm.jet(c/np.max(color)),
                ecolor=ERROR_COLOR,
                elinewidth=1.2,
                capsize=0,           # Removing horizontal caps usually yields a cleaner look
                alpha=0.85,
                zorder=3
            )
        plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=np.min(color), vmax=np.max(color))), ax=ax, label=clabel)

        r_log, p_log = pearsonr((x_data), y_mean)
        p_text = "p < 0.001" if p_log < 0.001 else f"p = {p_log:.3f}"
        stats_annotation = f"$r = {r_log:.2f}$\n{p_text}"
        ax.text(0.95, 0.95, stats_annotation, transform=ax.transAxes, fontsize=10, ha='right', va='top', color=PRIMARY_COLOR, fontweight='medium')
        
        ax.set_xlabel(xlabel, fontsize=11, fontweight='medium', labelpad=6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        ax.grid(True, which='major', linestyle='--', alpha=0.5, color='#e0e0e0', zorder=1)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#e0e0e0', zorder=1)

    axes[0, 0].set_ylabel('Comp. Time (sec)', fontsize=11, fontweight='medium')
    axes[1, 0].set_ylabel('Comp. Time (sec)', fontsize=11, fontweight='medium')
    axes[0, 1].set_ylabel('Comp. Time (sec)', fontsize=11, fontweight='medium')
    axes[1, 1].set_ylabel('Comp. Time (sec)', fontsize=11, fontweight='medium')

    fig.savefig('additional_figures/random_wds_runtime_analysis.pdf', bbox_inches='tight')

def print_latex_table(metrics, wds_metrics, model_names):
    latex_out = ""
    template = "{:.4f}{{\\scriptsize$\\pm {:.4f}$}}"
    for model_name in model_names:
        latex_out += f"& {model_name}"
        for wds_size, _metrics in sorted(metrics['mae'][model_name].items()):
            for density, m in sorted(_metrics.items()):
                all_maes = np.concatenate(m)
                latex_out += " & "
                latex_out += template.format(all_maes.mean(), all_maes.std())
        latex_out += " \\\ \n"
    latex_out += " \hline \n"
    for model_name in model_names:
        latex_out += f"& {model_name}"
        for wds_size, _metrics in sorted(metrics['t_model'][model_name].items()):
            for density, m in sorted(_metrics.items()):
                all_rts = np.concatenate(m)
                latex_out += " & "
                latex_out += template.format(all_rts.mean(), all_rts.std())
        latex_out += " \\\ \n"
    # latex_out += " \cdashline{2-8} \n"
    for k in ['Preprocessing', 'Simulator']:
        latex_out += "\n \cdashline{2-8} \n"
        latex_out += f"& {k}"
        for wds_size, _m in sorted(wds_metrics[k].items()):
            for density, m in sorted(_m.items()):
                all_rts = np.concatenate(m)
                latex_out += " & "
                latex_out += template.format(all_rts.mean(), all_rts.std())
        latex_out += " \\\ \n"


    with open(os.path.join(RANDOM_WDS_RESULTS_DIR, f"mae_table_latex.txt"), "w") as f:
        latex_out = key_to_label(latex_out)
        f.write(latex_out)

    print(latex_out)

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = argparse.ArgumentParser()
    args.add_argument("--eval_models", nargs="+", default=[])
    args.add_argument("--reevaluate", action="store_true", default=False)
    args = args.parse_args()

    if args.reevaluate:
        run_evaluation(model_names=args.eval_models, device=device)

    draw_example_networks(seed=42)
    
    # model_names = ["NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "MeGA-MP", "MeGA-MP+MLP", "MeGA-MP-learned"]

    metrics, wds_metrics = read_results(args.eval_models)
    plot_runtime_analysis(metrics, wds_metrics, "MeGA-MP+MLP")
    plot_error_results(metrics, wds_metrics, "MeGA-MP+MLP")
    print_latex_table(metrics, wds_metrics, args.eval_models)