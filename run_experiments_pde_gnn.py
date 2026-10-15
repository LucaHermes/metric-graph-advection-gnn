import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pickle
from copy import deepcopy
from itertools import product

import pandas as pd
from sklearn.metrics import explained_variance_score

from experiments.experiment_configs import CONFIGS
from utils import MODEL_COLOR, ZORDER
from experiment_config_loader import get_pde_gnn_config
from experiment_shared import evaluate, resolve_time_config, run_experiment
from wds_experiment_utils import key_to_label, plot_with_map

try:
    mp.set_start_method("spawn", force=True)
    print("spawned")
except RuntimeError:
    pass

PDE_GNN_CONFIG = get_pde_gnn_config()
TIME_CONFIG = resolve_time_config(PDE_GNN_CONFIG.get("time"))

RESULTS_DIR = PDE_GNN_CONFIG.get("results_dir")
EXP_SETUPS = {exp["id"]: exp for exp in PDE_GNN_CONFIG.get("experiments", [])}
DEFAULT_CT = PDE_GNN_CONFIG.get("cT", 1)
DEFAULT_TAU = PDE_GNN_CONFIG.get("Tau", 700)
HYDRAULIC_STEP = TIME_CONFIG["hydraulic_step"]

DEFAULT_TRAIN_MODELS = [
    "MeGA-MP",
    "MeGA-MP+MLP",
    "MeGA-MP-learned",
    "NNConv",
    "PDE-GNN",
    "GPSConv",
    "AntiSymmetricConv",
]
DEFAULT_EVAL_MODELS = [
    "NNConv",
    "PDE-GNN",
    "MeGA-MP",
    "MeGA-MP+MLP",
    "MeGA-MP-learned",
    "GPSConv",
    "AntiSymmetricConv",
    "AntiSymmetricConv-Dia",
]


def _get_model_config(model_name, device):
    config_key = "MeGA-MP" if model_name == "MeGA-MP+MLP" else model_name
    model_config = deepcopy(CONFIGS[config_key])
    model_config["device"] = device
    model_config["layer"] = "advect_react"
    if model_name == "MeGA-MP":
        model_config["layer"] = "advect"
    elif model_name == "MeGA-MP-learned":
        model_config["layer"] = "advect_react_learned"
    return model_config


def _prepare_dataset_split(exp_config):
    exp_cfg = deepcopy(exp_config)

    dataset_cfg = exp_cfg["train"]
    dataset_cfg["size"] = exp_cfg["data_splits"]["train"]
    dataset_cfg["idx_offset"] = 0
    idx_offset = dataset_cfg["size"]

    val_cfg = deepcopy(exp_cfg.get("validation", exp_cfg["train"]))
    val_cfg["size"] = exp_cfg["data_splits"]["validation"]
    val_cfg["idx_offset"] = idx_offset
    exp_cfg["validation"] = val_cfg

    idx_offset += exp_cfg["data_splits"]["validation"]

    dataset_cfg = exp_cfg["test"]

    if isinstance(dataset_cfg, dict):
        for sub_cfg in dataset_cfg.values():
            sub_cfg["size"] = exp_cfg["data_splits"]["test"]
            sub_cfg["idx_offset"] = idx_offset
    else:
        dataset_cfg["size"] = exp_cfg["data_splits"]["test"]
        dataset_cfg["idx_offset"] = idx_offset

    return exp_cfg


def _configure_antisymmetric_dia(model_config, wds_name=None, diameter=None):
    if diameter is None:
        diameter = 13 if wds_name == "Hanoi" else 79
    model_config = deepcopy(model_config)
    model_config["epsilon"] = model_config["epsilon"] * model_config["num_iters"] / diameter
    model_config["num_iters"] = diameter
    return model_config


def run_training(model_names=None, exp_ids=None, device=None, overwrite=False):
    if model_names is None:
        model_names = DEFAULT_TRAIN_MODELS
    if exp_ids is None:
        exp_ids = list(EXP_SETUPS.keys())
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_names:
        if model_name == "AntiSymmetricConv-Dia":
            continue

        model_config = _get_model_config(model_name, device)

        for exp_id in exp_ids:
            exp_config = deepcopy(EXP_SETUPS[exp_id])
            model_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", model_name)
            os.makedirs(model_path, exist_ok=True)
            train_cfg = _prepare_dataset_split(exp_config)
            model_file = os.path.join(model_path, "model.pt")
            
            if not os.path.exists(model_file) or overwrite:
                run_experiment(
                    DEFAULT_CT,
                    model_config["train_Tau"],
                    train_cfg,
                    model_config,
                    model_path,
                    device=device,
                    time_config=TIME_CONFIG,
                )

def run_evaluation(model_names=None, exp_ids=None, device=None):
    if model_names is None:
        model_names = DEFAULT_EVAL_MODELS
    if exp_ids is None:
        exp_ids = list(EXP_SETUPS.keys())
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_names:
        base_config = _get_model_config(
            "AntiSymmetricConv" if model_name == "AntiSymmetricConv-Dia" else model_name,
            device,
        )
        for exp_id in exp_ids:
            exp_config = deepcopy(EXP_SETUPS[exp_id])
            model_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", model_name)
            results_path = os.path.join(model_path, "results")
            model_path = model_path.replace("AntiSymmetricConv-Dia", "AntiSymmetricConv")
            os.makedirs(results_path, exist_ok=True)
            train_cfg = _prepare_dataset_split(exp_config)["test"]
            
            for wds_name, eval_cfg in train_cfg.items():
                model_config = base_config
                if model_name == "AntiSymmetricConv-Dia":
                    model_config = _configure_antisymmetric_dia(base_config, wds_name)
                
                result = evaluate(
                    DEFAULT_CT,
                    model_config["eval_Tau"],
                    eval_cfg,
                    model_config,
                    model_path,
                    device=device,
                    time_config=TIME_CONFIG,
                )
                result["key"] = wds_name
                out_file = os.path.join(results_path, f"eval_{wds_name}.pkl")
                with open(out_file, "wb") as f:
                    pickle.dump(result, f)


def run_plots(model_names=None, exp_ids=None, out_prefix=''):
    if not model_names:
        return #model_names = DEFAULT_EVAL_MODELS
    if exp_ids is None:
        exp_ids = list(EXP_SETUPS.keys())

    # Sample to choose for Hanoi and L-Town respectively for plotting.
    # These samples are cherry-picked to show interesting samples with a complex time series.
    sample_ids = [0, 1]
    plot_n = 8

    for exp_id in exp_ids:
        exp_config = deepcopy(EXP_SETUPS[exp_id])
        results_path = os.path.join(
            RESULTS_DIR, f"experiment_{exp_id}", model_names[0], "results"
        )
        out_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}")

        for sample_idx, wds_name in zip(sample_ids, exp_config["test"].keys()):
            with open(os.path.join(results_path, f"eval_{wds_name}.pkl"), "rb") as f:
                result = pickle.load(f)

            topology = result["topology"]
        
            n_nodes = topology.num_nodes
            batch_slice = slice(n_nodes * sample_idx, n_nodes * (sample_idx + 1))
            y_true = result["ground_truths"][batch_slice, :]
            flow_fields = result["flow_fields"]
            sources_at = result["sources_at"]

            plot_nodes = np.random.RandomState(32).choice(topology.node_name_list, size=plot_n)
            # plot_nodes = [*plot_nodes[1:], next(topology.reservoirs())[0]]

            preds = []
            subset_on_map = model_names.copy()
            
            for model_name in model_names:
                eval_path = os.path.join(
                    RESULTS_DIR, f"experiment_{exp_id}", model_name, "results"
                )
                with open(os.path.join(eval_path, f"eval_{wds_name}.pkl"), "rb") as f:
                    pred = pickle.load(f)["predictions"][batch_slice]
                    if pred.abs().max() > 1e3:
                        subset_on_map.remove(model_name)
                        continue
                    preds.append(pred[:, : y_true.shape[1]])
            
            if subset_on_map:
                subset_preds = [preds[subset_on_map.index(name)] for name in subset_on_map]
                fig, _ax = plot_with_map(
                    [*subset_preds, y_true],
                    flow_fields[: topology.num_links].cpu(),
                    topology,
                    sources_at=sources_at,
                    plot_nodes=plot_nodes,
                    plt_kwargs=[
                        *[
                            dict(
                                linestyle="-",
                                linewidth=2,
                                color=MODEL_COLOR[name],
                                alpha=0.7 if "MeGA-MP" not in name else 1.0,
                                label=key_to_label(name, glossary=False),
                                zorder=ZORDER.index(name) if name in ZORDER else 10,
                            )
                            for name in subset_on_map
                        ],
                        dict(linestyle=":", linewidth=2, color="k", label="Simulator", zorder=11),
                    ],
                    cT=DEFAULT_CT,
                    Tau=DEFAULT_TAU,
                    time_config=TIME_CONFIG,
                )
                fig.savefig(
                    os.path.join(out_path, f"{out_prefix}{wds_name}_all_models_with_map.pdf"),
                    bbox_inches="tight",
                )
                plt.close(fig)

            fig, axs = plt.subplots(4, 2, figsize=(12, 5), sharex=True, sharey=True, tight_layout=True)
            nsteps = preds[0].shape[1]
            ts = (
                np.linspace(0, DEFAULT_TAU * HYDRAULIC_STEP + 0.1, nsteps)
                / 60
                / 60
            )

            for i, (node, ax) in enumerate(zip(plot_nodes, axs.flat)):
                node_idx = topology.node_name_list.index(node)
                ax.text(
                    0.07,
                    0.99,
                    str(i + 1),
                    fontdict={"weight": "bold", "fontsize": 13, "color": (0.89, 0.48, 0.15, 1.0)},
                    va="top",
                    ha="left",
                )
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(min(ts), max(ts))
                if i % 2 == 0:
                    ax.set_ylabel("Conc.")
                for name in subset_on_map:
                    ax.plot(
                        ts,
                        preds[subset_on_map.index(name)][node_idx, :nsteps],
                        color=MODEL_COLOR[name],
                        alpha=0.7 if "MeGA-MP" not in name else 1.0,
                        label=key_to_label(name, glossary=False),
                        zorder=ZORDER.index(name) if name in ZORDER else 10,
                        linewidth=1.5,
                    )
                ax.plot(
                    ts,
                    y_true[node_idx, :nsteps],
                    linestyle=":",
                    linewidth=2.5,
                    color="k",
                    label="Simulator",
                    zorder=11,
                )
            axs[-1][0].set_xlabel("Time [hours]")
            axs[-1][1].set_xlabel("Time [hours]")

            linehandles, linelabels = ax.get_legend_handles_labels()
            lax = fig.add_axes((0.995, 0.98, 0.0, 0.0))
            lax.axis("off")
            lax.legend(
                linehandles,
                linelabels,
                ncols=len(linelabels),
                loc="lower right",
                bbox_to_anchor=(1.01, 1.0),
            )
            fig.savefig(
                os.path.join(out_path, f"{out_prefix}{wds_name}_all_models.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

def run_metrics(model_names=None, exp_ids=None, out_prefix=''):
    if model_names is None:
        model_names = DEFAULT_EVAL_MODELS
    if len(model_names) == 0 or len(exp_ids) ==  0:
        return
    if exp_ids is None:
        exp_ids = list(EXP_SETUPS.keys())

    metric_names = [
        "MAE (mean)",
        "MAE (std)",
        "MSE (mean)",
        "MSE (std)",
        "Exp Var (mean)",
        "Exp Var (std)",
    ]
    
    metrics_index = pd.MultiIndex.from_tuples(product(model_names, metric_names))

    for exp_id in exp_ids:
        exp_config = deepcopy(EXP_SETUPS[exp_id])
        results_path = os.path.join(
            RESULTS_DIR, f"experiment_{exp_id}", model_names[0], "results"
        )
        out_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}")
        metrics_df = pd.DataFrame(columns=exp_config["test"].keys(), index=metrics_index)

        for wds_name in exp_config["test"].keys():
            with open(os.path.join(results_path, f"eval_{wds_name}.pkl"), "rb") as f:
                y_true = pickle.load(f)["ground_truths"]

            for model_name in model_names:
                eval_path = os.path.join(
                    RESULTS_DIR, f"experiment_{exp_id}", model_name, "results"
                )
                with open(os.path.join(eval_path, f"eval_{wds_name}.pkl"), "rb") as f:
                    pred = pickle.load(f)["predictions"].clamp(0, 1)
                    #preds.append(pred[:, : y_true.shape[1]])
            
            #for pred, name in zip(preds, model_names):
                mae = torch.nn.L1Loss(reduction="none")(pred, y_true)
                mse = torch.nn.MSELoss(reduction="none")(pred, y_true)
                exp_var = np.zeros(10)
                # explained_variance_score(
                #     np.nan_to_num(y_true) + np.random.normal(0, 1e-12, pred.shape),
                #     np.nan_to_num(pred) + np.random.normal(0, 1e-12, pred.shape),
                #     multioutput="raw_values",
                #     force_finite=True,
                # )
                metrics_df.loc[(model_name, "MAE (mean)"), wds_name] = mae.mean().item()
                metrics_df.loc[(model_name, "MAE (std)"), wds_name] = mae.std().item()
                metrics_df.loc[(model_name, "MSE (mean)"), wds_name] = mse.mean().item()
                metrics_df.loc[(model_name, "MSE (std)"), wds_name] = mse.std().item()
                metrics_df.loc[(model_name, "Exp Var (mean)"), wds_name] = exp_var.mean().item()
                metrics_df.loc[(model_name, "Exp Var (std)"), wds_name] = exp_var.std().item()

        metrics_df.to_csv(os.path.join(out_path, f"{out_prefix}metrics.csv"))
    
    latex_out = ""
    template = "{:.4f}{{\\scriptsize$\\pm {:.4f}$}}"
    for model_name in model_names:
        latex_out += f"& {model_name}"
        for exp_id in exp_ids:
            out_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}")
            metrics_df = pd.read_csv(os.path.join(out_path, f"{out_prefix}metrics.csv"), index_col=[0, 1])
            exp_config = deepcopy(EXP_SETUPS[exp_id])
            for wds_name in exp_config["test"].keys():
                latex_out += " & "
                latex_out += template.format(metrics_df.loc[(model_name, 'MAE (mean)'), wds_name], metrics_df.loc[(model_name, 'MAE (std)'), wds_name])
        latex_out += " \\\ \n"
    
    with open(os.path.join(RESULTS_DIR, f"{out_prefix}metrics_latex.txt"), "w") as f:
        latex_out = key_to_label(latex_out)
        f.write(latex_out)
    
    print(latex_out)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Optional: set these to select a subset of models for training/evaluation.
    # # Leave as None to use defaults defined above.
    # train_model_names = ["GPSConv", "MeGA-MP-learned"] # ["MeGA-MP+MLP", "NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "PDE-GNN"]  # e.g., ["MeGA-MP"]
    # eval_model_names = ["GPSConv", "MeGA-MP-learned"] # ["MeGA-MP+MLP", "NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "PDE-GNN"] # ["MeGA-MP+MLP", "PDE-GNN"]  # e.g., ["MeGA-MP", "PDE-GNN"]
    # results_model_names = ["MeGA-MP+MLP", "NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "PDE-GNN", "MeGA-MP-learned"] # ["MeGA-MP+MLP", "PDE-GNN"]  # e.g., ["MeGA-MP", "PDE-GNN"]

    # exp_ids = [2]

    # Optional: set these to select a subset of models for training/evaluation.
    # Leave as None to use defaults defined above.
    # train_model_names = [
    #     "MeGA-MP+MLP", 
    #     "NNConv", 
    #     "AntiSymmetricConv", 
    #     "PDE-GNN", 
    #     "GPSConv", 
    #     "MeGA-MP-learned"
    # ] # ["MeGA-MP+MLP", "GPSConv", "MeGA-MP-learned"] # ["MeGA-MP+MLP", "NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "PDE-GNN"]  # e.g., ["MeGA-MP"]
    # eval_model_names = [
    #     "MeGA-MP", 
    #     "MeGA-MP+MLP",
    #     "NNConv", 
    #     "AntiSymmetricConv",
    #     "AntiSymmetricConv-Dia",
    #     "PDE-GNN",
    #     "GPSConv",
    #     "MeGA-MP-learned"
    # ] #  ["MeGA-MP+MLP", "GPSConv", "MeGA-MP-learned"] # ["MeGA-MP+MLP", "NNConv", "AntiSymmetricConv", "AntiSymmetricConv-Dia", "PDE-GNN"] # ["MeGA-MP+MLP", "PDE-GNN"]  # e.g., ["MeGA-MP", "PDE-GNN"]
    # results_plots_model_names = [
    #     "MeGA-MP" "MeGA-MP+MLP" "NNConv" "PDE-GNN" "GPSConv" "AntiSymmetricConv"
    # ] # ["MeGA-MP+MLP", "PDE-GNN"]  # e.g., ["MeGA-MP", "PDE-GNN"]
    # results_model_names = [
    #     MeGA-MP MeGA-MP+MLP NNConv GPSConv AntiSymmetricConv AntiSymmetricConv-Dia MeGA-MP-learned
    # ]
    # exp_ids = [1, 2]
    # skip_if_exists = False
    
    args = argparse.ArgumentParser() # Dummy line to avoid "unused import" warning for argparse, which is used in imported modules.
    args.add_argument("--train_models", nargs="+", default=[])
    args.add_argument("--eval_models", nargs="+", default=[])
    args.add_argument("--results_models", nargs="+", default=[])
    args.add_argument("--results_plots_models", nargs="+", default=[])
    args.add_argument("--exp_ids", nargs="+", type=int, default=[])
    args.add_argument("--overwrite", action="store_true", default=False)
    args.add_argument("--out_prefix", type=str, default='')
    args = args.parse_args()
    
    run_training(model_names=args.train_models, exp_ids=args.exp_ids, device=device, overwrite=args.overwrite)
    run_evaluation(model_names=args.eval_models, exp_ids=args.exp_ids, device=device)
    
    # Keep plots/metrics on full results unless explicitly overridden.
    # For result artifacts, run:
    # python -u run_experiments_pde_gnn.py --results_models MeGA-MP MeGA-MP+MLP NNConv PDE-GNN GPSConv AntiSymmetricConv AntiSymmetricConv-Dia MeGA-MP-learned --exp_ids 1 2
    # python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP+MLP NNConv PDE-GNN GPSConv AntiSymmetricConv --results_models MeGA-MP MeGA-MP+MLP NNConv PDE-GNN GPSConv AntiSymmetricConv AntiSymmetricConv-Dia MeGA-MP-learned --exp_ids 1 2
    # python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP MeGA-MP+MLP MeGA-MP-learned --exp_ids 1 2


    # python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP NNConv PDE-GNN GPSConv AntiSymmetricConv --exp_ids 1
    # python -u run_experiments_pde_gnn.py --results_plots_models MeGA-MP+MLP NNConv PDE-GNN GPSConv AntiSymmetricConv --exp_ids 2

    run_plots(model_names=args.results_plots_models, exp_ids=args.exp_ids, out_prefix=args.out_prefix)
    run_metrics(model_names=args.results_models, exp_ids=args.exp_ids, out_prefix=args.out_prefix)


