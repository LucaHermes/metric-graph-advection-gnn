from wds_experiment_utils import plot_on_map, plot_with_map
from experiment_config_loader import get_pde_gnn_config
from experiment_shared import evaluate, resolve_time_config, run_experiment
from experiments.experiment_configs import CONFIGS
from sklearn.metrics import explained_variance_score
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import pickle
from utils import MODEL_COLOR

RESULTS_DIR = 'Results/ablation'

PDE_GNN_CONFIG = get_pde_gnn_config()
TIME_CONFIG = resolve_time_config(PDE_GNN_CONFIG.get("time"))
EXP_SETUPS = {exp["id"]: exp for exp in PDE_GNN_CONFIG.get("experiments", [])}
DEFAULT_CT = PDE_GNN_CONFIG.get("cT", 1)
DEFAULT_TAU = PDE_GNN_CONFIG.get("Tau", 700)

DEFAULT_TRAIN_MODELS = ["MeGA-MP+MLP", "MeGA-MP"]
DEFAULT_EVAL_MODELS = ["MeGA-MP+MLP", "MeGA-MP"]


def _get_model_config(model_name, device):
    config_key = "MeGA-MP" if model_name.startswith("MeGA-MP") else model_name
    model_config = deepcopy(CONFIGS[config_key])
    model_config["device"] = device
    model_config["layer"] = "advect_react"
    if model_name == "MeGA-MP":
        model_config["layer"] = "advect"
    return model_config


def _prepare_dataset_split(exp_config, split_name, idx_offset):
    dataset_cfg = deepcopy(exp_config[split_name])
    dataset_cfg["size"] = exp_config["data_splits"][split_name]
    dataset_cfg["idx_offset"] = idx_offset
    return { 'train': dataset_cfg }


def run_training(model_names=None, exp_ids=None, device=None):
    if model_names is None:
        model_names = DEFAULT_TRAIN_MODELS
    if exp_ids is None:
        exp_ids = list(EXP_SETUPS.keys())
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_names:
        model_config = _get_model_config(model_name, device)
        for exp_id in exp_ids:
            exp_config = deepcopy(EXP_SETUPS[exp_id])
            model_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", model_name)
            os.makedirs(model_path, exist_ok=True)

            train_cfg = _prepare_dataset_split(exp_config, "train", idx_offset=0)
            model_file = os.path.join(model_path, "model.pt")
            if model_name == "MeGA-MP":
                model_config["layer"] = "advect"
            if not os.path.exists(model_file):
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
        model_config = _get_model_config(model_name, device)
        for exp_id in exp_ids:
            exp_config = deepcopy(EXP_SETUPS[exp_id])
            model_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", model_name)
            results_path = os.path.join(model_path, "results")
            os.makedirs(results_path, exist_ok=True)
            
            idx_offset = exp_config["data_splits"]["train"] + exp_config['data_splits']['validation']
            for wds_name, eval_cfg in exp_config["test"].items():
                eval_cfg = deepcopy(eval_cfg)
                eval_cfg["size"] = exp_config["data_splits"]["test"]
                eval_cfg["idx_offset"] = idx_offset
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
                with open(os.path.join(results_path, f"eval_{wds_name}.pkl"), "wb") as f:
                    pickle.dump(result, f)


def run_metrics_and_plots(model_names=None, exp_ids=None):
    if model_names is None:
        model_names = DEFAULT_EVAL_MODELS
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

    sample_idx = 0
    metrics_index = pd.MultiIndex.from_tuples(product(model_names, metric_names, EXP_SETUPS[0]["test"].keys()))

    for exp_id in exp_ids:
        exp_config = deepcopy(EXP_SETUPS[exp_id])
        results_path = os.path.join(
            RESULTS_DIR, f"experiment_{exp_id}", model_names[0], "results"
        )
        out_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}")
        metrics_df = pd.DataFrame(columns=exp_config["test"].keys(), index=metrics_index)

        for wds_name, _eval_config in exp_config["test"].items():
            with open(os.path.join(results_path, f"eval_{wds_name}.pkl"), "rb") as f:
                result = pickle.load(f)

            topology = result["topology"]
            n_nodes = topology.num_nodes
            batch_slice = slice(n_nodes * sample_idx, n_nodes * (sample_idx + 1))
            y_true = result["ground_truths"][batch_slice, :]
            flow_fields = result["flow_fields"]
            sources_at = result["sources_at"]
            preds = []

            for model_name in model_names:
                eval_path = os.path.join(
                    RESULTS_DIR, f"experiment_{exp_id}", model_name, "results"
                )
                with open(os.path.join(eval_path, f"eval_{wds_name}.pkl"), "rb") as f:
                    pred = pickle.load(f)["predictions"][batch_slice]
                    preds.append(pred[:, : y_true.shape[1]])

            fig, _ax = plot_on_map(
                [*preds, y_true],
                topology,
                sources_at=sources_at,
                plt_kwargs=[
                    *[
                        dict(linestyle="-", linewidth=2, color=f"C{midx}", label=m)
                        for midx, m in enumerate(model_names)
                    ],
                    dict(linestyle=":", linewidth=2, color="k", label="Ground Truth"),
                ],
                cT=DEFAULT_CT,
                Tau=DEFAULT_TAU,
                time_config=TIME_CONFIG,
            )
            fig.savefig(
                os.path.join(out_path, f"{wds_name}_advection_reaction_on_map.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            fig, _ax = plot_with_map(
                [*preds, y_true],
                flow_fields[: topology.num_links].cpu(),
                topology,
                sources_at=sources_at,
                plt_kwargs=[
                    *[
                        dict(
                            linestyle="-",
                            linewidth=2,
                            color=MODEL_COLOR[m],
                            label=m,
                        )
                        for midx, m in enumerate(model_names)
                    ],
                    dict(linestyle=":", linewidth=2, color="k", label="Ground Truth"),
                ],
                cT=DEFAULT_CT,
                Tau=DEFAULT_TAU,
                time_config=TIME_CONFIG,
            )
            fig.savefig(
                os.path.join(out_path, f"{wds_name}_advection_reaction_with_map.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            for pred, model_name in zip(preds, model_names):
                mae = torch.nn.L1Loss(reduction="none")(pred, y_true)
                mse = torch.nn.MSELoss(reduction="none")(pred, y_true)
                exp_var = explained_variance_score(
                    np.nan_to_num(y_true),
                    np.nan_to_num(pred),
                    multioutput="raw_values",
                    force_finite=False,
                )
                metrics_df.loc[model_name, "MAE (mean)", wds_name] = mae.mean().item()
                metrics_df.loc[model_name, "MAE (std)", wds_name] = mae.std().item()
                metrics_df.loc[model_name, "MSE (mean)", wds_name] = mse.mean().item()
                metrics_df.loc[model_name, "MSE (std)", wds_name] = mse.std().item()
                metrics_df.loc[model_name, "Exp Var (mean)", wds_name] = exp_var.mean().item()
                metrics_df.loc[model_name, "Exp Var (std)", wds_name] = exp_var.std().item()

        metrics_df.to_csv(os.path.join(out_path, "metrics.csv"))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional: set these to select a subset of models for training/evaluation.
    train_model_names = []  # e.g., ["MeGA-MP"]
    eval_model_names = ["MeGA-MP-learned", "MeGA-MP+MLP", "MeGA-MP"]  # e.g., ["MeGA-MP+MLP", "MeGA-MP"]

    run_training(model_names=train_model_names, device=device)
    run_evaluation(model_names=eval_model_names, device=device)
    run_metrics_and_plots()


# if __name__ == '__main__':
#     model_names = ['NNConv', 'PDE-GNN', 'MeGA-MP']
#     model_names = ['MeGA-MP']

#     # 1. Run MeGA-MP without learnable component on a purely advective system
#     # 2. Run MeGA-MP with learnable component on a purely advective system
#     # 3. Run MeGA-MP without learnable component on a advective-reactive system
#     # 4. Run MeGA-MP with learnable component on a purely advective-reactive system

#     # Important: Precompile njit function
#     modules.compute_backward_transit_times_fast(
#         np.ones([1]), np.ones([1, 1]), 0.1
#     )
        
#     for exp_id, exp_config in EXP_SETUPS.items():

#         for wds_name, eval_config in exp_config['eval'].items():
#             eval_config['size'] = exp_config['data_splits']['test']
#             eval_config['idx_offset'] = exp_config['data_splits']['train']

#             # Always run the EPANET simulation instead of loading cached files to get runtime of epanet
#             #with TemporaryDirectory() as temp_dir:
#             #eval_config['data_path'] = getattr(temp_dir, 'name', temp_dir)
#             eval_config['data_path'] = data_path

#             for model_name in model_names:
#                 #continue
#                 model_path = os.path.join(CHECKPOINT_DIR, f'experiment_{exp_id}', model_name)
#                 results_path = os.path.join(RESULTS_DIR, f'experiment_{exp_id}', model_name, 'results')
#                 os.makedirs(results_path, exist_ok=True)
#                 model_config = CONFIGS[model_name]
#                 model_config['device'] = device
#                 model_config['layer'] = 'advect_react' if exp_id != 1 else 'advect'
                
#                 # Set batch size to 1 so we get per-sample runtimes
#                 result = evaluate(1, model_config['eval_Tau'], eval_config, model_config, model_path, batch_size=1, device=device)
#                 result['key'] = wds_name

#                 print(f'{model_name}\t| '
#                         f'{(result["model_runtimes"] + result["preprocessing_times"].cpu()).mean().numpy()}\t| '
#                         f'{result["epanet_runtimes"].mean().cpu().numpy()}')

#                 with open(os.path.join(results_path, f'eval_{wds_name}.pkl'), 'wb') as f:
#                     pickle.dump(result, f)
    
#     template = '{:.3f}{{\scriptsize$\pm {:.3f}$}}'

#     for exp_id, exp_config in EXP_SETUPS.items():

#         for wds_name, _ in exp_config['eval'].items():
#             print(wds_name)
#             for model_name in ['NNConv', 'PDE-GNN', 'MeGA-MP']:
#                 results_path = os.path.join(RESULTS_DIR, f'experiment_{exp_id}', model_name, 'results')
#                 pkl_file = os.path.join(results_path, f'eval_{wds_name}.pkl')
#                 with open(pkl_file, 'rb') as f:
#                     result = pickle.load(f)
#                 model_rt_mean = (result['model_runtimes'] + result['preprocessing_times'].cpu()).mean().numpy()
#                 model_rt_std = (result['model_runtimes'] + result['preprocessing_times'].cpu()).std().numpy()
#                 print(f'{model_name:10}', 
#                     template.format(model_rt_mean, model_rt_std), 
#                     template.format(result['preprocessing_times'].mean(), result['preprocessing_times'].std()), 
#                     template.format(result['model_runtimes'].mean(), result['model_runtimes'].std())
#                 )
            
#             epanet_rt_mean = (result['epanet_runtimes']).mean().cpu().numpy().round(3)
#             epanet_rt_std = (result['epanet_runtimes']).std().cpu().numpy().round(3)
#             print('EPANET:    ' + template.format(epanet_rt_mean, epanet_rt_std))
#             print('-'*80)
#         print('-'*80)