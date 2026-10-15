import numpy as np
from run_experiments_pde_gnn import _get_model_config
import utils as functions
import simulate
import model.semi_lagrangian_model_wrapper as SLMPNN
import os
import pickle
from functools import partial
import time as timer
from copy import deepcopy
import modules
import argparse

from experiment_config_loader import get_pde_gnn_config, get_time_complexity_config
from experiment_shared import evaluate, init_model, resolve_time_config
from wds_experiment_utils import init_inp_file, key_to_label

PDE_GNN_CONFIG = get_pde_gnn_config()
PDE_TIME_CONFIG = resolve_time_config(PDE_GNN_CONFIG.get("time"))
EXP_SETUPS = {exp["id"]: exp for exp in PDE_GNN_CONFIG.get("experiments", [])}
CHECKPOINT_DIR = PDE_GNN_CONFIG.get("results_dir")
DEFAULT_CT = PDE_GNN_CONFIG.get("cT", 1)
EVAL_FILES_DIR = PDE_GNN_CONFIG.get("results_dir")

TIME_COMPLEXITY_CONFIG = get_time_complexity_config()
TIME_CONFIG = resolve_time_config(TIME_COMPLEXITY_CONFIG.get("time"))

RESULTS_DIR = TIME_COMPLEXITY_CONFIG.get("results_dir", "Results/performance")
HYDRAULIC_STEP = TIME_CONFIG["hydraulic_step"]
N_SECONDS = TIME_CONFIG["n_seconds"]
QUALITY_TIMESTEP = TIME_CONFIG["quality_timestep"]
PATTERN_STEP = TIME_CONFIG["pattern_step"]
PATTERN_N_WAVES = TIME_CONFIG["pattern_n_waves"]
SEED = TIME_CONFIG["seed"]
cT = TIME_COMPLEXITY_CONFIG.get("cT", 200)
Tau = TIME_COMPLEXITY_CONFIG.get("Tau", 700)
device = "cuda"

wds_path = TIME_COMPLEXITY_CONFIG.get("wds_path", "/tmp")
data_path = os.path.join(RESULTS_DIR, "data")

DEFAULT_MODEL_NAMES = [
    "NNConv",
    "PDE-GNN",
    "GPSConv",
    "AntiSymmetricConv",
    "AntiSymmetricConv-Dia",
    "MeGA-MP",
    "MeGA-MP-learned",
    "MeGA-MP+MLP",
]
DEFAULT_EXP_IDS = list(EXP_SETUPS.keys())
    
def run_experiments(exp_setup, result_file_out, data_file_out, n_repeats=1):
    inp_file_source = exp_setup['inp_file']
    msx_file = exp_setup['msx_file']
    remove_inline_nodes = exp_setup.get('remove_inline_nodes', False)
    n_clusters = exp_setup.get('n_clusters')
    sources_at = [str(n_clusters), str(n_clusters + 1)] if n_clusters is not None else ['R1', 'R2']
    
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(wds_path, exist_ok=True)
    
    inp_file_name = os.path.basename(inp_file_source).rstrip('.inp')
    inp_file_out = os.path.join(wds_path, f'{inp_file_name}_proc.inp')
    
    init_inp_file(inp_file_source, inp_file_out, sources_at, remove_inline_nodes, n_clusters)
    
    pattern_length = N_SECONDS // HYDRAULIC_STEP
    topology = functions.read_inp(inp_file_out)
    sources_at = [ n for n, ts in topology.query_node_attribute('demand_timeseries_list').items() if ts[0].base_value < 0 ]
    pattern = functions.create_wavy_pattern(pattern_length, PATTERN_N_WAVES, HYDRAULIC_STEP, SEED)
    
    # Run EPANET to generate the flow field and ground truth data
    if os.path.exists(data_file_out):
        with open(data_file_out, 'rb') as f:
            graph_data, topo = pickle.load(f)
    else:
        # Run this one without reactions to get the hydraulics
        graph_data = simulate.inp_to_graph_data(
            inp_file_out, pattern, sources_at, N_SECONDS, HYDRAULIC_STEP, QUALITY_TIMESTEP, out_file=data_file_out,
            sim_setup_fns=[ partial(functions.set_sim_demands, multiplier=.5) ], f_msx_in=None
        )
        topo = graph_data['topology']

    model_config = {
        'model_name' : 'MeGA-MP',
        'interpolation_mode' : 'bilinear',
        'layer' : 'advect',
        'mixing_at_nodes' : True,
        'max_msg_passing_rounds' : 1000,
        'mask_temp' : 1e-8
    }
    mega_mp = init_model(model_config)

    # Run MeGA-MP several times
    megamp_runtimes = []
    for i in range(n_repeats):
        t0 = timer.time()
        SLMPNN.semi_lagrangian_mpnn(
            graph_data['epanet_result'][:,:cT], graph_data['flow_field'], graph_data['edge_index'], graph_data['edge_lengths'], 
            output_times=[Tau * HYDRAULIC_STEP], dt=HYDRAULIC_STEP, nsteps=Tau, edge_capacities=graph_data['edge_diameter'],
            control_indices=graph_data['boundary_index'], control_inputs=graph_data['boundary_values'],
            interpolation='nearest', max_msg_passing_rounds=1000, model=mega_mp,
        )
        megamp_runtimes.append(timer.time() - t0)
        print('MeGA-MP ran for', megamp_runtimes)

    # Run EPANET-MSX several times
    epanet_runtimes = []
    epanet_with_msx_runtimes = []
    for i in range(n_repeats):
        # Since our method predicts 500 time steps, we also let EPANET predict 500 time steps
        n_seconds = Tau * HYDRAULIC_STEP

        graph_data = simulate.inp_to_graph_data(
            inp_file_out, pattern, sources_at, n_seconds, HYDRAULIC_STEP, QUALITY_TIMESTEP, out_file=data_file_out,
            sim_setup_fns=[ partial(functions.set_sim_demands, multiplier=.5) ], f_msx_in=None
        )
        print('R', graph_data['epanet_result'].shape, graph_data['epanet_result'].mean())
        epanet_runtimes.append(graph_data['runtime'])#timer.time() - t0)
        print('EPANET ran for', epanet_runtimes)
        graph_data = simulate.inp_to_graph_data(
            inp_file_out, pattern, sources_at, n_seconds, HYDRAULIC_STEP, QUALITY_TIMESTEP, out_file=data_file_out,
            sim_setup_fns=[ partial(functions.set_sim_demands, multiplier=.5) ], f_msx_in=msx_file
        )
        epanet_with_msx_runtimes.append(graph_data['runtime'])
        print('R', graph_data['epanet_result'].shape, graph_data['epanet_result'].mean())
        print('EPANET-MSX ran for', epanet_with_msx_runtimes)

    print(megamp_runtimes)
    print(epanet_runtimes)
    print(epanet_with_msx_runtimes)
    # results = {
    #     'pred' : pred,
    #     'edge_passes' : edge_passes,
    #     'agg_time' : agg_time,
    #     'aggs_all' : aggs_all,
    #     'runtime' : runtime
    # }
    
    # with open(result_file_out, 'wb') as f:
    #     graph_data.pop('res')
    #     pickle.dump((results, graph_data, topology), f)


def _configure_antisymmetric_dia(model_config, wds_name):
    diameter = 13 if wds_name == "Hanoi" else 79
    model_config = deepcopy(model_config)
    model_config["epsilon"] = model_config["epsilon"] * model_config["num_iters"] / diameter
    model_config["num_iters"] = diameter
    return model_config


def _precompile_backtracing():
    modules.compute_backward_transit_times_fast(
        np.ones([1]), np.ones([1, 1]), 0.1
    )


def run_megamp_vs_epanet(exp_ids=None, n_repeats=1):
    if exp_ids is None:
        exp_ids = DEFAULT_EXP_IDS

    for exp_id in exp_ids:
        exp_setup = EXP_SETUPS[exp_id]
        inp_file_name = os.path.basename(exp_setup["inp_file"]).rstrip(".inp")
        result_file_out = os.path.join(data_path, f"{inp_file_name}_mpnn_{exp_id}.pkl")
        data_file_out = os.path.join(data_path, f"{inp_file_name}_epanet_{exp_id}")
        run_experiments(exp_setup, result_file_out, data_file_out, n_repeats=n_repeats)


def run_runtime_evaluation(model_names=None, exp_ids=None):
    if model_names is None:
        model_names = DEFAULT_MODEL_NAMES
    if exp_ids is None:
        exp_ids = DEFAULT_EXP_IDS

    for exp_id in exp_ids:
        if exp_id == 0: continue  # Skip the first experiment
        exp_config = deepcopy(EXP_SETUPS[exp_id])
        for wds_name, eval_config in exp_config["test"].items():
            eval_config["size"] = exp_config["data_splits"]["test"]
            eval_config["idx_offset"] = exp_config["data_splits"]["train"]
            #eval_config["data_path"] = data_path

            for model_name in model_names:
                model_path = os.path.join(CHECKPOINT_DIR, f"experiment_{exp_id}", model_name)
                model_path = model_path.replace("AntiSymmetricConv-Dia", "AntiSymmetricConv")
                results_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", model_name, "results")
                base_config = _get_model_config(
                    "AntiSymmetricConv" if model_name == "AntiSymmetricConv-Dia" else model_name,
                    device,
                )
    
                out_file = os.path.join(results_path, f"eval_{wds_name}.pkl")
                if os.path.exists(out_file):
                    print(f"Evaluation for model {model_name} on {wds_name} already exists, skipping.")
                    continue
                
                os.makedirs(results_path, exist_ok=True)
                    
                model_config = base_config
                if model_name == "AntiSymmetricConv-Dia":
                    model_config = _configure_antisymmetric_dia(base_config, wds_name)
                
                # eval_config['size'] = 10
                
                result = evaluate(
                    DEFAULT_CT,
                    model_config["eval_Tau"],
                    eval_config,
                    model_config,
                    model_path,
                    batch_size=1,
                    device=device,
                    time_config=PDE_TIME_CONFIG,
                )
                result["key"] = wds_name

                with open(out_file, "wb") as f:
                    pickle.dump(result, f)


def run_runtime_summary(model_names=None, exp_ids=None):
    if model_names is None:
        model_names = DEFAULT_MODEL_NAMES
    if exp_ids is None:
        exp_ids = DEFAULT_EXP_IDS

    template = "{:.4f}{{\\scriptsize$\\pm {:.4f}$}}"
    
    latex_out = ''
    
    for model_name in model_names:
        latex_out += f"& {model_name}"
        
        for exp_id in exp_ids:
            if exp_id == 0: continue  # Skip the first experiment
            # out_str += f"Experiment {exp_id}\n"
            # out_str += "-" * 80 + "\n"
            exp_config = EXP_SETUPS[exp_id]
            for wds_name, _ in exp_config["test"].items():
                # out_str += f"{wds_name}\n"

                results_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", model_name, "results")
                pkl_file = os.path.join(results_path, f"eval_{wds_name}.pkl")
                
                with open(pkl_file, "rb") as f:
                    result = pickle.load(f)
                
                model_rt_mean = result["model_runtimes"].mean().cpu().numpy()
                model_rt_std = result["model_runtimes"].std().cpu().numpy()
                
                latex_out += " & " + template.format(model_rt_mean, model_rt_std)
                # out_str += '{} {} {} {} \n'.format(
                #     f"{model_name:<24}",
                #     template.format(model_rt_mean, model_rt_std),
                #     template.format(result["preprocessing_times"].mean(), result["preprocessing_times"].std()),
                #     template.format(result["model_runtimes"].mean(), result["model_runtimes"].std())
                # )

            # epanet_rt_mean = (result["epanet_runtimes"]).mean().cpu().numpy().round(3)
            # epanet_rt_std = (result["epanet_runtimes"]).std().cpu().numpy().round(3)
            # out_str += f'{"EPANET":<24}' + template.format(epanet_rt_mean, epanet_rt_std) + "\n"
            # out_str += "-" * 80 + "\n"
        latex_out += " \\\ \cdashline{2-6} \n"
    
    for label, key in zip(["preprocessing", "EPANET"], ["preprocessing_times", "epanet_runtimes"]):
        latex_out += f"& \\textit{{{label}}}"
        for exp_id in exp_ids:
            if exp_id == 0: continue  # Skip the first experiment
            exp_config = EXP_SETUPS[exp_id]
            for wds_name, _ in exp_config["test"].items():
                results_path = os.path.join(RESULTS_DIR, f"experiment_{exp_id}", "MeGA-MP", "results")
                pkl_file = os.path.join(results_path, f"eval_{wds_name}.pkl")
                with open(pkl_file, "rb") as f:
                    result = pickle.load(f)
                epanet_rt_mean = result[key].mean().cpu().numpy()
                epanet_rt_std = result[key].std().cpu().numpy()
                latex_out += " & " + template.format(epanet_rt_mean, epanet_rt_std)
        latex_out += " \\\ \n"
    #out_str += "-" * 80 + "\n"
    latex_out = key_to_label(latex_out)
    
    with open(os.path.join(RESULTS_DIR, "runtime_summary.txt"), "w") as f:
        f.write(latex_out)
        
    print(latex_out)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run time complexity evaluations.")
    parser.add_argument(
        "--mode",
        choices=["megamp", "test", "summary", "all"],
        default="all",
        help="Select which action to run.",
    )
    parser.add_argument(
        "--exp-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of experiment IDs to run.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model names to evaluate/summarize.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats for MeGA-MP vs EPANET runs.",
    )
    parser.add_argument(
        "--skip-precompile",
        action="store_true",
        help="Skip precompiling backtracing kernels.",
    )
    return parser.parse_args()

# def evaluate_runtime(cT, Tau, dataset_config, model_config, model_path):
#     model = init_model(model_config)

#     if not (model_config['model_name'] == 'MeGA-MP' and model_config['layer'] == 'advect'):
#         model_file = os.path.join(model_path, 'model.pt')
#         model.load_state_dict(torch.load(model_file, weights_only=True))
    
#     # For the runtime analysis, epanet results should not be stored as files
#     # Results have to be recomputed to measure runtime
#     #with TemporaryDirectory() as datadir:
#     #    dataset_config['data_path'] = datadir.name
#     #    assert os.path.isdir(datadir.name)
    
#     ds = init_dataset(dataset_config, cT, Tau)
#     data_loader = DataLoader(ds, batch_size=batch_size)
#     Tau = model_config['eval_Tau']

#     predictions = []
#     ground_truths = []
#     flow_fields = []
#     edge_lengths = []
#     edge_diameters = []
    
#     with torch.no_grad():
#         for model_inputs in tqdm(data_loader):
#             model_inputs = prepare_features(model_inputs)
#             pred, *_ = model(cT=cT, Tau=Tau, n_steps=1, **model_inputs)
#             pred = pred.squeeze(-1)
#             # boundary nodes will not be predicted, values assumed known
#             # mask boundary nodes
#             mask = ~torch.eye(len(pred), device=pred.device)[model_inputs.boundary_index].sum(0).bool()
#             pred[mask][:,:] = model_inputs['y'][mask] # set values at dirichlet boundaries
#             predictions.append(pred.detach().cpu())
#             ground_truths.append(model_inputs['y'].cpu())
#             flow_fields.append(model_inputs['flow_field'].cpu())
#             edge_lengths.append(model_inputs['edge_lengths'].cpu())
#             edge_diameters.append(model_inputs['edge_diameter'].cpu())
            
#     predictions = torch.cat(predictions, dim=0)
#     ground_truths = torch.cat(ground_truths, dim=0)
if __name__ == "__main__":
    args = _parse_args()
    if not args.skip_precompile:
        _precompile_backtracing()
    print(args.mode)
    
    if args.mode == "megamp":
        run_megamp_vs_epanet(exp_ids=args.exp_ids, n_repeats=args.repeats)
    elif args.mode == "test":
        run_runtime_evaluation(model_names=args.models, exp_ids=args.exp_ids)
    elif args.mode == "summary":
        run_runtime_summary(model_names=args.models, exp_ids=args.exp_ids)
    else:
        run_runtime_evaluation(model_names=args.models, exp_ids=args.exp_ids)
        run_runtime_summary(model_names=args.models, exp_ids=args.exp_ids)
    # wds = 'ltown'
    # inp_file_source = 'networks/l_town_no_tanks.inp'
    # inp_file_name = os.path.basename(wds).rstrip('.inp')
    # result_file_out = os.path.join(data_path, f'{inp_file_name}_mpnn_' + '{}.pkl')
    # data_file_out = os.path.join(data_path, f'{inp_file_name}_epanet_' + '{}')

    # for i, exp_setup in EXP_SETUPS.items():
    #     result_file = result_file_out.format(i)
    #     exp_setup['n_clusters'] = 200
    #     run_experiments(exp_setup, result_file, data_file_out.format(i))
    


        # with open(result_file, 'rb') as f:
        #     results, graph_data, topology = pickle.load(f)
        
        # print(results['runtime'])
        # print(graph_data['runtime'])