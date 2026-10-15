# Script to generate data for all experiments in parallel on a slurm cluster. 
# Each job will generate a fraction of the data for each experiment, as specified 
# by the START_FRAC and FRAC variables.
# Use with the sbatch script generate_data.sh


import sys

import torch
import multiprocessing as mp
from copy import deepcopy
from experiment_shared import init_dataset
from run_experiments_pde_gnn import _prepare_dataset_split, TIME_CONFIG, DEFAULT_CT, DEFAULT_TAU, EXP_SETUPS
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader

try:
    mp.set_start_method("spawn", force=True)
    print("spawned")
except RuntimeError:
    pass

def generate(start_idx, step, exp_ids=None, device='cpu', wds=None, overwrite_data=False):
    if wds is None: wds = []

    for exp_id in exp_ids:
        exp_config = deepcopy(EXP_SETUPS[exp_id])
        dataset_cfg = _prepare_dataset_split(exp_config)
        for subset in ["train", "validation", "test"]:
            cfg = dataset_cfg[subset]
            if 'inp_file' in cfg: cfg = { 'training' : cfg }
            for wds_name, subcfg in cfg.items():
                if wds_name not in wds: continue
                ds = init_dataset(
                    subcfg, DEFAULT_CT, DEFAULT_TAU, device=device, 
                    time_config=TIME_CONFIG, overwrite=overwrite_data
                )
                print('Running for ', list(range(start_idx, ds.size, step)))
                ds = torch.utils.data.Subset(ds, range(start_idx, ds.size, step))
                data_loader = DataLoader(ds, batch_size=10)
                [ _ for _ in tqdm(data_loader, desc="Generating data ...") ]


if __name__ == "__main__":
    exp_ids = [1, 2]
    wds = ["L-Town"]

    generate(
        start_idx=int(sys.argv[1]), step=int(sys.argv[2]), exp_ids=exp_ids, 
        wds=wds, overwrite_data=True
    )