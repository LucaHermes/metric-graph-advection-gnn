from experiment_shared import train, init_dataset, init_model
from experiment_config_loader import get_tuning_config
from torch_geometric.loader import DataLoader
import optuna
import torch
import wandb
import json
import os
import sys
from copy import deepcopy
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
    print("spawned")
except RuntimeError:
    pass

TUNING_CONFIG = get_tuning_config()
TARGET = sys.argv[1] # 'GPSConv'
VERSION = TUNING_CONFIG.get("version", "1")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_config = {
    'data_splits' : deepcopy(TUNING_CONFIG.get('data_splits', {})),
    'dataset_config' : deepcopy(TUNING_CONFIG.get('dataset_config', {})),
}

def objective(trial):
    config = {
        'model_name' : TARGET,
        'cT' : 1,
        'Tau' : 700,
        'clip_grad_norm' : 1e10,
        'loss_fn' : 'l1',
        'epochs' : 60,
        'scheduler_patience' : None,
        'train_Tau' : 700,
        'eval_Tau' : 700,

        "hidden_dim" : trial.suggest_categorical("hidden_dim", [64, 96, 128, 160]),
        "num_layers" : trial.suggest_int("num_layers", 2, 6),

        "learning_rate" : trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay" : trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "batch_size" : trial.suggest_categorical("batch_size", [16, 32, 64]),
    }

    # Add model-specific parameters
    if TARGET == 'GPSConv':
        config["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
        config["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        config["attn_dropout"] = trial.suggest_float("attn_dropout", 0.0, 0.3)
        config["conv"] = trial.suggest_categorical("conv", ['nnconv', 'gine', 'gated_gnn'])
    elif TARGET == 'NNConv':
        config['aggr'] = 'sum'
    elif TARGET == 'PDE-GNN':
        config['n_edge_features'] = 3
        config['method'] = 'euler'
        config['eval_discrete'] = True
        config['tol_scale'] = 100
        config['input_dim'] = 1
        config['output_dim'] = 1
        config['add_source'] = False
        config['alpha_train'] = True
        config['alpha_sigmoid'] = False
        config["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        config['train_Tau'] = trial.suggest_categorical("train_Tau", [32, 100, 150])
        config['step_size'] = trial.suggest_categorical("step_size", [1./4., 1/3., 1/2., 1.])
        config['clip_grad_norm'] = trial.suggest_categorical("clip_grad_norm", [1e-4, 0.5, 1.0, 2.0, 1e10])
    elif TARGET == 'AntiSymmetricConv':
        config['phi'] = trial.suggest_categorical("phi", ['nnconv', 'gine', 'gated_gnn'])
        config['num_iters'] = trial.suggest_categorical("num_iters", [1, 2, 5, 10, 20, 50])
        config['epsilon'] = trial.suggest_categorical("epsilon", [1., 0.1, 0.01, 0.001])
        config['gamma'] = trial.suggest_categorical("gamma", [1., 0.1, 0.01, 0.001])
    elif TARGET == 'MeGA-MP-learned':
        config['layer'] = 'advect_react_learned'
        config['mixing_at_nodes'] = True
        del config['hidden_dim']
        config['track_active_edges'] = trial.suggest_categorical("track_active_edges", [True, False])
        config['flow_attribute'] = trial.suggest_categorical("flow_attribute", ['delays', 'flows'])
        config['num_iters'] = trial.suggest_categorical("num_iters", [0, 13, 20])
    else:
        raise NotImplementedError('A model called', TARGET, 'is not implemented.')
    
    cT, Tau = config['cT'], config['Tau']
    dataset_config = exp_config['dataset_config']
    dataset_config['idx_offset'] = 0
    dataset_config['size'] = exp_config['data_splits']['train']
    ds = init_dataset(dataset_config, cT, Tau, device=device)
    ds.add_positional_encoding = (config['model_name'] == 'GPSConv')
    data_loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    dataset_config['idx_offset'] = exp_config['data_splits']['train'] #+ exp_config['data_splits']['test']
    dataset_config['size'] = exp_config['data_splits']['val']
    val_ds = init_dataset(dataset_config, cT, Tau, device=device)
    val_ds.add_positional_encoding = (config['model_name'] == 'GPSConv')
    val_data_loader = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True, persistent_workers=True)

    model = init_model(config, device=device)
    last_validation_loss = {}

    def callback(val_loss, train_loss, l1_loss, grad_norm, epoch):
        trial.report(val_loss, epoch)
        
        last_validation_loss['val_loss'] = val_loss

        wandb.log({
            'val_loss' : val_loss,
            'train_loss' : train_loss,
            'l1_loss' : l1_loss,
            'epoch' : epoch,
            'grad_norm' : grad_norm,
        })

        if trial.should_prune():
            raise optuna.TrialPruned()


    with wandb.init(
        project='MeGA-MP',
        group=f'{TARGET}_{VERSION}',
        config=config
    ) as run:
        model, losses, val_losses = train(
            model, data_loader, config['learning_rate'], config['epochs'], 
            config['loss_fn'], cT, config['train_Tau'], 
            clip_grad_norm=config['clip_grad_norm'], weight_decay=config["weight_decay"],
            scheduler_patience=config['scheduler_patience'], 
            device=device, val_dataset=val_data_loader, 
            callback=callback, validate_every=1
        )
        
    return last_validation_loss['val_loss']


if __name__ == "__main__":
    study_name = f'{TARGET}_{VERSION}_Tune'
    storage_url = f"sqlite:///{os.getcwd()}/best_configs/optuna.db"
    server = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {
                "timeout": 60,
                "check_same_thread": False,
            },
            "pool_size": 10,
            "max_overflow": 0,
            "pool_pre_ping": True,
        }
    )
    if False:
        optuna.delete_study(study_name=study_name, storage=server)
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,      # don't prune until 5 trials are finished
        n_warmup_steps=5,        # don't prune within first X steps of a trial
        interval_steps=1         # check pruning every N steps
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=pruner,
        storage=server,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=40, n_jobs=1)

    print("Best Trial:", study.best_params)

    config_path = os.path.join('best_configs', f'{TARGET}_{VERSION}')
    os.makedirs(config_path, exist_ok=True)

    with open(os.path.join(config_path, 'config.json'), 'w') as f:
        json.dump(study.best_params, f)