import os
import time as timer

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from epyt.epanet import ToolkitConstants
from epyt_flow import utils as epyt_utils

import experiments.random_hydraulics_dataset
import utils as functions
import modules
from baselines.pde_gnn import PDEFunction, PDEModelInputWrapper
from baselines.nn_conv import NNConvWrapper
from baselines.gps_conv import GPSConvWrapper
from baselines.anti_symmetric_conv import AntiSymmetricConvWrapper
from model.advection_reaction_layer import (
    AdvectionReactionLayer,
    AdvectionReactionLayerFullyLearned,
)
from model.advection_layer import AdvectionLayer
from model.advection_model_mp import AdvectionModelMP
from torchmetrics.aggregation import MeanMetric

DEFAULT_TIME_CONFIG = {
    "n_seconds": epyt_utils.to_seconds(hours=12),
    "hydraulic_step": 60,
    "quality_timestep": 1,
    "pattern_step": 60,
    "pattern_n_waves": 3,
    "seed": 42,
}

DEFAULT_DIAMETER_RANGE = (60, 250)  # [mm] -> 6 - 25 [cm]
DEFAULT_LENGTH_RANGE = (0.1, 80)  # [m]
DEFAULT_NORMALIZED_DEMAND_RANGE = (0.0, 7.0)  # [m^3/h]


def resolve_time_config(overrides=None):
    resolved = dict(DEFAULT_TIME_CONFIG)
    if overrides:
        resolved.update(overrides)
    return resolved


def safe_torch_load(path, map_location=None, **kwargs):
    if map_location is None and not torch.cuda.is_available():
        map_location = "cpu"
    if map_location is None:
        return torch.load(path, **kwargs)
    return torch.load(path, map_location=map_location, **kwargs)


def init_dataset(
    dataset_config,
    cT,
    Tau,
    device=None,
    time_config=None,
    **kwargs
):
    time_config = resolve_time_config(time_config)

    inp_file = dataset_config["inp_file"]
    msx_file = dataset_config["msx_file"]
    sources_at = dataset_config["sources_at"]
    idx_offset = dataset_config["idx_offset"]
    equal_demand_at_nodes = dataset_config["equal_demand_at_nodes"]
    size = dataset_config["size"]

    diameter_range = dataset_config.get("diameter_range", DEFAULT_DIAMETER_RANGE)
    length_range = dataset_config.get("length_range", DEFAULT_LENGTH_RANGE)
    normalized_demand_range = dataset_config.get("normalized_demand_range", DEFAULT_NORMALIZED_DEMAND_RANGE)

    pattern = None # Pattern is randomly generated

    ds = experiments.random_hydraulics_dataset.RandomHydraulicsDataset(
        inp_file,
        time_config["n_seconds"],
        cT,
        Tau,
        time_config["hydraulic_step"],
        time_config["quality_timestep"],
        time_config["pattern_step"],
        diameter_range,
        length_range,
        normalized_demand_range,
        pattern,
        sources_at,
        f_msx_in=msx_file,
        equal_demand_at_nodes=equal_demand_at_nodes,
        source_type=ToolkitConstants.EN_SETPOINT,
        data_path=dataset_config["data_path"],
        **kwargs
    )
    ds = experiments.random_hydraulics_dataset.TorchRandomHydraulicsDataset(
        size, ds, device=device, idx_offset=idx_offset
    )

    return ds


def init_model(config, device=None):
    model_name = config["model_name"]
    if model_name == "PDE-GNN":
        model = PDEModelInputWrapper(
            PDEFunction(1, 1, config, device),
            config,
        ).to(device)
    elif model_name == "NNConv":
        model = NNConvWrapper(
            config["cT"],
            config["Tau"],
            config["hidden_dim"],
            config["num_layers"],
            config["aggr"],
        ).to(device)
    elif model_name == "GPSConv":
        model = GPSConvWrapper(
            config["cT"],
            config["Tau"],
            config["hidden_dim"],
            config["num_layers"],
            pe_dim=8,
            conv=config.get("conv", "nnconv"),
            heads=config.get("num_heads", 1),
            dropout=config.get("dropout", 0.0),
            attn_dropout=config.get("attn_dropout", 0.0),
        ).to(device)
    elif model_name == "AntiSymmetricConv":
        model = AntiSymmetricConvWrapper(
            config["cT"],
            config["Tau"],
            config["hidden_dim"],
            config["phi"],
            num_iters=config["num_iters"],
            epsilon=config["epsilon"],
            gamma=config["gamma"],
        ).to(device)
    elif model_name.startswith("MeGA-MP"):
        advection_op = modules.AdvectionModuleGridSampleDynamic(
            interpolation_mode=config.get("interpolation_mode")
        )
        if config["layer"] == "advect_react":
            layer = AdvectionReactionLayer(
                advection_op, mixing_at_nodes=config["mixing_at_nodes"]
            )
        elif config["layer"] == "advect":
            layer = AdvectionLayer(
                advection_op, mixing_at_nodes=config["mixing_at_nodes"]
            )
        elif config["layer"] == "advect_react_learned":
            config["hidden_dim"] = config["cT"] + config["Tau"]
            layer = AdvectionReactionLayerFullyLearned(
                config["hidden_dim"],
                config["flow_attribute"],
                mixing_at_nodes=config["mixing_at_nodes"],
            )
            if config["track_active_edges"]:
                config["max_msg_passing_rounds"] = 1000
            if config["num_iters"] is not None and config["num_iters"] > 0:
                config["max_msg_passing_rounds"] = config["num_iters"]
            elif not config["track_active_edges"]:
                config["max_msg_passing_rounds"] = 1
        else:
            raise NotImplementedError(
                f"Layer {config['layer']} not implemented."
            )
        model = AdvectionModelMP(
            layer,
            max_msg_passing_rounds=config["max_msg_passing_rounds"],
            track_active_edges=config.get("track_active_edges", True),
            progress=False,
        ).to(device)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    return model


def prepare_features(model_inputs, device):
    model_inputs = model_inputs.to(device, non_blocking=True)
    model_inputs["edge_diameter_scaled"] = (
        model_inputs["edge_diameter"] / model_inputs["max_diameter"][0]
    )
    model_inputs["delay_steps_scaled"] = (
        model_inputs["delay_steps"] / model_inputs["max_delay_steps"][0]
    )
    model_inputs["edge_capacity_scaled"] = (
        model_inputs["edge_capacity"] / model_inputs["max_edge_capacity"][0]
    )
    model_inputs["flows_scaled"] = model_inputs["flows"] / model_inputs["max_flow"][0]
    model_inputs["x"] = model_inputs["x"].unsqueeze(-1)
    model_inputs["boundary_values"] = model_inputs["boundary_values"].unsqueeze(-1)
    return model_inputs

def train(
    model,
    dataset,
    learning_rate,
    epochs,
    loss_fn,
    cT,
    Tau,
    scheduler_factor=0.2,
    scheduler_patience=None,
    clip_grad_norm=0.0001,
    weight_decay=0,
    device=None,
    val_dataset=None,
    callback=None,
    eval_Tau=700,
    time_config=None,
    early_stopping_patience=30,
    early_stopping_warmup=60,
    validate_every=10
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler_patience = epochs if scheduler_patience is None else scheduler_patience
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience, threshold=1e-5
    )

    _losses, _val_losses, best_val_loss = [], [], torch.inf
    
    progress = tqdm(range(epochs))

    model.to(device)

    for epoch in progress:
        epoch_loss = 0
        epoch_l1 = 0
        grad_norms = 0

        for model_inputs in dataset:
            optimizer.zero_grad()
            model_inputs = prepare_features(model_inputs, dataset.dataset.device)
            pred, *_ = model(cT=cT, Tau=Tau, n_steps=1, **model_inputs)
            pred = pred.squeeze(-1)
            y_true = model_inputs["y"][:, : pred.shape[-1]]

            # boundary nodes will not be predicted, values assumed known
            # mask boundary nodes
            mask = (
                ~torch.eye(len(pred), device=pred.device)[
                    model_inputs.boundary_index
                ]
                .sum(0)
                .bool()
            )
            pred = pred[mask]
            y_true = y_true[mask]

            l1_loss = torch.nn.L1Loss()(pred[:, cT:], y_true[:, cT:])

            if loss_fn == "l1":
                loss = l1_loss
            elif loss_fn == "mape":
                loss = (
                    (pred[:, cT:] - y_true[:, cT:]) / (1e-3 + y_true[:, cT:])
                ).abs().mean()
            elif loss_fn == "huber":
                loss = torch.nn.HuberLoss(delta=0.3)(pred[:, cT:], y_true[:, cT:])
            elif loss_fn == "mse":
                loss = torch.nn.MSELoss()(pred[:, cT:], y_true[:, cT:])
            elif loss_fn == "relative_l1":
                loss = torch.norm(pred[:, cT:] - y_true[:, cT:], dim=0) / (
                    torch.norm(y_true[:, cT:], dim=0) + 1e-5
                )
                loss = loss.mean()
            elif loss_fn == "relative_l2":
                loss = torch.norm(pred[:, cT:] - y_true[:, cT:], dim=0) ** 2 / (
                    torch.norm(y_true[:, cT:], dim=0) ** 2 + 1e-5
                )
                loss = loss.mean()
            else:
                raise NotImplementedError(
                    f"A loss function with name {loss_fn} is not implemented."
                )

            loss.backward()
            epoch_loss += loss.detach()
            epoch_l1 += l1_loss.detach()
            grad_norms += torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )

            optimizer.step()
            _losses.append((l1_loss.detach().mean().cpu(), loss.detach().mean().cpu()))

        if val_dataset is not None and epoch % validate_every == 0:
            val_loss = evaluate(
                cT,
                eval_Tau,
                batch_size=8,
                device=device,
                metric_only=True,
                dataset=val_dataset,
                model=model,
                time_config=time_config,
            )
            if callback is not None:
                n = len(dataset)
                callback(val_loss, epoch_loss / n, epoch_l1 / n, grad_norms / n, epoch=epoch)

            # Use the validation loss for the learning rate schedule and early stopping
            _val_losses.append((epoch, val_loss.detach().cpu().item()))
            patience = early_stopping_patience // validate_every
            if epoch > max(early_stopping_warmup, early_stopping_patience) and len(_val_losses) > 1:
                current_val_loss = _val_losses[-1][1]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0  # Reset counter because we improved!
                else:
                    patience_counter += 1  # No improvement
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} with loss {val_loss:.5f}.")
                    break
            schedule.step(val_loss)

        progress.set_description(
            f"{_losses[-1][0]:.5f}  {val_loss:.5f}  {optimizer.state_dict()['param_groups'][0]['lr']:.5f}"
        )
        if val_dataset is None:
            schedule.step(epoch_loss)

    return model, _losses, _val_losses


def run_experiment(
    cT,
    Tau,
    dataset_config,
    model_config,
    model_path,
    device=None,
    time_config=None,
):
    ds = init_dataset(dataset_config["train"], cT, Tau, device=device, time_config=time_config)
    ds.add_positional_encoding = model_config["model_name"] == "GPSConv"
    data_loader = DataLoader(ds, batch_size=model_config["batch_size"], shuffle=True)
    data_loader_val = None

    if "validation" in dataset_config:
        ds_val = init_dataset(dataset_config["validation"], cT, model_config["eval_Tau"], device=device, time_config=time_config)
        ds_val.add_positional_encoding = model_config["model_name"] == "GPSConv"
        data_loader_val = DataLoader(ds_val, batch_size=model_config["batch_size"], shuffle=False)
        
    print('Start Training:', model_config["model_name"])
    model = init_model(model_config, device=device)
    model, losses, val_losses = train(
        model,
        data_loader,
        model_config["learning_rate"],
        model_config["epochs"],
        model_config["loss_fn"],
        cT,
        model_config["train_Tau"],
        clip_grad_norm=model_config["clip_grad_norm"],
        scheduler_patience=model_config["scheduler_patience"],
        device=device,
        val_dataset=data_loader_val,
        eval_Tau=model_config["eval_Tau"],
        time_config=time_config,
    )

    model_file = os.path.join(model_path, "model.pt")
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), model_file)
    np.save(os.path.join(model_path, "losses.npy"), np.array(losses))
    np.save(os.path.join(model_path, "val_losses.npy"), np.array(val_losses))


def evaluate(
    cT,
    Tau,
    dataset_config=None,
    model_config=None,
    model_path=None,
    batch_size=8,
    device=None,
    metric_only=False,
    dataset=None,
    model=None,
    time_config=None,
):
    if model is None:
        model = init_model(model_config, device=device)
        if not (
            model_config["model_name"].startswith("MeGA-MP")
            and model_config["layer"] == "advect"
        ):
            model_file = os.path.join(model_path, "model.pt")
            model.load_state_dict(
                safe_torch_load(model_file, weights_only=True, map_location=device)
            )

        Tau = model_config["eval_Tau"]
        
    if dataset is None:
        ds = init_dataset(dataset_config, cT, Tau, device=device, time_config=time_config)
        dataset = DataLoader(ds, batch_size=batch_size, )
        ds.add_positional_encoding = model_config["model_name"] == "GPSConv"

    predictions = []
    ground_truths = []
    flow_fields = []
    edge_lengths = []
    edge_diameters = []
    model_runtimes = []
    preprocessing_times = []
    epanet_runtimes = []
    metric = MeanMetric().to(device)
    l1_metric = torch.nn.L1Loss()

    print('Start Evaluating:', model.__class__.__name__)

    with torch.no_grad():
        for model_inputs in tqdm(dataset, leave=False, desc="Evaluating ..."):
            model_inputs = prepare_features(model_inputs, dataset.dataset.device)
            t0 = timer.time()
            pred, *_ = model(cT=cT, Tau=Tau, n_steps=1, **model_inputs)
            model_runtimes.append(timer.time() - t0)
            if "flow_preprocessing_time" in model_inputs:
                preprocessing_times.append(model_inputs["flow_preprocessing_time"])
            if "epanet_runtime" in model_inputs:
                epanet_runtimes.append(model_inputs["epanet_runtime"])
            pred = pred.squeeze(-1)
            # boundary nodes will not be predicted, values assumed known
            # mask boundary nodes
            mask = (
                torch.eye(len(pred), device=pred.device)[
                    model_inputs.boundary_index
                ]
                .sum(0)
                .bool()
            )
            pred[mask] = model_inputs["y"][mask][:, : pred.shape[-1]]

            if not metric_only:
                predictions.append(pred.detach().cpu())
                ground_truths.append(model_inputs["y"].cpu())
                flow_fields.append(model_inputs["flow_field"].cpu())
                edge_lengths.append(model_inputs["edge_lengths"].cpu())
                edge_diameters.append(model_inputs["edge_diameter"].cpu())
            else:
                y_true = model_inputs["y"][:, : pred.shape[-1]]
                l1_loss = l1_metric(pred[:, cT:], y_true[:, cT:])
                metric.update(l1_loss)

    
    if metric_only:
        return metric.compute()
    # print(torch.nn.L1Loss()(predictions[0][:, cT:], ground_truths[0][:, cT:]).item())
    predictions = torch.cat(predictions, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)
    flow_fields = torch.cat(flow_fields, dim=0)
    edge_lengths = torch.cat(edge_lengths, dim=0)
    edge_diameters = torch.cat(edge_diameters, dim=0)
    model_runtimes = torch.tensor(np.array(model_runtimes))
    if "flow_preprocessing_time" in model_inputs:
        preprocessing_times = torch.cat(preprocessing_times, dim=0)
    if "epanet_runtime" in model_inputs:
        epanet_runtimes = torch.cat(epanet_runtimes, dim=0)

    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "flow_fields": flow_fields,
        "sources_at": ds.hydraulics_ds.sources_at,
        "topology": ds.hydraulics_ds.topology,
        "edge_lengths": edge_lengths,
        "edge_diameters": edge_diameters,
        "model_runtimes": model_runtimes,
        "preprocessing_times": preprocessing_times,
        "epanet_runtimes": epanet_runtimes,
    }
