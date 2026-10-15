from copy import deepcopy
import os

import yaml

CONFIG_PATH = os.path.join(os.getcwd(), "experiments", "experiment_setups.yaml")


def _load_yaml():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

def _expand_defaults(section):
    defaults = section.get("defaults", {})
    experiments = section.get("experiments", [])
    expanded = []
    for idx, exp in enumerate(experiments):
        merged = _deep_merge(defaults, exp)
        if "id" not in merged:
            merged["id"] = idx
        expanded.append(merged)
    updated = deepcopy(section)
    updated["experiments"] = expanded
    return updated

def _resolve_dataset_paths(dataset_cfg, root):
    resolved = deepcopy(dataset_cfg)
    if "data_path" in resolved:
        resolved["data_path"] = os.path.join(root, resolved["data_path"])
    return resolved


def get_wds_config():
    data = _load_yaml()
    section = _expand_defaults(data.get("wds", {}))
    return section


def get_time_complexity_config():
    data = _load_yaml()
    return data.get("time_complexity", {})


def get_pde_gnn_config():
    data = _load_yaml()
    section = _expand_defaults(data.get("pde_gnn", {}))
    root = section.get("root")

    section["results_dir"] = os.path.join(root, section.get("results_dir"))

    experiments = []
    for exp in section.get("experiments", []):
        exp = deepcopy(exp)
        if "train" in exp:
            exp["train"] = _resolve_dataset_paths(exp["train"], root)
        if "test" in exp:
            exp["test"] = {
                name: _resolve_dataset_paths(cfg, root)
                for name, cfg in exp["test"].items()
            }
        experiments.append(exp)

    section["experiments"] = experiments
    return section


def get_euclidean_1d_config():
    data = _load_yaml()
    section = data.get("euclidean_1d", {})
    if "experiments" not in section:
        section["experiments"] = []
    return section


def get_tuning_config():
    data = _load_yaml()
    section = deepcopy(data.get("tuning", {}))
    root = section.get("root")
    dataset_config = section.get("dataset_config", {})
    if root:
        dataset_config = _resolve_dataset_paths(dataset_config, root)
    section["dataset_config"] = dataset_config
    return section
