import json
import os
from pathlib import Path
from typing import Union

import open3d as o3d
import torch
import yaml


def mkdir_decorator(func):
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_clouds(clouds: list, cloud_names: list, *, directory: Union[str, Path]) -> None:
    for cld_name, cloud in zip(cloud_names, clouds):
        o3d.io.write_point_cloud(str(directory / cld_name), cloud)


@mkdir_decorator
def save_dict_to_ckpt(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)


@mkdir_decorator
def save_dict_to_yaml(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)


@mkdir_decorator
def save_dict_to_json(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    with open(directory / file_name, "w") as f:
        json.dump(dictionary, f)


def load_config(path: str, default_path: str = None) -> dict:
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)
    inherit_from = cfg_special.get('inherit_from')
    cfg = dict()
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1: dict, dict2: dict) -> None:
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
