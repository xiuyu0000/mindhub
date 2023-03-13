import os
import sys

import difflib
from importlib.util import spec_from_file_location, module_from_spec

from mindhub.env import GITHUB_REPO_URL
from mindhub.utils.download import DownLoad


__all__ = [
    "get_remote_models",
    'register_model',
    'list_models',
    'load_local_model',
    'local_models',
    "remove_models",
    'local_models_info'
]


_local_models = dict()
_local_models_info = dict()


def _get_remote_models():
    """
    Get Remote Models List.
    """
    remote_models_list = [
        file_info["name"]
        for file_info in DownLoad().list_remote_files(GITHUB_REPO_URL)
        if file_info["type"] == "dir"
    ]

    return remote_models_list


def get_remote_models():
    return _get_remote_models()


def load_local_model(module_name: str,
                     module_local_path: str,
                     ):

    sys.path.insert(0, module_local_path)

    # Load the module from the specified file path
    spec = spec_from_file_location(module_name, os.path.join(module_local_path, "mindspore_hub_conf.py"))
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)

    sys.path.pop(0)

    if module_name not in _local_models:
        raise  ValueError(f"Failed to register {module_name} in model registry.")


def register_model(model_name: str,
                   pretrained: bool = False,
                   paper: str = "",
                   model_type: str = "",
                   ):
    """
    Register the class or method of the model to local models registry.

    Args:
        model_name(str): The name of model.
        pretrained(bool): Whether the model have pre-trained model. Default: False.
        paper(str): The name of paper of the model. Default: ''.
        model_type(str): The domain of the model. Default: ''.
    """
    def decorator(cls):
        _local_models[model_name] = cls
        _local_models_info[model_name] = \
            {"model_name": model_name,
             "pretrained": pretrained,
             "paper": paper,
             "model_type": model_type}
        return cls

    return decorator


def local_models(model_name: str):
    """The local models."""
    return _local_models.get(model_name, None)


def remove_models(model_name: str):
    """remove the local models"""
    return _local_models.pop(model_name, None)


def local_models_info(local_model_name: str):
    """
    List the specified model's info.

    Args:
        local_model_name(str): The registered model names.

    Returns:
        Dict, the info of the specified model.
    """
    return _local_models_info.get(local_model_name, None)


def list_models(model_name_filter: str):


    local_matched_models = [
        m for m in difflib.get_close_matches(
            model_name_filter, _local_models.keys(), n=1, cutoff=0.5)
    ]

    remote_matched_models = [
        m for m in difflib.get_close_matches(
            model_name_filter, get_remote_models(), n=1, cutoff=0.5)
    ]

    if not(local_matched_models or remote_matched_models):
        raise ValueError(f"{model_name_filter} not found in local registry or remote repository.")


    print(f"Matching local models: {local_matched_models}")
    print(f"Matching remote models: {remote_matched_models}")

    return local_matched_models, remote_matched_models
