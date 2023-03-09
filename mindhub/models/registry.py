import os
import sys

import difflib
from importlib.util import spec_from_file_location, module_from_spec

__all__ = [
    'register_model',
    'list_models_info',
    'load_local_model',
    'local_models',
    'local_models_info'
]

_local_models = dict()
_local_models_info = dict()


def load_local_model(module_name: str,
                     module_local_path: str,
                     **kwargs):

    sys.path.insert(0, module_local_path)

    # Load the module from the specified file path
    spec = spec_from_file_location(module_name, os.path.join(module_local_path, "mindspore_hub_conf.py"))
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # model = getattr(model_module, module_name, None)

    sys.path.pop(0)

    # return model


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
    return _local_models[model_name]


def local_models_info(model_name: str):
    """The info of local models"""
    return _local_models_info[model_name]


def list_models_info(model_name_filter: str,
                     pretrained: bool = False,
                     ):
    """
    Searching models which meet the condition.
    
    Args:
      model_name_filter(str): The possible name of model.
      pretrained(bool): Whether to require the wanted model to have pre-trained model.
    """
    matched_models = [
        m for m in difflib.get_close_matches(model_name_filter, _local_models.keys(), n=1, cutoff=0.5)
        if _local_models_info[m]["pretrained"] or not pretrained
    ]

    if not matched_models:
        return None

    print(f"Matching model: {matched_models}")
    models_info = {m: _local_models_info[m] for m in matched_models}
    return models_info
