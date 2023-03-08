import os
import sys
from importlib.util import spec_from_file_location, module_from_spec

from typing import Generic

from mindhub.utils.download import DownLoad
from mindhub.env import GITHUB_REPO_URL

__all__ = ['register_model_info',
           'list_models']

_local_models = dict()


def load_local_model(module_name: str,
                     module_local_path: str,
                     **kwargs):

    sys.path.insert(0, module_local_path)

    # Load the module from the specified file path
    spec = spec_from_file_location(module_name, os.path.join(module_local_path, "mindspore_hub_conf.py"))
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model = model_module[module_name]

    sys.path.pop(0)

    return model


def register_model_info(model_name: str,
                        pretrained: bool = False,
                        paper: str = "",
                        model_type: str = "",
                        ):
    """
    将需要执行的类进行注册.

    Args:
       model_name(str): 需要注册的模型名称+规格+数据集.
       pretrained(bool): 是否存在可加载的预训练模型.
       paper(str): 模型的原论文名称，方便用户区分简称相同的不同模型。
       model_type(str): 所属领域/子领域
    """

    # 注册模型信息
    def _wrapper(cls: Generic) -> Generic:
       pass

    return _wrapper


def list_models(
               filter: str,
               exclude_filter: str,
               pretrain: bool = False,
               ):
    """
    模糊匹配符合条件的模型并打印符合条件的模型列表.
    
    Args:
      filter(str): 模型检索，支持通配符过滤器.
      exclude_filter(str): 检索除通配符过滤器之外的所有模型.
      pretrained(bool): 当为True时，只检索出有预训练权重的模型.
    """
    pass
