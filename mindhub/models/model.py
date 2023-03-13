import os
import shutil
from typing import Optional

from mindhub.utils.download import DownLoad, get_default_download_root, \
    set_default_download_root
from mindhub.models.registry import local_models, list_models, \
    local_models_info, load_local_model, remove_models

__all__ = ["Model"]


class Model:
    """
    安装，卸载，加载模型
    """
    def __new__(cls,
                model_name: str,
                diretory: Optional[str] = None,
                pretrained: bool = False,
                **kwargs
                ):
        """
        模型加载，可以通过两种方式来进行加载。
        1. 加载mresource文件夹中的模型。
        2. 加载套件的模型.

        Args:
            model_name(str): 模型名称+规格+使用数据.
            pretrained(bool): 加载我们训练好的精度达标的模型预训练权重.
            diretory(str): 加载本地的模型文件的路径.
        """
        if model_name and diretory:
            model = cls._init_with_diretory(model_name, diretory, pretrained, **kwargs)
            return model
        elif model_name:
            model = cls._init_with_name(model_name, pretrained, **kwargs)
            return model
        elif not model_name:
            raise ValueError("The parameter model_name is required!")
    
    @classmethod
    def install_model(cls,
                      model_name: str,
                      download_path: Optional[str] = None,
                     ):
        """
        下载并加载模型到注册表中.
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
            download_path(str): 模型文件下载路径，默认使用默认下载路径. Default: None.
        """
        install_item = cls._search_model(model_name)

        if install_item:

            if install_item == "remote model":
                if download_path:
                    set_default_download_root(download_path)

                download_path = DownLoad().download_github_folder(model_name, download_path)
                load_local_model(model_name, download_path)
            else:
                print(f"{model_name} already exists in the local registry, please use it directly.")

            return download_path if download_path else \
                os.path.join(get_default_download_root(), model_name)

    @classmethod
    def remove_model(cls, model_name: str):
        """
        删除模型，同时将模型名称从注册表中删除
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
        """
        if local_models(model_name):
            default_model_path = os.path.join(get_default_download_root(), model_name)

            try:
                shutil.rmtree(default_model_path)
                print(f"{default_model_path} has been successfully deleted.")
            except OSError as e:
                print(f"Error deleting {default_model_path}: {e}")

            remove_models(model_name)
        else:
            raise ValueError(f"{model_name} is not in the local models registry.")


    @classmethod
    def _search_model(cls, model_name: str):
        """
        搜索模型名称是否存在于本地模型注册表中，如果不存在是否存在于远程已存在模型注册表中
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
        """
        local_matched_models, remote_matched_models = list_models(model_name)
        local_model_path = os.path.join(get_default_download_root(), model_name)

        if not (model_name in local_matched_models or model_name in remote_matched_models):
            print(f"Do you mean one of these models, Local Models: {local_matched_models}, "
                  f"Remote Models: {remote_matched_models}")

        elif model_name in local_matched_models:
            return model_name

        elif model_name not in local_matched_models and model_name in remote_matched_models:

            if os.path.exists(local_model_path):
                load_local_model(model_name, local_model_path)
                return model_name

            print(f"{model_name} is not installed!")
            return "remote model"

    @classmethod
    def _load_model_info(cls, model_name: str):
        """
        加载模型注册信息，将模型注册到本地可调用的模型的注册表中.
        
        Args:
            model_name(str): 模型名称+规格+数据集.
        """
        model_info = local_models_info(model_name)

        if model_info:
            print(model_info)
        else:
            raise KeyError(f"{model_name} is not in local registry.")

    @classmethod
    def _init_with_diretory(cls,
                            model_name: str,
                            diretory: str,
                            pretrained: bool,
                            **kwargs,
                           ):
        """
        通过本地模型文件加载模型.
        
        Args:
            mdoel_name(str): 模型名称+规格+数据集。
            diretory(str): 模型文件路径.
            pretrained(bool): 是否加载预训练权重.
        """

        if os.path.exists(diretory):
            load_local_model(model_name, diretory)
            model = local_models(model_name)(model_name, pretrained=pretrained, **kwargs)
            return model
        else:
            raise FileNotFoundError(f"Please check whether the path {diretory} exists.")

    @classmethod
    def _init_with_name(cls,
                        model_name: str,
                        pretrained: bool,
                        **kwargs,
                        ):
        """
        通过模型名称加载模型，如果模型不在本地模型注册表中，则下载对应模型文件
        
        Args:
            model_name(str): 模型名称+规格+使用数据集
            pretrained(bool): 是否加载预训练权重.
        """
        model_cls = local_models(model_name)

        if not model_cls:
            download_path = cls.install_model(model_name)

            if download_path:
                model = cls._init_with_diretory(model_name, download_path, pretrained, **kwargs)
                return model
        else:
            model = model_cls(model_name, pretrained, **kwargs)
            return model
