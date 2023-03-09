from typing import Optional

from mindhub.utils import DownLoad
from mindhub.models.registry import _local_models
from mindhub.env import GITHUB_REPO_URL

__all__ = ["Model"]


class Model:
    """
    安装，卸载，加载模型
    """
    _models_from_url = None
    _models_from_local = _local_models.keys()

    def __new__(cls,
                model_name: Optional[str] = None,
                pretrained: bool = False,
                diretory: Optional[str] = None,
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
        cls._models_from_url = cls._check_remote_models()

        if diretory:
            model = cls._init_with_diretory(diretory, pretrained, **kwargs)
        elif model_name:

            if model_name not in cls._models_from_local:
                if model_name not in cls._models_from_url:
                    raise FileNotFoundError
                cls.install_model(model_name)

            model_path = cls._search_model(model_name, pretrained)
            model = cls._init_with_diretory(model_path, pretrained, **kwargs)
        else:
            raise TypeError

        return model
    
    @classmethod
    def install_model(cls,
                     model_name: str
                     ):
        """
        下载并加载模型到注册表中.
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
        """
        pass
    
    @classmethod
    def remove_model(cls,
                     model_name: str
                     ):
        """
        删除模型，同时将模型名称从注册表中删除
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
        """
        pass

    @classmethod
    def _check_remote_models(cls):
        """
        查看远程仓库已存在的模型
        Returns:
            List, the list of models in

        """
        remote_models_list = [file_info["name"]
                       for file_info in DownLoad().list_remote_files(GITHUB_REPO_URL)
                       if file_info["type"] == "dir"]

        return remote_models_list

    @classmethod
    def _search_model(cls,
                     model_name: str
                     ):
        """
        搜索模型名称是否存在于本地模型注册表中，如果不存在是否存在于远程已存在模型注册表中
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
        """


    @classmethod
    def _load_model_info(cls,
                         diretory: str
                         ):
        """
        加载模型注册信息，将模型注册到本地可调用的模型的注册表中.
        
        Args:
            diretory(str): 模型文件的路径.
        """
        pass

    @classmethod
    def _init_with_diretory(cls,
                           diretory: str,
                           pretrained: bool,
                           **kwargs, 
                           ):
        """
        通过本地模型文件加载模型.
        
        Args:
            diretory(str): 模型文件路径.
            pretrained(bool): 是否加载预训练权重.
        """
        pass
    
    @classmethod
    def _init_with_name(cls,
                           model_name: str,
                           pretrained: bool,
                           **kwargs, 
                           ):
        """
        通过模型名称加载模型，如果模型不在本地模型注册表中，则下载对应模型文件
        
        Args:
            diretory(str): 模型名称+规格+使用数据集
            pretrained(bool): 是否加载预训练权重.
        """
        pass
