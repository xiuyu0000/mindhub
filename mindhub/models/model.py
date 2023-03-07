from typing import Optional


class Model:
    '''
    模型加载，可以通过两种方式来进行加载。
    1. 加载mresource文件夹中的模型。
    2. 加载套件的模型.

    Args:
        model_name(str): 模型名称+规格+使用数据.
        pretrained(bool): 加载我们训练好的精度达标的模型预训练权重.
        diretory(str): 加载本地的模型文件的路径.
    '''
    def __new__(cls,
                model_name: Optional[str] = None,
                pretrained: bool = False,
                diretory: Optional[str] = None,
                **kwargs
                ):
        if diretory:
            model = cls._init_with_diretory(diretory, pretrained, **kwargs)
        elif model_name:
            model_path = cls._search_model(model_name, pretrained, **kwargs)
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
    def _search_model(cls,
                     model_name: str
                     ):
        """
        搜索模型名称是否存在于本地模型注册表中，如果不存在是否存在于远程已存在模型注册表中
        
        Args:
            model_name(str): 模型名称+规格+使用数据集.
        """
        pass

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
