from typing import Optional


class Model:
    '''
    模型加载，可以通过两种方式来进行加载。
    1. 加载mresource文件夹中的模型。
    2. 加载套件的模型.

    Args:
        source(str): 模型的来源，”mindcv“,"mindface"或者”resource“(resource文件夹中的模型).
        name(str): 模型的名称，包括模型，规格，数据集等信息.
        pretrained(bool): 是否加载预训练模型.
    '''
    def __new__(cls,
                source: Optional[str] = None,
                name: Optional[str] = None,
                pretrained: bool = False,
                **kwargs
                ):
        if source == "mindcv":
            import mindcv
            if mindcv.is_model(name):
                model = mindcv.create_model(name=name, pretrained=pretrained, **kwargs)
            else:
                raise ModuleNotFoundError
        elif source == "mindface":
            # 加载过程类似mindcv
            pass
        elif source == "resource":
            model = cls.init_with_name(name=name, pretrained=pretrained, **kwargs)

        return model

    @classmethod
    def init_with_name(cls,
                       name: str,
                       pretrained: bool,
                       ):
        # 通过模型名称，加载模型的注册信息，然后加载模型。
        pass

    @classmethod
    def load_model_info(cls):
        # 加载模型注册信息。
        pass
