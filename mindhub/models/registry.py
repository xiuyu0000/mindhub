
_local_models = dict()

def register_model_info(
                       model_name: str,
                       paper: str = "",
                       type: str = "",
                       ):
    '''
    将需要执行的类进行注册
    Args:
        call_class(class): 需要注册的类
    '''

    # 注册模型信息
    def _wrapper():
        return 0

    return _wrapper

def list_models(
               filter: str,
               exclude_filter: str,
               module: str,
               pretrain: bool = False,
               ):
    """
    模糊匹配符合条件的模型并打印符合条件的模型列表
    """
    pass
