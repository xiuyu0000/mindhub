
_local_models = dict()

def register_model_info(
                       model_name: str,
                       paper: str = "",
                       type: str = "",
                       ):
    '''
    将需要执行的类进行注册.
    
    Args:
       model_name(str): 需要注册的模型名称+规格+数据集.
       paper(str): 模型的原论文名称，方便用户区分简称相同的不同模型。
       type(str): 所属领域/子领域
    '''

    # 注册模型信息
    def _wrapper():
        return 0

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
