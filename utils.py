import numpy as np
import torch

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置多进程数据加载的工作线程种子
    import random
    random.seed(seed)

    # 如果在Windows上，设置以下环境变量可能有助于解决多进程问题
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)