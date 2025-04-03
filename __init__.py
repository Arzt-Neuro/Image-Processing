"""
mypackage - 一个示例包，展示如何组织Python包结构

这个包演示了如何通过__init__.py文件组织包结构，
使得用户可以直接通过包名调用内部函数。
"""


import sys
import importlib.util
# Versionsdetails
__version__ = '0.2.0'
__author__ = 'Yi-Lun Ge'
__affiliation1__ = 'The George Institute for Global Health, UNSW, Sydney, Australia'
__affiliation2__ = 'The Second Affiliated Hospital, Soochow University, P.R.China'
__bio__ = 'Herzlich willkommen! Viele danke fur mein package utilisieren.'



# 创建一个延迟加载类
class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module(f'.{self.module_name}', package=__name__)
        return getattr(self._module, name)

# 创建延迟加载器实例
lesen = LazyLoader('lesen')
MRBExtractor = lambda *args, **kwargs: lesen.MRBExtractor(*args, **kwargs)
convert_nii = lambda *args, **kwargs: lesen.convert_nii(*args, **kwargs)

logger = LazyLoader('logger')
StreamLogRecorder = lambda *args, **kwargs: logger.StreamLogRecorder(*args, **kwargs)

bildlern = LazyLoader('bildlern')
utils = LazyLoader('utils')
auswt = LazyLoader('auswertung')
auswt.fall = LazyLoader('auswertung.fall_auswt')
auswt.qk = LazyLoader('auswertung.qk')

datei = LazyLoader('datei')
datei.ich = LazyLoader('datei.ich')
datei.image = LazyLoader('datei.image')
datei.text = LazyLoader('datei.text')

# # 从子包中导入
# subpackage = LazyLoader('subpackage.subpackage')
# sub_function = lambda *args, **kwargs: subpackage.sub_function(*args, **kwargs)

# 定义包的公共API
__all__ = [
    # Klasse
    'MRBExtractor',
    'StreamLogRecorder',

    # Funktion
    'convert_nii',

    # Unterpaket
]

# 包级常量
DEFAULT_TIMEOUT = 30
MAX_RETRY_COUNT = 3

# 可选：提供包的初始化功能
def initialize(config=None):
    """初始化包的配置"""
    global DEFAULT_TIMEOUT, MAX_RETRY_COUNT

    if config:
        if 'timeout' in config:
            DEFAULT_TIMEOUT = config['timeout']
        if 'max_retries' in config:
            MAX_RETRY_COUNT = config['max_retries']