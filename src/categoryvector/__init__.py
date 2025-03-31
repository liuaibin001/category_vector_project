"""
CategoryVector - 基于向量搜索的分类查询系统.
"""

__version__ = "1.0.0"

# 首先导入基础配置和模型
from .config import CategoryVectorConfig
from .models import Category

# 然后导入数据处理相关
from .data_processing import CategoryProcessor, CategoryNode

# 最后导入核心功能模块
from .vector_generation import VectorGenerator
from .vector_storage import VectorStorage

# 导出API相关模块
try:
    from .api import app
except ImportError:
    app = None

__all__ = [
    "CategoryVectorConfig", 
    "CategoryProcessor", 
    "CategoryNode",
    "VectorGenerator", 
    "VectorStorage",
    "Category",
    "app"
]
