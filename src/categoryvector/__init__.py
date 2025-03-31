"""
CategoryVector - 基于向量搜索的分类查询系统.
"""

__version__ = "0.1.0"

# 基础模型和配置
from .models import Category
from .config import CategoryVectorConfig

# 数据处理相关
from .data_processing import CategoryProcessor, CategoryNode

# 核心功能模块
from .vector_generation import VectorGenerator
from .vector_storage import VectorStorage

# API相关
def get_app():
    """获取FastAPI应用实例"""
    from .api import app
    return app

__all__ = [
    "Category",
    "CategoryVectorConfig",
    "CategoryProcessor",
    "CategoryNode",
    "VectorGenerator",
    "VectorStorage",
    "get_app"
]
