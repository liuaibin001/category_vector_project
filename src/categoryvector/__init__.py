"""
CategoryVector - 基于向量搜索的分类查询系统.
"""

__version__ = "1.0.0"

try:
    from src.categoryvector.config import CategoryVectorConfig
    from src.categoryvector.data_processing import CategoryProcessor, CategoryNode
    from src.categoryvector.vector_generation import VectorGenerator
    from src.categoryvector.vector_storage import VectorStorage
    from src.categoryvector.models import Category

    # 导出API相关模块
    from src.categoryvector.api import app
except ImportError:
    # 尝试相对导入
    from .config import CategoryVectorConfig
    from .data_processing import CategoryProcessor, CategoryNode
    from .vector_generation import VectorGenerator
    from .vector_storage import VectorStorage
    from .models import Category
    
    # 导出API相关模块
    try:
        from .api import app
    except ImportError:
        pass

__all__ = [
    "CategoryVectorConfig", 
    "CategoryProcessor", 
    "CategoryNode",
    "VectorGenerator", 
    "VectorStorage",
    "Category",
    "app"
]
