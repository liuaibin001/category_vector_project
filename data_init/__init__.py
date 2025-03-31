"""数据初始化包

此包包含用于初始化和生成分类数据的工具。
主要包括：
1. 从CSV生成分类数据
2. 合并单个分类文件
"""

from .scripts.category_generator import CategoryGenerator
from .scripts.merge_categories import merge_category_files

__all__ = ['CategoryGenerator', 'merge_category_files'] 