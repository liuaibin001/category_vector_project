"""向量生成模块."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import torch

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryNode, CategoryProcessor
from categoryvector.utils.logging_utils import default_logger as logger
from .models import Category


class VectorGenerator:
    """向量生成器"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", config: Optional[CategoryVectorConfig] = None):
        """初始化向量生成器
        
        Args:
            model_name: 使用的模型名称
            config: 配置对象，可选
        """
        self.config = config
        self.model = SentenceTransformer(model_name)
        
    def generate_category_vector(self, category: Category) -> np.ndarray:
        """生成分类的完整向量表示
        
        Args:
            category: 分类对象
            
        Returns:
            分类的向量表示
        """
        # 组合所有文本信息
        texts = [
            category.path,
            category.description or "",
            " ".join(category.keywords),
            " ".join(category.examples)
        ]
        text = " ".join(filter(None, texts))
        
        # 生成向量
        return self.model.encode(text)
    
    def generate_level_vectors(self, category: Category) -> Dict[str, np.ndarray]:
        """生成各层级的向量表示
        
        Args:
            category: 分类对象
            
        Returns:
            各层级的向量表示字典
        """
        vectors = {}
        
        for i, level in enumerate(category.levels):
            # 构建层级文本
            if i == 0:
                level_text = level
            else:
                # 添加上下文
                level_text = f"{category.levels[i-1]} > {level}"
            
            # 生成向量
            vectors[f"level_{i+1}"] = self.model.encode(level_text)
            
        return vectors
    
    def generate_exclusion_vector(self, exclusions: List[str]) -> np.ndarray:
        """生成排除词的向量表示
        
        Args:
            exclusions: 排除词列表
            
        Returns:
            排除词的向量表示
        """
        if not exclusions:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # 生成每个排除词的向量
        exclusion_vectors = [self.model.encode(ex) for ex in exclusions]
        
        # 取平均向量
        return np.mean(exclusion_vectors, axis=0)
    
    def generate_query_vector(self, query: str) -> np.ndarray:
        """生成查询文本的向量表示
        
        Args:
            query: 查询文本
            
        Returns:
            查询文本的向量表示
        """
        return self.model.encode(query)
    
    def enrich_category_vectors(self, category: Category) -> Category:
        """为分类对象生成所有向量
        
        Args:
            category: 分类对象
            
        Returns:
            添加了向量的分类对象
        """
        # 生成完整向量
        category.vector = self.generate_category_vector(category)
        
        # 生成层级向量
        category.level_vectors = self.generate_level_vectors(category)
        
        return category
