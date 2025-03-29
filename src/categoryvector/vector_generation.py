"""向量生成模块."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import torch
import time

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryNode, CategoryProcessor
from categoryvector.utils.logging_utils import default_logger as logger
from .models import Category
from categoryvector.vector_storage import VectorStorage


class VectorGenerator:
    """向量生成器"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", config: Optional[CategoryVectorConfig] = None):
        """初始化向量生成器
        
        Args:
            model_name: 使用的模型名称
            config: 配置对象，可选
        """
        self.config = config
        logger.info(f"开始加载Sentence Transformer模型: {model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        logger.info(f"模型向量维度: {self.model.get_sentence_embedding_dimension()}")
        
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
            " ".join(category.keywords) if category.keywords else "",
            " ".join(category.examples) if category.examples else ""
        ]
        text = " ".join(filter(None, texts))
        
        logger.debug(f"为分类ID={category.id}生成完整向量，文本长度={len(text)}字符")
        
        # 生成向量
        start_time = time.time()
        vector = self.model.encode(text)
        encode_time = time.time() - start_time
        
        logger.debug(f"分类ID={category.id}向量生成完成，向量维度={vector.shape}，耗时={encode_time:.4f}秒")
        
        return vector
    
    def generate_level_vectors(self, category: Category) -> Dict[str, np.ndarray]:
        """生成各层级的向量表示
        
        Args:
            category: 分类对象
            
        Returns:
            各层级的向量表示字典
        """
        vectors = {}
        logger.debug(f"开始为分类ID={category.id}生成{len(category.levels)}个层级向量")
        
        start_total_time = time.time()
        for i, level in enumerate(category.levels):
            # 构建层级文本
            if i == 0:
                level_text = level
            else:
                # 添加上下文
                level_text = f"{category.levels[i-1]} > {level}"
            
            # 生成向量
            start_time = time.time()
            level_vector = self.model.encode(level_text)
            encode_time = time.time() - start_time
            
            vectors[f"level_{i+1}"] = level_vector
            logger.debug(f"分类ID={category.id}的层级{i+1}向量生成完成，文本='{level_text}'，耗时={encode_time:.4f}秒")
            
        total_time = time.time() - start_total_time
        logger.debug(f"分类ID={category.id}的所有层级向量生成完成，共{len(vectors)}个层级，总耗时={total_time:.4f}秒")
            
        return vectors
    
    def generate_exclusion_vector(self, exclusions: List[str]) -> np.ndarray:
        """生成排除词的向量表示
        
        Args:
            exclusions: 排除词列表
            
        Returns:
            排除词的向量表示
        """
        if not exclusions:
            logger.debug(f"无排除词，返回零向量")
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        logger.debug(f"为{len(exclusions)}个排除词生成向量: {', '.join(exclusions)}")
        
        # 生成每个排除词的向量
        exclusion_vectors = []
        for ex in exclusions:
            start_time = time.time()
            ex_vector = self.model.encode(ex)
            encode_time = time.time() - start_time
            exclusion_vectors.append(ex_vector)
            logger.debug(f"排除词'{ex}'向量生成完成，耗时={encode_time:.4f}秒")
        
        # 取平均向量
        mean_vector = np.mean(exclusion_vectors, axis=0)
        logger.debug(f"已生成{len(exclusions)}个排除词的平均向量")
        return mean_vector
    
    def generate_query_vector(self, query: str) -> np.ndarray:
        """生成查询文本的向量表示
        
        Args:
            query: 查询文本
            
        Returns:
            查询文本的向量表示
        """
        logger.debug(f"生成查询向量，查询文本: '{query}'，长度={len(query)}字符")
        start_time = time.time()
        vector = self.model.encode(query)
        encode_time = time.time() - start_time
        logger.debug(f"查询向量生成完成，向量维度={vector.shape}，耗时={encode_time:.4f}秒")
        return vector
    
    def enrich_category_vectors(self, category: Category) -> Category:
        """为分类对象生成所有向量
        
        Args:
            category: 分类对象
            
        Returns:
            添加了向量的分类对象
        """
        start_time = time.time()
        
        # 生成完整向量
        logger.debug(f"为分类ID={category.id}生成完整向量")
        category.vector = self.generate_category_vector(category)
        
        # 生成层级向量
        logger.debug(f"为分类ID={category.id}生成层级向量")
        category.level_vectors = self.generate_level_vectors(category)
        
        total_time = time.time() - start_time
        logger.debug(f"分类ID={category.id}的所有向量生成完成，总耗时={total_time:.4f}秒")
        
        return category
