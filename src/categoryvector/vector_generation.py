"""向量生成模块."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Union

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryNode, CategoryProcessor
from categoryvector.utils.logging_utils import default_logger as logger


class VectorGenerator:
    """向量生成器，负责将类别文本转换为向量表示."""
    
    def __init__(self, config: CategoryVectorConfig):
        """初始化向量生成器.
        
        Args:
            config: 配置对象
        """
        self.config = config
        logger.info(f"正在加载模型: {config.model_name}")
        try:
            self.model = SentenceTransformer(config.model_name)
            logger.info(f"模型加载成功，向量维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def generate_vectors(
        self, 
        category_texts: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """为所有类别生成向量.
        
        Args:
            category_texts: (类别ID, 文本)元组的列表
            
        Returns:
            (向量数组, 类别ID列表)的元组
        """
        # 提取文本和ID
        ids = [item[0] for item in category_texts]
        texts = [item[1] for item in category_texts]
        
        logger.info(f"正在为 {len(texts)} 个类别生成向量")
        
        # 批量生成向量
        vectors = self.model.encode(
            texts, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        logger.info(f"向量生成完成，向量维度: {vectors.shape[1]}")
        return vectors, ids
    
    def generate_query_vector(self, query_text: str) -> np.ndarray:
        """为查询文本生成向量.
        
        Args:
            query_text: 查询文本
            
        Returns:
            查询向量
        """
        logger.debug(f"为查询生成向量: {query_text}")
        vector = self.model.encode(
            query_text, 
            normalize_embeddings=True
        )
        return vector
    
    def find_similar_categories(
        self,
        vectors: np.ndarray,
        names: List[str],
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """查找与查询文本最相似的类别.
        
        Args:
            vectors: 类别向量数组
            names: 类别名称列表
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (类别名称, 相似度)元组的列表
        """
        # 生成查询向量
        query_vector = self.generate_query_vector(query_text)
        
        # 计算相似度
        similarities = np.dot(vectors, query_vector)
        
        # 获取最相似的类别
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回结果
        results = []
        for idx in top_indices:
            results.append((names[idx], similarities[idx]))
        
        return results
