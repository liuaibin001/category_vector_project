"""向量存储模块."""

import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryNode, CategoryProcessor
from categoryvector.utils.logging_utils import default_logger as logger


class VectorStorage:
    """向量存储器，负责存储和检索类别向量."""
    
    def __init__(self, config: CategoryVectorConfig):
        """初始化向量存储器.
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.index = None
        self.id_mapping: Dict[int, str] = {}  # faiss索引ID到类别ID的映射
        self.category_processor: Optional[CategoryProcessor] = None
    
    def store_vectors(
        self,
        vectors: np.ndarray,
        names: List[str],
        metadata: Optional[Dict] = None
    ) -> None:
        """存储向量和元数据.
        
        Args:
            vectors: 向量数组
            names: 类别名称列表
            metadata: 可选的元数据字典
        """
        try:
            # 保存向量
            vector_file = self.config.data_dir / "vectors.npy"
            np.save(vector_file, vectors)
            
            # 保存名称和元数据
            info = {
                "names": names,
                "metadata": metadata or {}
            }
            info_file = self.config.data_dir / "info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
            
            logger.info(f"存储了 {len(vectors)} 个向量到 {self.config.data_dir}")
        except Exception as e:
            logger.error(f"存储向量时发生错误: {e}")
            raise
    
    def load_vectors(self) -> Tuple[np.ndarray, List[str], Dict]:
        """加载向量和元数据.
        
        Returns:
            (向量数组, 类别名称列表, 元数据字典)的元组
        """
        try:
            # 加载向量
            vector_file = self.config.data_dir / "vectors.npy"
            vectors = np.load(vector_file)
            
            # 加载名称和元数据
            info_file = self.config.data_dir / "info.json"
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            logger.info(f"从 {self.config.data_dir} 加载了 {len(vectors)} 个向量")
            return vectors, info["names"], info["metadata"]
        except Exception as e:
            logger.error(f"加载向量时发生错误: {e}")
            raise
    
    def build_index(
        self, 
        id_to_vector: Dict[str, np.ndarray],
        category_processor: CategoryProcessor
    ) -> None:
        """构建向量索引.
        
        Args:
            id_to_vector: 类别ID到向量的映射
            category_processor: 类别处理器
        """
        if not id_to_vector:
            logger.warning("没有向量数据，无法构建索引")
            return
        
        # 保存类别处理器引用
        self.category_processor = category_processor
        
        # 准备向量数据
        category_ids = list(id_to_vector.keys())
        vectors = np.array([id_to_vector[id_] for id_ in category_ids], dtype=np.float32)
        
        # 创建ID映射
        self.id_mapping = {i: category_id for i, category_id in enumerate(category_ids)}
        
        # 获取向量维度
        d = vectors.shape[1]
        
        # 根据配置创建不同类型的索引
        if self.config.index_type == "flat":
            self.index = faiss.IndexFlatIP(d)  # 内积相似度索引
        elif self.config.index_type == "ivf":
            # IVF索引需要先训练
            quantizer = faiss.IndexFlatIP(d)
            nlist = min(self.config.nlist, vectors.shape[0])  # 聚类数不能超过向量数
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # 训练索引
            if not self.index.is_trained:
                logger.info(f"正在训练IVF索引，nlist={nlist}")
                self.index.train(vectors)
        elif self.config.index_type == "hnsw":
            # HNSW索引
            self.index = faiss.IndexHNSWFlat(d, self.config.m_factor, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"不支持的索引类型: {self.config.index_type}")
        
        # 添加向量到索引
        logger.info(f"正在添加 {len(vectors)} 个向量到索引")
        self.index.add(vectors)
        
        logger.info(f"索引构建完成，类型: {self.config.index_type}, 向量数: {self.index.ntotal}")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """搜索最相似的类别.
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量，默认使用配置中的值
            
        Returns:
            搜索结果列表，每个结果包含类别信息和相似度分数
        """
        if self.index is None:
            logger.error("索引未构建，无法搜索")
            return []
        
        if self.category_processor is None:
            logger.error("类别处理器未设置，无法搜索")
            return []
        
        # 使用默认top_k或配置中的值
        if top_k is None:
            top_k = self.config.top_k
        
        # 确保查询向量是二维的
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # 执行搜索
        scores, indices = self.index.search(query_vector, top_k)
        
        # 整理搜索结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.id_mapping):
                continue  # 无效的索引
                
            category_id = self.id_mapping[idx]
            category = self.category_processor.get_category_by_id(category_id)
            
            if category and score >= self.config.similarity_threshold:
                results.append({
                    "id": category.id,
                    "name": category.name,
                    "description": category.description,
                    "parent_id": category.parent_id,
                    "level": category.level,
                    "score": float(score)
                })
        
        return results
    
    def save(self, directory: Union[str, Path]) -> None:
        """保存索引和相关数据.
        
        Args:
            directory: 保存目录
        """
        if self.index is None:
            logger.error("索引未构建，无法保存")
            return
        
        # 确保目录存在
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        index_file = directory / "faiss_index.bin"
        logger.info(f"正在保存索引到 {index_file}")
        faiss.write_index(self.index, str(index_file))
        
        # 保存ID映射
        mapping_file = directory / "id_mapping.pkl"
        logger.info(f"正在保存ID映射到 {mapping_file}")
        with open(mapping_file, 'wb') as f:
            pickle.dump(self.id_mapping, f)
        
        # 保存配置信息
        config_file = directory / "index_config.json"
        logger.info(f"正在保存索引配置到 {config_file}")
        with open(config_file, 'w', encoding='utf-8') as f:
            config_data = {
                "index_type": self.config.index_type,
                "vector_dim": self.index.d,
                "ntotal": self.index.ntotal
            }
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"索引保存完成，位置: {directory}")
    
    def load(self, directory: Union[str, Path]) -> None:
        """加载索引和相关数据.
        
        Args:
            directory: 索引目录
        """
        directory = Path(directory)
        
        # 加载索引
        index_file = directory / "faiss_index.bin"
        logger.info(f"正在加载索引: {index_file}")
        self.index = faiss.read_index(str(index_file))
        
        # 加载ID映射
        mapping_file = directory / "id_mapping.pkl"
        logger.info(f"正在加载ID映射: {mapping_file}")
        with open(mapping_file, 'rb') as f:
            self.id_mapping = pickle.load(f)
        
        # 加载类别处理器
        category_file = self.config.data_dir / "categories.json"
        if category_file.exists():
            logger.info(f"正在加载类别数据: {category_file}")
            self.category_processor = CategoryProcessor(self.config)
            self.category_processor.load_from_json(category_file)
        else:
            logger.warning(f"类别数据文件不存在: {category_file}")
        
        logger.info(f"索引加载完成，向量数: {self.index.ntotal}")
