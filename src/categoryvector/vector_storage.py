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
from .models import Category


class VectorStorage:
    """向量存储管理器"""
    
    def __init__(self, dimension: int, config: Optional[CategoryVectorConfig] = None):
        """初始化向量存储
        
        Args:
            dimension: 向量维度
            config: 配置对象，可选
        """
        self.dimension = dimension
        self.config = config
        self.index = faiss.IndexFlatL2(dimension)
        self.categories: Dict[int, Category] = {}
        
    def add_category(self, category: Category):
        """添加分类到索引
        
        Args:
            category: 分类对象
        """
        if category.vector is None:
            raise ValueError("Category must have a vector")
            
        # 添加到FAISS索引
        self.index.add(np.array([category.vector]).astype('float32'))
        
        # 保存分类对象
        self.categories[category.id] = category
        
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.6,
        exclude_ids: Optional[List[int]] = None
    ) -> List[Tuple[Category, float]]:
        """搜索相似分类
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            threshold: 相似度阈值
            exclude_ids: 要排除的分类ID列表
            
        Returns:
            (分类对象, 相似度)元组的列表，按相似度降序排序
        """
        print(f"开始搜索，索引包含 {self.index.ntotal} 个向量，分类数量 {len(self.categories)}")
        print(f"请求返回 {top_k} 个结果，相似度阈值 {threshold}")
        
        # 确保查询向量是正确的形状和类型
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        
        # 搜索向量 - 获取更多结果用于过滤，确保有足够的候选项
        fetch_k = min(top_k * 3, self.index.ntotal)  # 获取更多结果以应对阈值过滤
        distances, indices = self.index.search(
            query_vector,
            fetch_k
        )
        
        print(f"FAISS搜索结果：获取了 {fetch_k} 个候选项")
        
        # 处理结果
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS填充值
                continue
            
            # 将索引转换为分类ID
            if idx < len(self.categories):
                # 直接使用索引获取对应的分类ID
                category_id = list(self.categories.keys())[idx]
                category = self.categories[category_id]
                
                # 排除指定ID
                if exclude_ids and category.id in exclude_ids:
                    continue
                
                # L2距离转换为相似度
                # 使用指数衰减函数，让相似度更符合直觉
                similarity = np.exp(-distance / 10.0)  # 除以10是为了调整衰减速率
                
                # 应用阈值
                if similarity >= threshold:
                    results.append((category, similarity))
            else:
                print(f"警告：索引 {idx} 超出分类数量范围 {len(self.categories)}")
        
        # 按相似度排序（降序）
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 只返回指定数量的结果
        final_results = results[:top_k]
        
        print(f"过滤后找到 {len(results)} 个结果，返回前 {len(final_results)} 个")
        
        # 打印返回的结果
        for i, (category, similarity) in enumerate(final_results):
            print(f"结果 {i+1}: 分类 {category.id} ('{category.path}'), 相似度 {similarity:.4f}")
        
        return final_results
    
    def search_by_level(
        self,
        query_vector: np.ndarray,
        level: int,
        top_k: int = 10,
        threshold: float = 0.6
    ) -> List[Tuple[Category, float]]:
        """按层级搜索分类
        
        Args:
            query_vector: 查询向量
            level: 目标层级
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            (分类对象, 相似度)元组的列表，按相似度降序排序
        """
        print(f"开始按层级 {level} 搜索，分类数量 {len(self.categories)}")
        print(f"请求返回 {top_k} 个结果，相似度阈值 {threshold}")
        
        # 确保查询向量是正确的形状
        if query_vector.ndim != 1:
            query_vector = query_vector.reshape(-1)
            
        # 收集所有满足条件的结果
        results = []
        candidates_count = 0
        
        for category in self.categories.values():
            # 检查是否有当前层级
            if category.level_depth < level:
                continue
                
            candidates_count += 1
                
            # 获取层级向量
            level_vector = category.level_vectors.get(f"level_{level}")
            if level_vector is None:
                continue
                
            # 计算相似度
            # 对于层级向量，我们使用余弦相似度
            norm_query = np.linalg.norm(query_vector)
            norm_level = np.linalg.norm(level_vector)
            
            if norm_query > 0 and norm_level > 0:
                dot_product = np.dot(query_vector, level_vector)
                # 余弦相似度范围为[-1,1]，我们将其映射到[0,1]
                similarity = (dot_product / (norm_query * norm_level) + 1) / 2
            else:
                similarity = 0.0
            
            # 应用阈值
            if similarity >= threshold:
                results.append((category, similarity))
                
        print(f"层级 {level} 下有 {candidates_count} 个分类，满足阈值的有 {len(results)} 个")
            
        # 按相似度排序（降序）
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 只返回指定数量的结果
        final_results = results[:top_k]
        
        print(f"返回前 {len(final_results)} 个结果")
        
        # 打印返回的结果
        for i, (category, similarity) in enumerate(final_results):
            print(f"结果 {i+1}: 分类 {category.id} ('{category.path}'), 相似度 {similarity:.4f}")
        
        return final_results
    
    def save(self, directory: Path):
        """保存索引和分类数据
        
        Args:
            directory: 保存目录
        """
        # 创建目录
        directory.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # 保存分类数据为数组格式
        categories_data = [
            cat.to_dict() 
            for cat in self.categories.values()
        ]
        with open(directory / "categories.json", "w", encoding="utf-8") as f:
            json.dump(categories_data, f, ensure_ascii=False, indent=2)
    
    def load(self, directory: Path):
        """加载索引和分类数据
        
        Args:
            directory: 数据目录
        """
        directory = Path(directory)
        
        # 尝试加载新格式索引
        index_file = directory / "index.faiss"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            
            # 加载分类数据
            categories_file = directory / "categories.json"
            if categories_file.exists():
                with open(categories_file, "r", encoding="utf-8") as f:
                    categories_data = json.load(f)
                    
                if isinstance(categories_data, list):
                    # 数组格式 [{}, {}, ...]
                    self.categories = {
                        cat_data["id"]: Category.from_dict(cat_data)
                        for cat_data in categories_data
                    }
                else:
                    # 字典格式 {"1": {}, "2": {}, ...}
                    self.categories = {
                        int(cat_id): Category.from_dict(cat_data)
                        for cat_id, cat_data in categories_data.items()
                    }
        # 尝试加载旧格式索引
        else:
            vectors_file = directory / "vectors.npy"
            info_file = directory / "info.json"
            
            if vectors_file.exists() and info_file.exists():
                # 加载向量
                vectors = np.load(vectors_file)
                
                # 加载名称和元数据
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                # 构建索引
                self.index = faiss.IndexFlatL2(vectors.shape[1])
                self.index.add(vectors.astype('float32'))
                
                # 创建临时分类
                for i, name in enumerate(info["names"]):
                    category = Category(
                        id=i+1,
                        path=name,
                        levels=[name],
                        level_depth=1,
                        description=f"Auto-generated category for {name}"
                    )
                    self.categories[category.id] = category
