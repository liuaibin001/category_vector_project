"""向量存储模块."""

import os
import json
import pickle
import numpy as np
import faiss
import time
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
        start_time = time.time()
        logger.info(f"开始保存索引到目录: {directory}")
        
        # 创建目录
        directory.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        index_path = directory / "index.faiss"
        logger.info(f"保存FAISS索引到: {index_path}")
        faiss_start_time = time.time()
        faiss.write_index(self.index, str(index_path))
        faiss_time = time.time() - faiss_start_time
        index_size = index_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"FAISS索引保存完成，耗时: {faiss_time:.2f}秒，文件大小: {index_size:.2f} MB，向量数: {self.index.ntotal}")
        
        # 保存分类数据为数组格式
        categories_path = directory / "categories.json"
        logger.info(f"保存类别数据到: {categories_path}")
        json_start_time = time.time()
        
        # 序列化前计算类别数
        categories_count = len(self.categories)
        logger.info(f"序列化 {categories_count} 个类别...")
        
        # 将分类转换为字典
        categories_data = []
        for i, cat in enumerate(self.categories.values()):
            if i % 100 == 0 and i > 0:
                logger.debug(f"已序列化 {i}/{categories_count} 个类别")
            try:
                cat_dict = cat.to_dict()
                categories_data.append(cat_dict)
            except Exception as e:
                logger.error(f"序列化类别 ID={cat.id} 时出错: {e}")
        
        # 写入JSON文件
        json_write_start = time.time()
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(categories_data, f, ensure_ascii=False, indent=2)
        json_write_time = time.time() - json_write_start
        json_total_time = time.time() - json_start_time
        
        categories_size = categories_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"类别数据保存完成，序列化耗时: {json_total_time:.2f}秒，写入耗时: {json_write_time:.2f}秒")
        logger.info(f"类别数据文件大小: {categories_size:.2f} MB")
        
        # 记录总耗时
        total_time = time.time() - start_time
        total_size = index_size + categories_size
        logger.info(f"保存完成，总耗时: {total_time:.2f}秒，总文件大小: {total_size:.2f} MB")
    
    def load(self, directory: Path):
        """加载索引和分类数据
        
        Args:
            directory: 数据目录
        """
        start_time = time.time()
        logger.info(f"开始从目录加载索引: {directory}")
        
        directory = Path(directory)
        
        # 尝试加载新格式索引
        index_file = directory / "index.faiss"
        if index_file.exists():
            logger.info(f"找到FAISS索引文件: {index_file}")
            index_size = index_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"开始读取FAISS索引，文件大小: {index_size:.2f} MB")
            
            index_start_time = time.time()
            self.index = faiss.read_index(str(index_file))
            index_time = time.time() - index_start_time
            
            logger.info(f"FAISS索引加载完成，耗时: {index_time:.2f}秒，索引包含 {self.index.ntotal} 个向量")
            
            # 加载分类数据
            categories_file = directory / "categories.json"
            if categories_file.exists():
                logger.info(f"找到类别数据文件: {categories_file}")
                categories_size = categories_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"开始读取类别数据，文件大小: {categories_size:.2f} MB")
                
                json_start_time = time.time()
                with open(categories_file, "r", encoding="utf-8") as f:
                    categories_data = json.load(f)
                json_time = time.time() - json_start_time
                logger.info(f"类别数据JSON解析完成，耗时: {json_time:.2f}秒，包含 {len(categories_data)} 条数据")
                
                # 开始转换类别数据
                process_start_time = time.time()
                categories_count = 0
                
                if isinstance(categories_data, list):
                    # 数组格式 [{}, {}, ...]
                    logger.info(f"检测到数组格式的类别数据，开始处理 {len(categories_data)} 个类别")
                    for i, cat_data in enumerate(categories_data):
                        try:
                            if i % 100 == 0 and i > 0:
                                logger.debug(f"已加载 {i}/{len(categories_data)} 个类别")
                            category = Category.from_dict(cat_data)
                            self.categories[category.id] = category
                            categories_count += 1
                        except Exception as e:
                            logger.error(f"加载类别 ID={cat_data.get('id', '未知')} 时出错: {e}")
                else:
                    # 字典格式 {"1": {}, "2": {}, ...}
                    logger.info(f"检测到字典格式的类别数据，开始处理 {len(categories_data)} 个类别")
                    for i, (cat_id, cat_data) in enumerate(categories_data.items()):
                        try:
                            if i % 100 == 0 and i > 0:
                                logger.debug(f"已加载 {i}/{len(categories_data)} 个类别")
                            category = Category.from_dict(cat_data)
                            self.categories[int(cat_id)] = category
                            categories_count += 1
                        except Exception as e:
                            logger.error(f"加载类别 ID={cat_id} 时出错: {e}")
                
                process_time = time.time() - process_start_time
                logger.info(f"类别数据处理完成，耗时: {process_time:.2f}秒，成功加载 {categories_count} 个类别")
            else:
                logger.warning(f"未找到类别数据文件: {categories_file}")
        
        # 尝试加载旧格式索引
        else:
            vectors_file = directory / "vectors.npy"
            info_file = directory / "info.json"
            
            if vectors_file.exists() and info_file.exists():
                logger.info(f"未找到FAISS索引文件，尝试加载旧格式数据")
                logger.info(f"找到向量文件: {vectors_file} 和信息文件: {info_file}")
                
                # 加载向量
                vectors_start_time = time.time()
                vectors = np.load(vectors_file)
                vectors_time = time.time() - vectors_start_time
                logger.info(f"向量数据加载完成，耗时: {vectors_time:.2f}秒，形状: {vectors.shape}")
                
                # 加载名称和元数据
                info_start_time = time.time()
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                info_time = time.time() - info_start_time
                logger.info(f"信息数据加载完成，耗时: {info_time:.2f}秒，包含 {len(info['names'])} 个名称")
                
                # 构建索引
                index_build_start = time.time()
                self.index = faiss.IndexFlatL2(vectors.shape[1])
                self.index.add(vectors.astype('float32'))
                index_build_time = time.time() - index_build_start
                logger.info(f"FAISS索引构建完成，耗时: {index_build_time:.2f}秒，向量数: {self.index.ntotal}")
                
                # 创建临时分类
                logger.info(f"开始从旧格式创建临时分类对象")
                for i, name in enumerate(info["names"]):
                    category = Category(
                        id=i+1,
                        path=name,
                        levels=[name],
                        level_depth=1,
                        description=f"Auto-generated category for {name}"
                    )
                    self.categories[category.id] = category
                logger.info(f"临时分类对象创建完成，共 {len(self.categories)} 个")
            else:
                error_msg = f"无法加载索引: 目录 {directory} 中没有有效的索引文件"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # 记录总耗时
        total_time = time.time() - start_time
        logger.info(f"索引加载完成，总耗时: {total_time:.2f}秒，加载了 {len(self.categories)} 个类别和 {self.index.ntotal} 个向量")
