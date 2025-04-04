"""向量存储模块."""

import os
import json
import pickle
import numpy as np
# 替换FAISS为Milvus
# import faiss
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm

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
        self.categories: Dict[int, Category] = {}
        
        # 设置集合名称
        self.collection_name = "category_vectors"
        if self.config and hasattr(self.config, "collection_name"):
            self.collection_name = self.config.collection_name
        
        # 初始化集合
        self.collection = None
        
    def check_milvus_connection(self, host: str, port: str) -> bool:
        """检查 Milvus 服务器连通性
        
        Args:
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            
        Returns:
            bool: 是否连接成功
        """
        try:
            # 如果已有连接，先断开
            try:
                connections.disconnect("default")
            except:
                pass
                
            # 尝试建立连接
            connections.connect("default", host=host, port=port)
            logger.info(f"Milvus连接成功: {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Milvus连接失败: {host}:{port}, 错误: {e}")
            return False
            
    def connect_to_milvus(self):
        """连接到Milvus服务器"""
        # 从配置中获取连接信息
        host = self.config.milvus_host if self.config and hasattr(self.config, "milvus_host") else "localhost"
        port = self.config.milvus_port if self.config and hasattr(self.config, "milvus_port") else "19530"
        
        logger.info(f"尝试连接到Milvus服务器: {host}:{port}")
        
        # 检查连接
        if not self.check_milvus_connection(host, port):
            raise ConnectionError(f"无法连接到Milvus服务器: {host}:{port}")
        
    def setup_collection(self):
        """创建或获取Milvus集合"""
        # 检查集合是否存在
        if utility.has_collection(self.collection_name):
            logger.info(f"集合已存在: {self.collection_name}")
            self.collection = Collection(self.collection_name)
        else:
            logger.info(f"创建新集合: {self.collection_name}")
            # 定义字段
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="category_id", dtype=DataType.INT64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields=fields, description="Category vectors collection")
            
            # 创建集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引 - 修改为余弦相似度(IP)
            index_params = {
                "metric_type": "IP",  # 使用IP (Inner Product)，用于余弦相似度
                "index_type": "FLAT"  # 默认使用FLAT索引
            }
            
            # 根据配置选择不同索引类型
            if self.config and self.config.index_type:
                if self.config.index_type.lower() == "ivf":
                    index_params["index_type"] = "IVF_FLAT"
                    index_params["params"] = {"nlist": self.config.nlist}
                elif self.config.index_type.lower() == "hnsw":
                    index_params["index_type"] = "HNSW"
                    index_params["params"] = {"M": self.config.m_factor, "efConstruction": 200}
            
            # 创建向量字段的索引
            try:
                self.collection.create_index(field_name="vector", index_params=index_params)
                logger.info(f"成功创建索引: {index_params}")
            except Exception as e:
                logger.error(f"创建索引失败: {e}")
                raise  # 确保索引创建失败时抛出异常
        
        # 确保集合存在并有索引后，再加载到内存
        if self.collection:
            try:
                self.collection.load()
                # 验证集合是否已成功加载
                entities_count = self.collection.num_entities
                # 如果发现集合为空但categories中有数据，发出警告
                if entities_count == 0 and len(self.categories) > 0:
                    logger.warning(f"集合 {self.collection_name} 加载成功，但不包含向量。将无法执行搜索操作。")
                    logger.warning("请确保成功执行build命令并向集合中添加了数据。")
            except Exception as e:
                logger.error(f"加载集合失败: {e}")
                raise  # 确保加载失败时抛出异常
        
    def add_category(self, category: Category):
        """添加分类到索引
        
        Args:
            category: 分类对象
        """
        if category.vector is None:
            raise ValueError("Category must have a vector")
        
        # 确保集合已初始化
        if self.collection is None:
            self.setup_collection()
            
        try:
            # 首先检查是否已存在该分类ID的数据
            category_id = int(category.id)
            
            # 查询现有数据
            if self.collection.num_entities > 0:
                try:
                    # 构建查询条件
                    expr = f"category_id == {category_id}"
                    result = self.collection.query(
                        expr=expr,
                        output_fields=["pk", "category_id"]
                    )
                    
                    # 如果找到匹配的记录，先删除
                    if result and len(result) > 0:
                        logger.debug(f"发现已存在的分类 ID={category_id}，将进行覆盖")
                        # 获取主键ID用于删除
                        pk_to_delete = [r["pk"] for r in result]
                        # 删除现有记录
                        self.collection.delete(f"pk in {pk_to_delete}")
                        # 强制执行刷新，确保删除生效
                        self.collection.flush()
                        # 验证删除是否成功
                        verify_result = self.collection.query(expr=expr, output_fields=["pk"])
                        if not verify_result:
                            logger.debug(f"已成功删除分类 ID={category_id} 的现有记录")
                        else:
                            logger.warning(f"分类 ID={category_id} 的记录可能未完全删除")
                except Exception as e:
                    logger.warning(f"检查或删除分类 ID={category_id} 时出错: {e}")
            
            # 标准化向量以准备余弦相似度计算
            vector = category.vector.astype('float32')
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vector = vector / norm
            else:
                normalized_vector = vector
                
            # 检查向量维度是否正确
            if len(normalized_vector) != self.dimension:
                logger.error(f"向量维度不匹配：分类 ID={category_id} 的向量维度为 {len(normalized_vector)}，但集合需要 {self.dimension}")
                raise ValueError(f"Vector dimension mismatch: {len(normalized_vector)} vs {self.dimension}")
                
            # 插入到Milvus - 使用归一化的向量
            logger.info(f"正在将分类 ID={category_id} 插入Milvus，向量维度={len(normalized_vector)}")
            insert_result = self.collection.insert([
                {"category_id": category_id, 
                 "vector": normalized_vector.tolist()}
            ])
            
            # 验证插入是否成功
            if hasattr(insert_result, 'insert_count') and insert_result.insert_count > 0:
                logger.info(f"成功插入分类 ID={category_id}，插入数量: {insert_result.insert_count}")
            else:
                logger.warning(f"插入分类 ID={category_id} 后没有收到确认，可能未成功")
            
            # 强制执行刷新，确保数据被保存
            try:
                self.collection.flush()
                logger.debug(f"成功刷新集合，确保数据持久化")
            except Exception as e:
                logger.warning(f"刷新集合时出错: {e}")
            
            # 验证插入后的实体数
            try:
                entity_count = self.collection.num_entities
                logger.info(f"当前集合实体数: {entity_count}")
            except Exception as e:
                logger.warning(f"获取集合实体数时出错: {e}")
            
            # 保存分类对象
            self.categories[category.id] = category
            
            # 增加日志
            logger.debug(f"成功添加分类 ID={category.id} 到Milvus集合")
            
        except Exception as e:
            logger.error(f"添加分类 ID={category.id} 到Milvus失败: {e}")
            raise
        
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
        # 延迟导入 redis_client，避免循环导入
        from categoryvector.utils.redis_client import redis_client
        
        # 确保集合已初始化
        if self.collection is None:
            self.setup_collection()
            
        # 获取集合中实体数量
        entity_count = self.collection.num_entities
        logger.info(f"开始搜索，集合包含 {entity_count} 个向量")
        print(f"开始搜索，集合包含 {entity_count} 个向量")
        print(f"请求返回 {top_k} 个结果，相似度阈值 {threshold}")
        
        # 实体数为0时给出明确错误
        if entity_count == 0:
            logger.error("集合中没有向量数据，无法执行搜索。请先运行build命令添加数据。")
            return []
            
        # 确保查询向量是正确的形状和类型
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(-1)
        query_vector = query_vector.astype('float32')
        
        # 标准化查询向量用于余弦相似度
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
            
        # 默认使用IP作为搜索参数（余弦相似度）
        search_params = {"metric_type": "IP"}
        logger.info(f"搜索使用度量类型: IP (余弦相似度)")
        
        # 尝试查看索引信息
        try:
            index_dict = self.collection.index()
            if index_dict and "_index_params" in dir(index_dict) and "params" in index_dict._index_params:
                metric_type = index_dict._index_params["params"].get("metric_type")
                if metric_type:
                    search_params["metric_type"] = metric_type
                    logger.info(f"使用索引度量类型: {metric_type}")
        except Exception as e:
            logger.debug(f"无法获取详细索引信息: {e}, 使用默认IP度量")
        
        # 根据配置调整搜索参数
        if self.config and self.config.index_type:
            if self.config.index_type.lower() == "ivf":
                search_params["params"] = {"nprobe": min(self.config.nlist // 4, 10)}
            elif self.config.index_type.lower() == "hnsw":
                search_params["params"] = {"ef": 50}
        
        # 搜索向量 - 获取更多结果用于过滤
        fetch_k = min(top_k * 3, entity_count)
        
        try:
            logger.debug(f"执行搜索，参数: fetch_k={fetch_k}, 搜索参数={search_params}")
            # 首先尝试IP搜索
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=fetch_k,
                output_fields=["category_id"]
            )
            logger.debug(f"搜索完成，获取到 {len(results[0])} 个初始结果")
        except Exception as e:
            logger.error(f"IP搜索失败: {e}")
            # 如果IP失败，尝试L2
            try:
                logger.info("尝试使用L2度量搜索")
                search_params["metric_type"] = "L2"
                results = self.collection.search(
                    data=[query_vector.tolist()],
                    anns_field="vector",
                    param=search_params,
                    limit=fetch_k,
                    output_fields=["category_id"]
                )
            except Exception as e2:
                logger.error(f"L2度量搜索也失败: {e2}")
                return []
        
        # 处理结果
        category_results = []
        metric_type = search_params["metric_type"]
        filtered_hits = []
        
        for hits in results:
            for hit in hits:
                category_id = hit.entity.get('category_id')
                
                # 排除特定ID
                if exclude_ids and category_id in exclude_ids:
                    continue
                
                # 根据metric_type计算相似度分数
                if metric_type == "IP":
                    # 对于IP (余弦相似度)，分数直接就是相似度，范围是[0,1]
                    similarity = hit.score
                else:
                    # 对于L2距离，需要转换为相似度分数
                    distance = hit.distance
                    # 使用指数衰减函数，与原FAISS中相同的转换方式
                    similarity = np.exp(-distance / 10.0)
                
                # 应用阈值
                if similarity >= threshold:
                    filtered_hits.append((category_id, similarity))
        
        print(f"过滤后找到 {len(filtered_hits)} 个结果")
        
        # 从Redis中获取详细信息
        if redis_client.client is not None:
            for category_id, similarity in filtered_hits:
                redis_key = f"categories:{category_id}"
                category_data = redis_client.get(redis_key)
                
                if category_data:
                    try:
                        # 从Redis数据创建Category对象
                        category = Category.from_dict(category_data)
                        category_results.append((category, similarity))
                    except Exception as e:
                        logger.warning(f"从Redis加载类别ID={category_id}失败: {e}")
                        # 尝试从本地缓存获取
                        if category_id in self.categories:
                            category_results.append((self.categories[category_id], similarity))
                elif category_id in self.categories:
                    # 如果Redis没有，但本地缓存有
                    category_results.append((self.categories[category_id], similarity))
        else:
            # Redis不可用，使用本地缓存
            logger.warning("Redis不可用，使用本地缓存数据")
            for category_id, similarity in filtered_hits:
                if category_id in self.categories:
                    category_results.append((self.categories[category_id], similarity))
        
        # 按相似度降序排序
        category_results.sort(key=lambda x: x[1], reverse=True)
        
        # 只返回前top_k个结果
        final_results = category_results[:top_k]
        
        print(f"组装后返回 {len(final_results)} 个结果")
        
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
        # 延迟导入 redis_client，避免循环导入
        from categoryvector.utils.redis_client import redis_client
        
        # 确保查询向量是正确的形状
        if query_vector.ndim != 1:
            query_vector = query_vector.reshape(-1)
            
        # 检查Redis连接是否可用
        if redis_client.client is None:
            print("注意: Redis不可用，无法按层级搜索。正在回退到使用本地缓存搜索。")
            if not self.categories:
                print("错误: 本地缓存中没有分类数据，无法执行按层级搜索")
                return []
                
            print(f"开始按层级 {level} 搜索，分类数量 {len(self.categories)}")
            print(f"请求返回 {top_k} 个结果，相似度阈值 {threshold}")
            return self._search_by_level_local(query_vector, level, top_k, threshold)
        
        # 从Redis获取所有类别信息
        print("正在从Redis获取分类数据...")
        all_keys = redis_client.client.keys("categories:*")
        if not all_keys:
            print("Redis中未找到分类数据，无法执行按层级搜索")
            if self.categories:
                print("正在回退到使用本地缓存搜索...")
                return self._search_by_level_local(query_vector, level, top_k, threshold)
            return []
            
        # 从Redis获取并处理每个类别
        results = []
        candidates_count = 0
        
        for key in all_keys:
            try:
                # 获取类别数据
                category_data = redis_client.get(key)
                if not category_data:
                    continue
                    
                # 检查层级是否符合要求
                if category_data.get('level_depth', 0) < level:
                    continue
                    
                candidates_count += 1
                
                # 创建Category对象
                category = Category.from_dict(category_data)
                
                # 检查是否有层级向量
                level_vector_key = f"level_{level}"
                if not category.level_vectors or level_vector_key not in category.level_vectors:
                    continue
                    
                # 获取层级向量
                level_vector = category.level_vectors.get(level_vector_key)
                
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
            except Exception as e:
                print(f"处理Redis分类数据时出错: {e}")
                continue
                
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
        
    def _search_by_level_local(
        self,
        query_vector: np.ndarray,
        level: int,
        top_k: int = 10,
        threshold: float = 0.6
    ) -> List[Tuple[Category, float]]:
        """使用本地缓存按层级搜索分类
        
        Args:
            query_vector: 查询向量
            level: 目标层级
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            (分类对象, 相似度)元组的列表，按相似度降序排序
        """
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
        print(f"正在保存数据到目录: {directory}")
        
        # 创建目录
        directory.mkdir(parents=True, exist_ok=True)
        
        # 保存Milvus配置信息
        config_path = directory / "milvus_config.json"
        
        # 从配置中获取连接信息
        host = self.config.milvus_host if self.config and hasattr(self.config, "milvus_host") else "localhost"
        port = self.config.milvus_port if self.config and hasattr(self.config, "milvus_port") else "19530"
        
        milvus_config = {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "host": host,
            "port": port
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(milvus_config, f, ensure_ascii=False, indent=2)
            
        print(f"- Milvus配置: 主机={host}, 端口={port}, 集合={self.collection_name}")
        
        # 保存分类数据为JSON格式
        categories_path = directory / "categories.json"
        print(f"正在序列化 {len(self.categories)} 个类别...")
        
        # 使用进度条显示序列化进度
        categories_data = []
        with tqdm(total=len(self.categories), desc="序列化类别", unit="个", colour="green") as pbar:
            for i, cat in enumerate(self.categories.values()):
                try:
                    cat_dict = cat.to_dict()
                    categories_data.append(cat_dict)
                except Exception as e:
                    print(f"❌ 序列化类别 ID={cat.id} 失败: {e}")
                finally:
                    pbar.update(1)
        
        # 写入JSON文件
        print("正在写入数据到文件...")
        json_write_start = time.time()
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(categories_data, f, ensure_ascii=False, indent=2)
        json_write_time = time.time() - json_write_start
        
        categories_size = categories_path.stat().st_size / (1024 * 1024)  # MB
        total_time = time.time() - start_time
        
        print(f"✓ 保存完成:")
        print(f"  - 类别数量: {len(categories_data)}")
        print(f"  - 文件大小: {categories_size:.2f} MB")
        print(f"  - 写入耗时: {json_write_time:.2f}秒")
        print(f"  - 总耗时: {total_time:.2f}秒")
    
    def load(self, directory: Path):
        """加载索引和分类数据
        
        Args:
            directory: 数据目录
        """
        start_time = time.time()
        print(f"正在从目录加载数据: {directory}")
        
        directory = Path(directory)
        
        # 加载Milvus配置
        config_file = directory / "milvus_config.json"
        if config_file.exists():
            print(f"找到Milvus配置文件...")
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # 更新Milvus连接信息
            self.collection_name = config_data.get("collection_name", "category_vectors")
            self.dimension = config_data.get("dimension", self.dimension)
            host = config_data.get("host", "localhost")
            port = config_data.get("port", "19530")
            
            # 重新连接Milvus
            try:
                print(f"正在连接到Milvus: {host}:{port}...")
                connections.disconnect("default")  # 断开现有连接
                connections.connect(
                    "default", 
                    host=host, 
                    port=port
                )
                print(f"✓ Milvus连接成功")
            except Exception as e:
                print(f"✗ 连接Milvus失败: {e}")
                raise
        
        # 获取或创建集合
        self.setup_collection()
        
        # 加载分类数据
        categories_file = directory / "categories.json"
        if categories_file.exists():
            categories_size = categories_file.stat().st_size / (1024 * 1024)  # MB
            print(f"正在读取类别数据文件 ({categories_size:.2f} MB)...")
            
            with open(categories_file, "r", encoding="utf-8") as f:
                categories_data = json.load(f)
            
            total_categories = len(categories_data)
            print(f"找到 {total_categories} 个类别")
            
            # 开始转换类别数据
            categories_count = 0
            
            if isinstance(categories_data, list):
                # 数组格式 [{}, {}, ...]
                print("正在处理数组格式的类别数据...")
                
                # 使用改进的进度条
                with tqdm(total=total_categories, desc="加载类别", unit="类别", colour="blue", dynamic_ncols=True) as pbar:
                    for i, cat_data in enumerate(categories_data):
                        try:
                            category = Category.from_dict(cat_data)
                            self.categories[category.id] = category
                            categories_count += 1
                            pbar.update(1)
                            pbar.set_postfix({"完成": f"{(i+1)/total_categories*100:.1f}%"})
                        except Exception as e:
                            pbar.set_description(f"❌ ID={cat_data.get('id', '未知')} 出错")
                            pbar.update(1)
            else:
                # 字典格式 {"1": {}, "2": {}, ...}
                print("正在处理字典格式的类别数据...")
                
                # 使用改进的进度条
                items = list(categories_data.items())
                with tqdm(total=len(items), desc="加载类别", unit="类别", colour="blue", dynamic_ncols=True) as pbar:
                    for i, (cat_id, cat_data) in enumerate(items):
                        try:
                            category = Category.from_dict(cat_data)
                            self.categories[int(cat_id)] = category
                            categories_count += 1
                            pbar.update(1)
                            pbar.set_postfix({"完成": f"{(i+1)/len(items)*100:.1f}%"})
                        except Exception as e:
                            pbar.set_description(f"❌ ID={cat_id} 出错")
                            pbar.update(1)
            
            total_time = time.time() - start_time
            print(f"✓ 加载完成:")
            print(f"  - 类别数量: {categories_count}/{total_categories}")
            print(f"  - 集合实体数: {self.collection.num_entities if self.collection else 0}")
            print(f"  - 总耗时: {total_time:.2f}秒")
        else:
            print(f"✗ 未找到类别数据文件: {categories_file}")
            raise FileNotFoundError(f"类别数据文件不存在: {categories_file}")

    def batch_add_categories(self, categories: List[Category], batch_size: int = 100):
        """批量添加分类到索引
        
        Args:
            categories: 分类对象列表
            batch_size: 每批次处理的数量
        """
        if not categories:
            logger.warning("没有类别需要添加")
            return
            
        # 确保集合已初始化
        if self.collection is None:
            self.setup_collection()
            
        categories_to_add = []
        for category in categories:
            if category.vector is None:
                logger.warning(f"类别 ID={category.id} 没有向量，已跳过")
                continue
                
            # 标准化向量以准备余弦相似度计算
            vector = category.vector.astype('float32')
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vector = vector / norm
            else:
                normalized_vector = vector
                
            # 检查向量维度是否正确
            if len(normalized_vector) != self.dimension:
                logger.warning(f"向量维度不匹配：分类 ID={category.id} 的向量维度为 {len(normalized_vector)}，但集合需要 {self.dimension}，已跳过")
                continue
                
            # 添加到待插入列表
            categories_to_add.append({
                "category_id": int(category.id),
                "vector": normalized_vector.tolist()
            })
            
            # 保存分类对象
            self.categories[category.id] = category
                
        # 按批次插入数据
        total_count = len(categories_to_add)
        if total_count == 0:
            logger.warning("没有有效的类别需要添加")
            return
        
        total_inserted = 0
        
        # 使用蓝色进度条显示批量插入进度
        with tqdm(total=total_count, desc="添加向量", unit="类别", colour="blue", dynamic_ncols=True) as pbar:
            for i in range(0, total_count, batch_size):
                batch = categories_to_add[i:i+batch_size]
                batch_count = len(batch)
                
                try:
                    # 批量插入到 Milvus
                    insert_result = self.collection.insert(batch)
                    
                    # 验证插入是否成功
                    if hasattr(insert_result, 'insert_count') and insert_result.insert_count > 0:
                        successful = insert_result.insert_count
                        total_inserted += successful
                        pbar.update(successful)
                        pbar.set_postfix({"完成": f"{total_inserted}/{total_count}", "批次": f"{i//batch_size + 1}/{(total_count-1)//batch_size + 1}"})
                    else:
                        pbar.set_description("⚠️ 插入未确认")
                        
                except Exception as e:
                    pbar.set_description(f"❌ 批次 {i//batch_size + 1} 出错")
                    logger.error(f"批量插入类别时出错: {e}")
                
        # 最后执行一次刷新，确保所有数据被保存
        try:
            self.collection.flush()
            print(f"✓ 成功保存 {total_inserted}/{total_count} 个类别到 Milvus 集合")
        except Exception as e:
            logger.warning(f"刷新集合时出错: {e}")
            
        # 验证插入后的实体数
        try:
            entity_count = self.collection.num_entities
            print(f"- 当前集合实体数: {entity_count}")
        except Exception:
            pass

    def save_to_redis(self):
        """保存分类数据到 Redis
        
        Returns:
            bool: 是否保存成功
        """
        # 延迟导入 redis_client，避免循环导入
        from categoryvector.utils.redis_client import redis_client
        
        # 检查 Redis 客户端是否可用
        if redis_client.client is None:
            print("✗ Redis 连接不可用，无法保存数据")
            return False
            
        start_time = time.time()
        print("正在保存分类数据到 Redis...")
        
        try:
            # 遍历所有分类并保存到 Redis
            total_count = len(self.categories)
            success_count = 0
            
            for category_id, category in self.categories.items():
                try:
                    redis_key = f"categories:{category_id}"
                    # 将分类对象转换为字典
                    category_dict = {
                        "id": category.id,
                        "path": category.path,
                        "levels": category.levels,
                        "level_depth": category.level_depth,
                        "description": category.description,
                        "keywords": category.keywords,
                        "examples": category.examples,
                        "exclusions": category.exclusions,
                        "vector": category.vector.tolist() if hasattr(category, 'vector') and category.vector is not None else None
                    }
                    redis_client.set(redis_key, category_dict, 60*60*24*30)
                    success_count += 1
                except Exception as e:
                    print(f"❌ 保存分类 ID={category_id} 到 Redis 失败: {e}")
            
            save_time = time.time() - start_time
            print(f"✓ 成功保存 {success_count}/{total_count} 个分类到 Redis，耗时 {save_time:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"✗ 保存到 Redis 失败: {str(e)}")
            return False
