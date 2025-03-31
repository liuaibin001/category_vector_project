"""向量生成模块."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import torch
import time
import concurrent.futures
from multiprocessing import cpu_count
from tqdm import tqdm

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
        # 组合文本信息，但分配不同权重
        path_text = category.path
        description = category.description or ""
        keywords = " ".join(category.keywords) if category.keywords else ""
        examples = " ".join(category.examples) if category.examples else ""
        
        # 从category.path中提取最后一级作为核心产品类型
        core_product_type = category.levels[-1] if category.levels else ""
        
        # 构建加权文本组合
        weighted_texts = [
            path_text,                       # 基础权重为1
            core_product_type * 3,           # 最后一级分类(核心产品类型)加强权重3倍
            description,                     # 基础权重为1
            keywords * 2,                    # 关键词加强权重2倍
            examples                         # 基础权重为1
        ]
        
        # 合并所有文本
        combined_text = " ".join(filter(None, weighted_texts))
        
        logger.debug(f"为分类ID={category.id}生成完整向量，文本长度={len(combined_text)}字符")
        logger.debug(f"核心产品类型: {core_product_type}")
        
        # 生成向量
        start_time = time.time()
        vector = self.model.encode(combined_text)
        encode_time = time.time() - start_time
        
        # 归一化向量，为余弦相似度做准备
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
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
        
        # 改进产品类型词识别逻辑
        important_terms = []
        
        # 查找可能的产品类型词
        parts = query.split()
        
        # 1. 查找末尾词 - 很多查询的产品类型在末尾，如"iPad 保护壳"
        if len(parts) >= 1:
            last_word = parts[-1]
            if len(last_word) >= 2:
                important_terms.append(last_word)
                
        # 2. 检查常见产品类型词
        common_product_types = ["壳", "保护壳", "支架", "平板壳", "手机壳", "套", "保护套", 
                              "贴膜", "充电器", "数据线", "电源", "配件", "膜"]
        
        for product_type in common_product_types:
            if product_type in query:
                important_terms.append(product_type)
        
        # 3. 扫描所有中间词，更全面地捕获潜在的产品类型词
        if len(parts) >= 3:
            potential_type_words = parts[1:-1]  # 中间位置的词
            for word in potential_type_words:
                if len(word) >= 2:  # 过滤太短的词
                    important_terms.append(word)
        
        # 如果没有找到合适的词，就用整个查询
        if not important_terms:
            important_terms = [query]
            
        # 构建加权查询文本 - 增加产品类型词的权重
        weighted_query = f"{query} {' '.join(important_terms) * 5}"
        logger.debug(f"加权查询文本: '{weighted_query}'")
        
        start_time = time.time()
        vector = self.model.encode(weighted_query)
        encode_time = time.time() - start_time
        
        # 归一化向量，为余弦相似度做准备
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
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
    
    def batch_enrich_category_vectors(self, categories: List[Category], max_workers: int = None) -> List[Category]:
        """并行为多个分类对象生成所有向量
        
        Args:
            categories: 分类对象列表
            max_workers: 最大并行工作线程数，None表示使用默认值(CPU核心数*5)
            
        Returns:
            添加了向量的分类对象列表
        """
        if not categories:
            return []
            
        # 如果未指定线程数，使用CPU核心数*5作为默认值
        if max_workers is None:
            max_workers = min(32, cpu_count() * 5)  # 最多32个线程
            
        start_time = time.time()
        total_count = len(categories)
        
        logger.info(f"开始使用{max_workers}个并行工作线程为{total_count}个类别批量生成向量...")
        
        # 使用线程池执行并行处理
        enriched_categories = []
        
        # 创建一个进度条
        with tqdm(total=total_count, desc="生成向量", unit="类别") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_category = {
                    executor.submit(self.enrich_category_vectors, category): category 
                    for category in categories
                }
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_category):
                    original_category = future_to_category[future]
                    try:
                        enriched_category = future.result()
                        enriched_categories.append(enriched_category)
                        # 更新进度条而不是打印日志
                        pbar.update(1)
                        pbar.set_postfix({"完成": f"{pbar.n/pbar.total*100:.1f}%"})
                    except Exception as e:
                        logger.error(f"为分类 ID={original_category.id} 生成向量时出错: {e}")
                        # 即使出错也更新进度条
                        pbar.update(1)
                    
        # 只在完成时输出一条日志
        total_time = time.time() - start_time
        logger.info(f"批量向量生成完成，耗时: {total_time:.2f}秒，平均每类耗时: {total_time/total_count:.4f}秒")
        
        return enriched_categories
