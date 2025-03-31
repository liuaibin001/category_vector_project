#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import shutil
from tqdm import tqdm
import json
import time

import numpy as np
from pymilvus import utility, connections

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryProcessor
from categoryvector.vector_generation import VectorGenerator
from categoryvector.vector_storage import VectorStorage
from categoryvector.utils.logging_utils import setup_logger, default_logger as logger
from categoryvector.utils.redis_client import redis_client
from categoryvector.models import Category


def parse_build_args(args=None):
    """解析构建索引命令参数"""
    parser = argparse.ArgumentParser(description="构建类别向量索引")
    parser.add_argument(
        "--categories",
        "-c",
        required=True,
        help="类别数据JSON文件路径"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        help="输出索引目录，如未指定则使用配置文件中的设置"
    )
    parser.add_argument(
        "--vector-dim",
        "-d",
        type=int,
        default=None,
        help="向量维度，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="向量模型名称，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出（自动设置日志级别为DEBUG）"
    )
    # 添加并行处理参数
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="并行处理的工作线程数，默认为CPU核心数*5"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="批量插入Milvus的批次大小，默认为100"
    )
    # 添加Milvus相关参数
    parser.add_argument(
        "--milvus-host",
        default=None,
        help="Milvus服务器地址，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--milvus-port",
        default=None,
        help="Milvus服务器端口，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Milvus集合名称，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default=None,
        help="索引类型: flat, ivf, hnsw，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--config",
        help="配置文件路径，默认为项目根目录的config.toml"
    )
    
    return parser.parse_args(args)


def parse_search_args(args=None):
    """解析搜索命令参数"""
    parser = argparse.ArgumentParser(description="搜索类别")
    parser.add_argument(
        "--index",
        "-i",
        required=True,
        help="索引目录路径"
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="搜索查询文本"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="返回结果数量，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="相似度阈值，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--level",
        "-l",
        type=int,
        help="指定搜索层级"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出（自动设置日志级别为DEBUG）"
    )
    parser.add_argument(
        "--milvus-host",
        default=None,
        help="Milvus服务器地址，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--milvus-port",
        default=None,
        help="Milvus服务器端口，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Milvus集合名称，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--config",
        help="配置文件路径，默认为项目根目录的config.toml"
    )
    
    return parser.parse_args(args)


def parse_update_args(args=None):
    """解析更新分类命令参数"""
    parser = argparse.ArgumentParser(description="更新特定分类的向量")
    parser.add_argument(
        "--index",
        "-i",
        required=True,
        help="索引目录路径"
    )
    parser.add_argument(
        "--category-id",
        "-c",
        required=True,
        type=int,
        help="要更新的分类ID"
    )
    parser.add_argument(
        "--vector-dim",
        "-d",
        type=int,
        default=None,
        help="向量维度，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="向量模型名称，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日志级别，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出（自动设置日志级别为DEBUG）"
    )
    parser.add_argument(
        "--milvus-host",
        default=None,
        help="Milvus服务器地址，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--milvus-port",
        default=None,
        help="Milvus服务器端口，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Milvus集合名称，默认使用配置文件中的设置"
    )
    parser.add_argument(
        "--config",
        help="配置文件路径，默认为项目根目录的config.toml"
    )
    
    return parser.parse_args(args)


def build_index(categories_file: str, output_dir: Optional[str] = None, config: Optional[CategoryVectorConfig] = None):
    """构建索引.
    
    Args:
        categories_file: 分类数据文件路径
        output_dir: 输出目录
        config: 配置对象，可选
    """
    start_time = time.time()
    categories_file = Path(categories_file)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("data/vectors")
        
    print(f"开始构建索引，输入文件: {categories_file}")
    
    # 检查输入文件
    if not categories_file.exists():
        print(f"错误: 输入文件不存在: {categories_file}")
        return
    
    # 确保配置对象存在
    if config is None:
        config = CategoryVectorConfig()
    
    # 尝试连接 Milvus
    try:
        # 从配置中获取连接信息
        host = config.milvus_host
        port = config.milvus_port
        
        print(f"正在连接 Milvus 服务器: {host}:{port}...")
        try:
            # 断开可能存在的连接
            connections.disconnect("default")
        except:
            pass
            
        # 尝试建立连接
        connections.connect("default", host=host, port=port)
        print(f"✓ Milvus 连接成功")
    except Exception as e:
        print(f"✗ Milvus 连接失败: {host}:{port}, 错误: {e}")
        print("请检查 Milvus 服务器是否正在运行")
        return
        
    # 检查是否存在已生成的向量数据
    existing_vectors_path = Path("data/vectors")
    if existing_vectors_path.exists():
        print(f"发现已存在的向量数据目录，正在检查...")
        
        # 检查必要的文件是否存在
        categories_json = existing_vectors_path / "categories.json"
        milvus_config = existing_vectors_path / "milvus_config.json"
        
        if categories_json.exists() and milvus_config.exists():
            print("找到完整的向量数据文件，正在验证...")
            
            try:
                # 加载 Milvus 配置
                with open(milvus_config, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                
                # 验证配置是否匹配
                if (
                    str(config.milvus_port) != str(saved_config.get("port")) or
                    config.milvus_host != saved_config.get("host") or
                    config.collection_name != saved_config.get("collection_name")
                ):
                    print("✗ 现有配置与当前配置不匹配，将重新构建向量")
                else:
                    # 加载分类数据
                    print("正在加载现有类别数据...")
                    with open(categories_json, "r", encoding="utf-8") as f:
                        categories_data = json.load(f)
                    
                    # 转换为 Category 对象
                    categories = []
                    
                    with tqdm(total=len(categories_data), desc="加载分类", unit="个") as pbar:
                        for cat_data in categories_data:
                            try:
                                category = Category.from_dict(cat_data)
                                categories.append(category)
                            except Exception:
                                pass  # 静默忽略错误
                            finally:
                                pbar.update(1)
                    
                    if categories:
                        print(f"✓ 成功加载 {len(categories)} 个现有类别")
                        
                        # 初始化向量存储
                        storage = VectorStorage(
                            dimension=config.vector_dim,
                            config=config
                        )
                        
                        # 检查并删除现有集合
                        collection_name = config.collection_name
                        try:
                            if utility.has_collection(collection_name):
                                print(f"正在删除现有 Milvus 集合: {collection_name}...")
                                utility.drop_collection(collection_name)
                        except Exception:
                            pass
                              
                        # 批量添加到 Milvus
                        print("正在将类别添加到 Milvus...")
                        storage.batch_add_categories(categories)
                        
                        # 保存到新的输出目录
                        print(f"正在保存到输出目录: {output_dir}...")
                        storage.save(output_dir)
                        
                        # 更新 Redis
                        print("正在更新 Redis...")
                        redis_success = storage.save_to_redis()
                        if redis_success:
                            print("✓ Redis 更新成功")
                        else:
                            print("✗ Redis 更新失败")
                        
                        total_time = time.time() - start_time
                        print(f"\n构建索引完成:")
                        print(f"- 总类别数量: {len(storage.categories)}")
                        print(f"- 向量维度: {config.vector_dim}")
                        print(f"- Milvus主机: {config.milvus_host}:{config.milvus_port}")
                        print(f"- 总耗时: {total_time:.2f}秒")
                        return
                    else:
                        print("✗ 没有找到有效的类别数据，将重新构建向量")
                        
            except Exception as e:
                print(f"✗ 加载现有向量数据时出错: {str(e)}")
                print("将重新构建向量")
    
    # 如果没有现有数据或加载失败，重新构建向量
    print("\n开始重新构建向量...\n")
    
    # 加载分类数据
    print("正在加载分类数据...")
    processor = CategoryProcessor(config)
    processor.load_from_json(categories_file)
    categories = list(processor.categories.values())
    
    if not categories:
        print("错误: 没有找到有效的分类数据")
        return
        
    print(f"✓ 成功加载 {len(categories)} 个分类")
    
    # 生成向量
    print(f"\n正在加载模型: {config.model_name}...")
    generator = VectorGenerator(
        model_name=config.model_name,
        config=config
    )
    
    # 批量生成向量
    print("\n开始生成向量:")
    enriched_categories = generator.batch_enrich_category_vectors(categories)
    print(f"✓ 完成 {len(enriched_categories)} 个分类的向量生成")
    
    # 初始化向量存储
    storage = VectorStorage(
        dimension=config.vector_dim,
        config=config
    )
    
    # 检查并删除现有集合
    collection_name = config.collection_name
    try:
        if utility.has_collection(collection_name):
            print(f"正在删除现有 Milvus 集合: {collection_name}...")
            utility.drop_collection(collection_name)
    except Exception:
        pass
    
    # 批量添加到 Milvus
    print("\n正在添加向量到 Milvus...")
    storage.batch_add_categories(enriched_categories)
    
    # 保存到输出目录
    print(f"\n正在保存到目录: {output_dir}...")
    storage.save(output_dir)
    
    # 更新 Redis
    print("\n正在更新 Redis...")
    redis_success = storage.save_to_redis()
    if redis_success:
        print("✓ Redis 更新成功")
    else:
        print("✗ Redis 更新失败")
    
    total_time = time.time() - start_time
    print(f"\n构建索引完成:")
    print(f"- 总类别数量: {len(storage.categories)}")
    print(f"- 向量维度: {config.vector_dim}")
    print(f"- Milvus主机: {config.milvus_host}:{config.milvus_port}")
    print(f"- 总耗时: {total_time:.2f}秒")


def search(args=None):
    """搜索类别"""
    if args is None:
        args = parse_search_args()
    
    # 加载配置文件
    config = CategoryVectorConfig.from_toml(args.config if hasattr(args, 'config') else None)
    
    # 处理详细日志模式
    if hasattr(args, 'verbose') and args.verbose:
        args.log_level = "DEBUG"
        
    # 命令行参数覆盖配置文件
    log_level = args.log_level or config.log_level
    top_k = args.top_k or config.top_k
    threshold = args.threshold or config.similarity_threshold
    milvus_host = args.milvus_host or config.milvus_host
    milvus_port = args.milvus_port or config.milvus_port
    collection_name = args.collection_name or config.collection_name
    
    # 设置日志
    logger = setup_logger("categoryvector", level=log_level)
    
    logger.info(f"开始搜索过程...")
    
    try:
        # 检查索引目录
        index_path = Path(args.index)
        if not index_path.exists():
            logger.error(f"索引目录不存在: {index_path}")
            sys.exit(1)
            
        # 创建配置
        search_config = CategoryVectorConfig(
            model_name=config.model_name,
            data_dir=index_path,
            log_level=log_level,
            # 添加Milvus配置
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            top_k=top_k,
            similarity_threshold=threshold
        )
        
        # 创建存储实例并检查 Milvus 连通性
        logger.info("检查 Milvus 服务器连通性...")
        storage = VectorStorage(config.vector_dim, search_config)
        try:
            storage.connect_to_milvus()
        except ConnectionError as e:
            logger.error(f"Milvus 服务器连接失败: {e}")
            logger.error("请检查 Milvus 服务器是否正在运行，以及网络连接是否正常")
            sys.exit(1)
        
        # 初始化Milvus集合
        logger.info("初始化Milvus集合...")
        try:
            storage.setup_collection()
        except Exception as e:
            logger.error(f"初始化Milvus集合失败: {e}")
            sys.exit(1)
            
        # 检查集合中是否有数据
        entity_count = storage.collection.num_entities if storage.collection else 0
        if entity_count == 0:
            logger.error("Milvus集合中没有向量数据。请先运行build命令构建索引并确保数据成功保存到Milvus。")
            print("\n错误: Milvus集合为空，无法搜索。请先构建索引。")
            print("请运行: python -m src.categoryvector.cli build --categories your_categories.json")
            sys.exit(1)
            
        # 生成查询向量
        logger.info(f"查询: {args.query}")
        generator = VectorGenerator(model_name=search_config.model_name, config=search_config)
        query_vector = generator.generate_query_vector(args.query)
        
        # 执行搜索
        if args.level:
            # 按层级搜索
            results = storage.search_by_level(
                query_vector,
                level=args.level,
                top_k=top_k,
                threshold=threshold
            )
        else:
            # 全局搜索
            results = storage.search(
                query_vector,
                top_k=top_k,
                threshold=threshold
            )
        
        # 打印结果
        print(f"\n查询: {args.query}")
        print("=" * 70)
        print(f"找到 {len(results)} 个结果，显示前 {min(len(results), top_k)} 个（相似度阈值 {threshold}）")
        print("=" * 70)
        
        if not results:
            print("没有找到匹配的结果")
        else:
            # 打印每一行结果
            print(f"{'排名':<5}{'ID':<5}{'相似度':<10}{'路径':<20}{'描述':<30}")
            print("-" * 70)
            
            for i, (category, score) in enumerate(results):
                rank = f"{i+1}."
                
                # 截断过长的路径
                path = category.path
                if len(path) > 18:
                    path = path[:15] + "..."
                
                # 截断过长的描述
                description = category.description if category.description else ""
                if len(description) > 28:
                    description = description[:25] + "..."
                
                # 打印当前行
                print(f"{rank:<5}{category.id:<5}{score:.4f}    {path:<20}{description:<30}")
                
        print()
    
    except Exception as e:
        logger.exception(f"搜索时发生错误: {e}")
        sys.exit(1)


def update_category(args=None):
    """更新特定分类的向量"""
    start_time = time.time()
    
    if args is None:
        args = parse_update_args()
    
    # 加载配置文件
    config = CategoryVectorConfig.from_toml(args.config if hasattr(args, 'config') else None)
    
    # 处理详细日志模式
    if hasattr(args, 'verbose') and args.verbose:
        args.log_level = "DEBUG"
        
    # 命令行参数覆盖配置文件
    log_level = args.log_level or config.log_level
    model_name = args.model or config.model_name
    vector_dim = args.vector_dim or config.vector_dim
    milvus_host = args.milvus_host or config.milvus_host
    milvus_port = args.milvus_port or config.milvus_port
    collection_name = args.collection_name or config.collection_name
    
    # 设置日志
    logger = setup_logger("categoryvector", level=log_level)
    
    print(f"开始更新分类ID={args.category_id}的向量...")
    print(f"- 索引目录: {args.index}")
    print(f"- 分类ID: {args.category_id}")
    print(f"- 向量维度: {vector_dim}")
    print(f"- 模型: {model_name}")
    print(f"- Milvus配置: 主机={milvus_host}, 端口={milvus_port}, 集合={collection_name}")
    
    try:
        # 创建配置
        update_config = CategoryVectorConfig(
            model_name=model_name,
            data_dir=Path(args.index),
            log_level=log_level,
            vector_dim=vector_dim,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            nlist=config.nlist,
            m_factor=config.m_factor,
            index_type=config.index_type
        )
        
        # 尝试连接 Milvus
        try:
            print(f"正在连接 Milvus 服务器: {milvus_host}:{milvus_port}...")
            try:
                # 断开可能存在的连接
                connections.disconnect("default")
            except:
                pass
                
            # 尝试建立连接
            connections.connect("default", host=milvus_host, port=milvus_port)
            print(f"✓ Milvus 连接成功")
        except Exception as e:
            print(f"✗ Milvus 连接失败: {milvus_host}:{milvus_port}, 错误: {e}")
            print("请检查 Milvus 服务器是否正在运行")
            return
            
        # 创建存储实例
        storage = VectorStorage(dimension=vector_dim, config=update_config)
        
        # 加载索引目录
        index_path = Path(args.index)
        if not index_path.exists():
            print(f"错误: 索引目录不存在: {index_path}")
            return
            
        # 加载现有类别数据
        print(f"正在加载索引数据...")
        try:
            storage.load(index_path)
        except Exception as e:
            print(f"✗ 加载索引数据失败: {e}")
            return
            
        if not storage.categories:
            print("✗ 索引中没有分类数据")
            return
            
        # 查找要更新的分类
        category_id = args.category_id
        if category_id not in storage.categories:
            print(f"✗ 未找到分类ID={category_id}，无法更新")
            return
            
        category = storage.categories[category_id]
        print(f"✓ 找到分类: ID={category.id}, 路径={category.path}")
        
        # 重新生成向量
        print(f"\n正在加载模型: {update_config.model_name}...")
        generator = VectorGenerator(model_name=update_config.model_name, config=update_config)
        
        print(f"\n为分类ID={category.id}重新生成向量...")
        try:
            # 先清除旧向量
            category.vector = None
            category.level_vectors = {}
            
            # 生成新向量
            updated_category = generator.enrich_category_vectors(category)
            print(f"✓ 分类ID={category.id}的向量生成成功")
            
            # 更新到Milvus
            print(f"\n正在更新 Milvus 中的分类向量...")
            storage.add_category(updated_category)
            print(f"✓ Milvus 向量更新成功")
            
            # 保存更新后的数据
            print(f"\n正在保存到索引目录: {index_path}...")
            storage.save(index_path)
            print(f"✓ 索引数据保存完成")
            
            # 更新 Redis
            print("\n正在更新 Redis...")
            redis_key = f"categories:{category_id}"
            try:
                # 检查 Redis 客户端是否可用
                if redis_client.client is not None:
                    # 将分类对象转换为字典
                    category_dict = {
                        "id": updated_category.id,
                        "path": updated_category.path,
                        "levels": updated_category.levels,
                        "level_depth": updated_category.level_depth,
                        "description": updated_category.description,
                        "keywords": updated_category.keywords,
                        "examples": updated_category.examples,
                        "exclusions": updated_category.exclusions,
                        "vector": updated_category.vector.tolist() if hasattr(updated_category, 'vector') and updated_category.vector is not None else None,
                        "level_vectors": {
                            k: v.tolist() for k, v in updated_category.level_vectors.items()
                        } if updated_category.level_vectors else {}
                    }
                    redis_client.set(redis_key, category_dict, 60*60*24*30)
                    print("✓ Redis 更新成功")
                else:
                    print("✗ Redis 连接不可用，无法更新")
            except Exception as e:
                print(f"✗ Redis 更新失败: {e}")
            
            total_time = time.time() - start_time
            print(f"\n更新完成:")
            print(f"- 分类ID: {category.id}")
            print(f"- 分类路径: {category.path}")
            print(f"- 向量维度: {vector_dim}")
            print(f"- 总耗时: {total_time:.2f}秒")
            
        except Exception as e:
            print(f"✗ 更新分类向量时出错: {e}")
            return
    
    except Exception as e:
        print(f"✗ 更新分类时发生错误: {e}")
        return


def main():
    """主程序入口"""
    # 确定当前运行的命令名称
    program_name = Path(sys.argv[0]).name
    
    if program_name == "build":
        # 直接作为构建命令运行
        build_index()
    elif program_name == "search":
        # 直接作为搜索命令运行
        search()
    elif program_name == "update":
        # 直接作为更新命令运行
        update_category()
    else:
        # 作为主命令运行，需要解析子命令
        parser = argparse.ArgumentParser(description="类别向量工具")
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="日志级别"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="可用命令")
        
        # 构建索引子命令
        build_parser = subparsers.add_parser("build", help="构建类别向量索引")
        build_parser.add_argument(
            "--categories",
            "-c",
            required=True,
            help="类别数据JSON文件路径"
        )
        build_parser.add_argument(
            "--output",
            "-o",
            required=False,
            help="输出索引目录"
        )
        build_parser.add_argument(
            "--vector-dim",
            "-d",
            type=int,
            default=384,
            help="向量维度"
        )
        build_parser.add_argument(
            "--model",
            "-m",
            default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            help="向量模型名称"
        )
        build_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="启用详细日志输出（自动设置日志级别为DEBUG）"
        )
        
        # 添加并行处理参数到build子命令
        build_parser.add_argument(
            "--workers",
            "-w",
            type=int,
            default=None,
            help="并行处理的工作线程数，默认为CPU核心数*5"
        )
        build_parser.add_argument(
            "--batch-size",
            "-b",
            type=int,
            default=100,
            help="批量插入Milvus的批次大小，默认为100"
        )
        
        # 添加Milvus相关参数到build子命令
        build_parser.add_argument(
            "--milvus-host",
            default=None,
            help="Milvus服务器地址，默认使用配置文件中的设置"
        )
        build_parser.add_argument(
            "--milvus-port",
            default=None,
            help="Milvus服务器端口，默认使用配置文件中的设置"
        )
        build_parser.add_argument(
            "--collection-name",
            default=None,
            help="Milvus集合名称，默认使用配置文件中的设置"
        )
        build_parser.add_argument(
            "--index-type",
            choices=["flat", "ivf", "hnsw"],
            default="flat",
            help="索引类型: flat, ivf, hnsw"
        )
        
        # 搜索子命令
        search_parser = subparsers.add_parser("search", help="搜索类别")
        search_parser.add_argument(
            "--index",
            "-i",
            required=True,
            help="索引目录路径"
        )
        search_parser.add_argument(
            "--query",
            "-q",
            required=True,
            help="搜索查询文本"
        )
        search_parser.add_argument(
            "--top-k",
            "-k",
            type=int,
            default=None,
            help="返回结果数量，默认使用配置文件中的设置"
        )
        search_parser.add_argument(
            "--threshold",
            "-t",
            type=float,
            default=None,
            help="相似度阈值，默认使用配置文件中的设置"
        )
        search_parser.add_argument(
            "--level",
            "-l",
            type=int,
            help="指定搜索层级"
        )
        
        # 添加verbose参数
        search_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="启用详细日志输出（自动设置日志级别为DEBUG）"
        )
        
        # 添加Milvus相关参数到search子命令
        search_parser.add_argument(
            "--milvus-host",
            default=None,
            help="Milvus服务器地址，默认使用配置文件中的设置"
        )
        search_parser.add_argument(
            "--milvus-port",
            default=None,
            help="Milvus服务器端口，默认使用配置文件中的设置"
        )
        search_parser.add_argument(
            "--collection-name",
            default=None,
            help="Milvus集合名称，默认使用配置文件中的设置"
        )
        
        # 更新子命令
        update_parser = subparsers.add_parser("update", help="更新特定分类的向量")
        update_parser.add_argument(
            "--index",
            "-i",
            required=True,
            help="索引目录路径"
        )
        update_parser.add_argument(
            "--category-id",
            "-c",
            required=True,
            type=int,
            help="要更新的分类ID"
        )
        update_parser.add_argument(
            "--vector-dim",
            "-d",
            type=int,
            default=None,
            help="向量维度，默认使用配置文件中的设置"
        )
        update_parser.add_argument(
            "--model",
            "-m",
            default=None,
            help="向量模型名称，默认使用配置文件中的设置"
        )
        update_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="启用详细日志输出（自动设置日志级别为DEBUG）"
        )
        
        # 添加Milvus相关参数到update子命令
        update_parser.add_argument(
            "--milvus-host",
            default=None,
            help="Milvus服务器地址，默认使用配置文件中的设置"
        )
        update_parser.add_argument(
            "--milvus-port",
            default=None,
            help="Milvus服务器端口，默认使用配置文件中的设置"
        )
        update_parser.add_argument(
            "--collection-name",
            default=None,
            help="Milvus集合名称，默认使用配置文件中的设置"
        )
        
        args = parser.parse_args()
        
        # 设置日志
        logger = setup_logger("categoryvector", level=args.log_level)
        
        try:
            if args.command == "build":
                # 加载配置文件
                config = CategoryVectorConfig.from_toml(args.config if hasattr(args, 'config') else None)
                
                # 处理详细日志模式
                if args.verbose:
                    args.log_level = "DEBUG"
                    
                # 命令行参数覆盖配置文件
                log_level = args.log_level or config.log_level
                
                # 创建配置
                build_config = CategoryVectorConfig(
                    model_name=args.model or config.model_name,
                    data_dir=Path(args.categories).parent,
                    log_level=log_level,
                    vector_dim=args.vector_dim or config.vector_dim,
                    # 添加Milvus配置参数
                    milvus_host=args.milvus_host or config.milvus_host,
                    milvus_port=args.milvus_port or config.milvus_port,
                    collection_name=args.collection_name or config.collection_name,
                    index_type=args.index_type or config.index_type,
                    # 添加其他配置参数
                    output_dir=args.output or (config.output_dir if config.output_dir else "data/vectors"),
                    top_k=config.top_k,
                    similarity_threshold=config.similarity_threshold,
                    nlist=config.nlist,
                    m_factor=config.m_factor
                )
                
                build_index(args.categories, args.output, build_config)
            elif args.command == "search":
                search(args)
            elif args.command == "update":
                update_category(args)
            else:
                logger.error("未指定命令。使用 --help 获取帮助。")
                parser.print_help()
                sys.exit(1)
        except Exception as e:
            logger.exception(f"执行时发生错误: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
