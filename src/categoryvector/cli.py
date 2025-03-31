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

import numpy as np
from pymilvus import utility

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryProcessor
from categoryvector.vector_generation import VectorGenerator
from categoryvector.vector_storage import VectorStorage
from categoryvector.utils.logging_utils import setup_logger


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


def build_index(args=None):
    """构建索引"""
    if args is None:
        args = parse_build_args()
    
    # 加载配置文件
    config = CategoryVectorConfig.from_toml(args.config if hasattr(args, 'config') else None)
    print(f"通过 from_toml 加载的配置: milvus_host={config.milvus_host}, milvus_port={config.milvus_port}")
    
    # 处理详细日志模式
    if args.verbose:
        args.log_level = "DEBUG"
        
    # 命令行参数覆盖配置文件
    log_level = args.log_level or config.log_level
    
    # 设置日志
    logger = setup_logger("categoryvector", level=log_level)
    
    logger.info(f"开始构建索引过程...")
    
    try:
        # 参数覆盖配置
        print(f"命令行参数: milvus_host={args.milvus_host}, milvus_port={args.milvus_port}")
        
        model_name = args.model or config.model_name
        vector_dim = args.vector_dim or config.vector_dim
        milvus_host = args.milvus_host or config.milvus_host
        milvus_port = args.milvus_port or config.milvus_port
        
        print(f"合并后的 Milvus 参数: host={milvus_host}, port={milvus_port}")
        logger.debug(f"合并后的 Milvus 参数: host={milvus_host}, port={milvus_port}")
        
        collection_name = args.collection_name or config.collection_name
        index_type = args.index_type or config.index_type
        output_dir = args.output or (config.output_dir if config.output_dir else "data/vectors")
        
        logger.info(f"参数信息: 类别数据={args.categories}, 输出目录={output_dir}, 向量维度={vector_dim}, 模型={model_name}, 日志级别={log_level}")
        logger.info(f"Milvus配置: 主机={milvus_host}, 端口={milvus_port}, 集合={collection_name}")
        
        # 创建配置
        logger.info(f"正在创建配置...")
        build_config = CategoryVectorConfig(
            model_name=model_name,
            data_dir=Path(args.categories).parent,
            log_level=log_level,
            vector_dim=vector_dim,
            # 添加Milvus配置参数
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            index_type=index_type,
            # 添加其他配置参数
            output_dir=output_dir,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            nlist=config.nlist,
            m_factor=config.m_factor
        )
        
        # 创建存储实例并检查 Milvus 连通性
        logger.info("检查 Milvus 服务器连通性...")
        logger.debug(f"传递给 VectorStorage 的配置: milvus_host={build_config.milvus_host}, milvus_port={build_config.milvus_port}")
        storage = VectorStorage(dimension=vector_dim, config=build_config)
        try:
            storage.connect_to_milvus()
        except ConnectionError as e:
            logger.error(f"Milvus 服务器连接失败: {e}")
            logger.error("请检查 Milvus 服务器是否正在运行，以及网络连接是否正常")
            sys.exit(1)
            
        # Milvus 连接成功后，删除输出目录和 Milvus 集合
        output_path = Path(output_dir)
        if output_path.exists():
            try:
                logger.info(f"删除现有输出目录: {output_path}")
                shutil.rmtree(output_path)
                logger.info(f"成功删除输出目录: {output_path}")
            except Exception as e:
                logger.warning(f"删除输出目录时出错: {e}")
        
        # 删除现有 Milvus 集合
        try:
            if utility.has_collection(collection_name):
                logger.info(f"删除现有 Milvus 集合: {collection_name}")
                utility.drop_collection(collection_name)
                logger.info(f"成功删除 Milvus 集合: {collection_name}")
        except Exception as e:
            logger.warning(f"删除 Milvus 集合时出错: {e}")
            
        logger.debug(f"配置已创建: 模型={build_config.model_name}, 向量维度={build_config.vector_dim}, Milvus主机={build_config.milvus_host}, Milvus端口={build_config.milvus_port}")
        
        # 加载类别数据
        categories_file = Path(args.categories)
        logger.info(f"正在加载类别数据: {categories_file}")
        processor = CategoryProcessor(build_config)
        processor.load_from_json(categories_file)
        
        categories_count = len(processor.categories)
        logger.info(f"成功加载 {categories_count} 个类别")
        if categories_count > 0 and log_level == "DEBUG":
            # 在DEBUG级别输出一些分类样例
            sample_size = min(3, categories_count)
            sample_categories = list(processor.categories.values())[:sample_size]
            for i, cat in enumerate(sample_categories):
                logger.debug(f"分类样例 {i+1}/{sample_size}: ID={cat.id}, 路径={cat.path}, 层级={cat.level_depth}")
        
        # 生成向量
        logger.info(f"开始加载模型: {build_config.model_name}")
        generator = VectorGenerator(model_name=build_config.model_name, config=build_config)
        logger.info(f"模型加载完成，向量维度: {generator.model.get_sentence_embedding_dimension()}")
        
        # 为每个分类生成向量 - 使用并行处理加速
        logger.info(f"开始为 {categories_count} 个分类生成向量...")
        total_categories = categories_count
        
        # 获取所有分类
        all_categories = list(processor.categories.values())
        
        # 并行处理 - 使用batch_enrich_category_vectors中的进度条，这里不需要额外的进度条
        try:
            # 调用并行处理方法
            enriched_categories = generator.batch_enrich_category_vectors(
                all_categories, 
                max_workers=args.workers
            )
            
            # 更新分类词典
            for category in enriched_categories:
                processor.categories[category.id] = category
                
        except Exception as e:
            logger.error(f"并行生成向量时出错: {e}")
            sys.exit(1)
                
        logger.info(f"所有分类向量生成完成")
        
        # 构建向量索引
        logger.info("开始构建Milvus向量索引...")
        
        # 使用批量添加来提高性能
        logger.info(f"开始批量添加 {total_categories} 个分类到索引...")
        
        # 准备批量处理
        batch_size = args.batch_size or 100  # 每批次处理100个分类
        
        # 使用tqdm创建进度条显示整体进度
        with tqdm(total=total_categories, desc="构建索引", unit="类别") as pbar:
            # 收集所有分类为列表，以便批量处理
            categories_list = list(processor.categories.values())
            
            # 批量添加分类
            try:
                storage.batch_add_categories(categories_list, batch_size=batch_size)
                # 更新进度条到完成
                pbar.update(total_categories)
                pbar.set_postfix({"完成": "100.0%"})
            except Exception as e:
                logger.error(f"批量添加分类到索引时出错: {e}")
                
            # 显示添加结果
            added_count = len(storage.categories)
                
        logger.info(f"索引构建完成，共添加了 {added_count} 个分类")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存索引和分类数据
        logger.info(f"开始保存索引到 {output_path}")
        storage.save(output_path)
        
        # 输出索引统计信息
        logger.info(f"索引保存完成:")
        logger.info(f"- Milvus集合名称: {build_config.collection_name}")
        logger.info(f"- Milvus主机地址: {build_config.milvus_host}:{build_config.milvus_port}")
        logger.info(f"- 索引类型: {build_config.index_type}")
        logger.info(f"- 总类别数量: {len(storage.categories)}")
        logger.info(f"- 向量维度: {build_config.vector_dim}")
        
        logger.info("索引构建过程成功完成")
    
    except Exception as e:
        logger.exception(f"构建索引时发生错误: {e}")
        sys.exit(1)


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
        
        # 加载索引和类别数据
        logger.info(f"加载索引: {args.index}")
        try:
            storage.load(index_path)
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            sys.exit(1)
            
        if not storage.categories:
            logger.error(f"索引中没有分类数据")
            sys.exit(1)
            
        # 生成查询向量
        logger.info(f"查询: {args.query}")
        generator = VectorGenerator(model_name=search_config.model_name, config=search_config)
        query_vector = generator.generate_query_vector(args.query)
        
        # 检查集合中是否有数据
        entity_count = storage.collection.num_entities if storage.collection else 0
        if entity_count == 0:
            logger.error("Milvus集合中没有向量数据。请先运行build命令构建索引并确保数据成功保存到Milvus。")
            print("\n错误: Milvus集合为空，无法搜索。请先构建索引。")
            print("请运行: python -m src.categoryvector.cli build --categories your_categories.json")
            sys.exit(1)
            
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
    if args is None:
        args = parse_update_args()
    
    # 加载配置文件
    config = CategoryVectorConfig.from_toml(args.config if hasattr(args, 'config') else None)
    
    # 处理详细日志模式
    if args.verbose:
        args.log_level = "DEBUG"
        
    # 命令行参数覆盖配置文件
    log_level = args.log_level or config.log_level
    
    # 设置日志
    logger = setup_logger("categoryvector", level=log_level)
    
    logger.info(f"开始更新分类ID={args.category_id}的向量...")
    
    try:
        # 参数覆盖配置
        model_name = args.model or config.model_name
        vector_dim = args.vector_dim or config.vector_dim
        milvus_host = args.milvus_host or config.milvus_host
        milvus_port = args.milvus_port or config.milvus_port
        collection_name = args.collection_name or config.collection_name
        
        logger.info(f"参数信息: 索引目录={args.index}, 分类ID={args.category_id}, 向量维度={vector_dim}, 模型={model_name}")
        logger.info(f"Milvus配置: 主机={milvus_host}, 端口={milvus_port}, 集合={collection_name}")
        
        # 创建配置
        logger.info(f"正在创建配置...")
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
            m_factor=config.m_factor
        )
        
        # 创建存储实例并检查 Milvus 连通性
        logger.info("检查 Milvus 服务器连通性...")
        storage = VectorStorage(dimension=vector_dim, config=update_config)
        try:
            storage.connect_to_milvus()
        except ConnectionError as e:
            logger.error(f"Milvus 服务器连接失败: {e}")
            logger.error("请检查 Milvus 服务器是否正在运行，以及网络连接是否正常")
            sys.exit(1)
            
        # 加载索引目录
        index_path = Path(args.index)
        if not index_path.exists():
            logger.error(f"索引目录不存在: {index_path}")
            sys.exit(1)
            
        # 加载现有类别数据
        try:
            storage.load(index_path)
        except Exception as e:
            logger.error(f"加载索引数据失败: {e}")
            sys.exit(1)
            
        if not storage.categories:
            logger.error("索引中没有分类数据")
            sys.exit(1)
            
        # 查找要更新的分类
        category_id = args.category_id
        if category_id not in storage.categories:
            logger.error(f"未找到分类ID={category_id}，无法更新")
            sys.exit(1)
            
        category = storage.categories[category_id]
        logger.info(f"找到分类: ID={category.id}, 路径={category.path}")
        
        # 重新生成向量
        logger.info(f"加载模型: {update_config.model_name}")
        generator = VectorGenerator(model_name=update_config.model_name, config=update_config)
        
        logger.info(f"为分类ID={category.id}重新生成向量...")
        try:
            # 先清除旧向量
            category.vector = None
            category.level_vectors = {}
            
            # 生成新向量
            updated_category = generator.enrich_category_vectors(category)
            logger.info(f"分类ID={category.id}的向量生成成功")
            
            # 更新到Milvus
            logger.info(f"更新Milvus中的分类向量...")
            storage.add_category(updated_category)
            logger.info(f"Milvus向量更新成功")
            
            # 保存更新后的数据
            logger.info(f"保存更新后的索引数据...")
            storage.save(index_path)
            logger.info(f"索引数据保存完成")
            
            logger.info(f"分类ID={category.id}的向量已成功更新")
            
        except Exception as e:
            logger.error(f"更新分类向量时出错: {e}")
            sys.exit(1)
    
    except Exception as e:
        logger.exception(f"更新分类时发生错误: {e}")
        sys.exit(1)


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
            default=5,
            help="返回结果数量"
        )
        search_parser.add_argument(
            "--threshold",
            "-t",
            type=float,
            default=0.3,
            help="相似度阈值 (使用余弦相似度，范围0-1)"
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
                build_index(args)
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
