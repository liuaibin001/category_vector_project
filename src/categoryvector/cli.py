#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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
        required=True,
        help="输出索引目录"
    )
    parser.add_argument(
        "--vector-dim",
        "-d",
        type=int,
        default=384,
        help="向量维度"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="向量模型名称"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出（自动设置日志级别为DEBUG）"
    )
    # 添加Milvus相关参数
    parser.add_argument(
        "--milvus-host",
        default="localhost",
        help="Milvus服务器地址"
    )
    parser.add_argument(
        "--milvus-port",
        default="19530",
        help="Milvus服务器端口"
    )
    parser.add_argument(
        "--collection-name",
        default="category_vectors",
        help="Milvus集合名称"
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="索引类型: flat, ivf, hnsw"
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
        default=5,
        help="返回结果数量"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.6,
        help="相似度阈值"
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
        default="INFO",
        help="日志级别"
    )
    
    return parser.parse_args(args)


def build_index(args=None):
    """构建索引"""
    if args is None:
        args = parse_build_args()
    
    # 处理详细日志模式
    if args.verbose:
        args.log_level = "DEBUG"
        
    # 设置日志
    logger = setup_logger("categoryvector", level=args.log_level)
    
    logger.info(f"开始构建索引过程...")
    logger.info(f"参数信息: 类别数据={args.categories}, 输出目录={args.output}, 向量维度={args.vector_dim}, 模型={args.model}, 日志级别={args.log_level}")
    
    try:
        # 创建配置
        logger.info(f"正在创建配置...")
        config = CategoryVectorConfig(
            model_name=args.model,
            data_dir=Path(args.categories).parent,
            log_level=args.log_level,
            vector_dim=args.vector_dim,
            # 添加Milvus配置参数
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            collection_name=args.collection_name,
            index_type=args.index_type
        )
        logger.debug(f"配置已创建: 模型={config.model_name}, 向量维度={config.vector_dim}, Milvus主机={config.milvus_host}, Milvus端口={config.milvus_port}")
        
        # 加载类别数据
        categories_file = Path(args.categories)
        logger.info(f"正在加载类别数据: {categories_file}")
        processor = CategoryProcessor(config)
        processor.load_from_json(categories_file)
        
        categories_count = len(processor.categories)
        logger.info(f"成功加载 {categories_count} 个类别")
        if categories_count > 0 and args.log_level == "DEBUG":
            # 在DEBUG级别输出一些分类样例
            sample_size = min(3, categories_count)
            sample_categories = list(processor.categories.values())[:sample_size]
            for i, cat in enumerate(sample_categories):
                logger.debug(f"分类样例 {i+1}/{sample_size}: ID={cat.id}, 路径={cat.path}, 层级={cat.level_depth}")
        
        # 生成向量
        logger.info(f"开始加载模型: {config.model_name}")
        generator = VectorGenerator(model_name=config.model_name, config=config)
        logger.info(f"模型加载完成，向量维度: {generator.model.get_sentence_embedding_dimension()}")
        
        # 为每个分类生成向量
        logger.info(f"开始为 {categories_count} 个分类生成向量...")
        total_categories = categories_count
        processed_count = 0
        
        for cat_id, category in processor.categories.items():
            processed_count += 1
            if processed_count % 10 == 0 or processed_count == total_categories:
                logger.info(f"处理进度: {processed_count}/{total_categories} ({processed_count/total_categories*100:.1f}%)")
            elif processed_count % 5 == 0:
                logger.debug(f"处理进度: {processed_count}/{total_categories} ({processed_count/total_categories*100:.1f}%)")
                
            logger.debug(f"正在生成分类向量: ID={category.id}, 路径={category.path}")
            try:
                category = generator.enrich_category_vectors(category)
                logger.debug(f"分类 ID={category.id} 向量生成成功，向量形状={category.vector.shape}，层级向量数量={len(category.level_vectors)}")
            except Exception as e:
                logger.error(f"为分类 ID={category.id} 生成向量时出错: {e}")
                
        logger.info(f"所有分类向量生成完成")
        
        # 构建向量索引
        logger.info("开始构建Milvus向量索引...")
        storage = VectorStorage(
            dimension=generator.model.get_sentence_embedding_dimension(),
            config=config
        )
        
        # 添加分类到索引
        added_count = 0
        for cat_id, category in processor.categories.items():
            try:
                storage.add_category(category)
                added_count += 1
                if added_count % 50 == 0:
                    logger.debug(f"已添加 {added_count}/{total_categories} 个分类到索引")
            except Exception as e:
                logger.error(f"添加分类 ID={category.id} 到索引时出错: {e}")
                
        logger.info(f"索引构建完成，共添加了 {added_count} 个分类")
        
        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引和分类数据
        logger.info(f"开始保存索引到 {output_dir}")
        storage.save(output_dir)
        
        # 输出索引统计信息
        logger.info(f"索引保存完成:")
        logger.info(f"- Milvus集合名称: {config.collection_name}")
        logger.info(f"- Milvus主机地址: {config.milvus_host}:{config.milvus_port}")
        logger.info(f"- 索引类型: {config.index_type}")
        logger.info(f"- 总类别数量: {len(storage.categories)}")
        logger.info(f"- 向量维度: {config.vector_dim}")
        
        logger.info("索引构建过程成功完成")
    
    except Exception as e:
        logger.exception(f"构建索引时发生错误: {e}")
        sys.exit(1)


def search(args=None):
    """搜索类别"""
    if args is None:
        args = parse_search_args()
    
    # 处理详细日志模式
    if hasattr(args, 'verbose') and args.verbose:
        args.log_level = "DEBUG"
        
    # 设置日志
    logger = setup_logger("categoryvector", level=args.log_level)
    
    logger.info(f"开始搜索过程...")
    
    try:
        # 检查索引目录
        index_path = Path(args.index)
        if not index_path.exists():
            logger.error(f"索引目录不存在: {index_path}")
            sys.exit(1)
            
        # 创建配置
        config = CategoryVectorConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 默认模型
            data_dir=index_path,
            log_level=args.log_level,
            # 添加Milvus配置
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            collection_name=args.collection_name
        )
        
        # 加载索引和类别数据
        logger.info(f"加载索引: {args.index}")
        storage = VectorStorage(384, config)  # 默认维度
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
        generator = VectorGenerator(model_name=config.model_name, config=config)
        query_vector = generator.generate_query_vector(args.query)
        
        # 执行搜索
        if args.level:
            # 按层级搜索
            results = storage.search_by_level(
                query_vector,
                level=args.level,
                top_k=args.top_k,
                threshold=args.threshold
            )
        else:
            # 全局搜索
            results = storage.search(
                query_vector,
                top_k=args.top_k,
                threshold=args.threshold
            )
        
        # 打印结果
        print(f"\n查询: {args.query}")
        print("=" * 70)
        print(f"找到 {len(results)} 个结果，显示前 {min(len(results), args.top_k)} 个（相似度阈值 {args.threshold}）")
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
            required=True,
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
        
        # 添加Milvus相关参数到build子命令
        build_parser.add_argument(
            "--milvus-host",
            default="localhost",
            help="Milvus服务器地址"
        )
        build_parser.add_argument(
            "--milvus-port",
            default="19530",
            help="Milvus服务器端口"
        )
        build_parser.add_argument(
            "--collection-name",
            default="category_vectors",
            help="Milvus集合名称"
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
            default=0.6,
            help="相似度阈值"
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
            default="localhost",
            help="Milvus服务器地址"
        )
        search_parser.add_argument(
            "--milvus-port",
            default="19530",
            help="Milvus服务器端口"
        )
        search_parser.add_argument(
            "--collection-name",
            default="category_vectors",
            help="Milvus集合名称"
        )
        
        args = parser.parse_args()
        
        # 设置日志
        logger = setup_logger("categoryvector", level=args.log_level)
        
        try:
            if args.command == "build":
                build_index(args)
            elif args.command == "search":
                search(args)
            else:
                logger.error("未指定命令。使用 --help 获取帮助。")
                parser.print_help()
                sys.exit(1)
        except Exception as e:
            logger.exception(f"执行时发生错误: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
