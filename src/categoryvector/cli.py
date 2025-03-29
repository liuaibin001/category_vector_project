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
    
    # 设置日志
    logger = setup_logger("categoryvector", level=args.log_level)
    
    try:
        # 创建配置
        config = CategoryVectorConfig(
            model_name=args.model,
            data_dir=Path(args.categories).parent,
            log_level=args.log_level,
            vector_dim=args.vector_dim
        )
        
        # 加载类别数据
        categories_file = Path(args.categories)
        logger.info(f"加载类别数据: {categories_file}")
        processor = CategoryProcessor(config)
        processor.load_from_json(categories_file)
        logger.info(f"加载了 {len(processor.categories)} 个类别")
        
        # 生成向量
        logger.info(f"使用模型 {config.model_name} 生成向量")
        generator = VectorGenerator(model_name=config.model_name, config=config)
        
        # 为每个分类生成向量
        for category in processor.categories.values():
            category = generator.enrich_category_vectors(category)
        
        # 构建向量索引
        logger.info("构建向量索引")
        storage = VectorStorage(
            dimension=generator.model.get_sentence_embedding_dimension(),
            config=config
        )
        for category in processor.categories.values():
            storage.add_category(category)
        
        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引和分类数据
        logger.info(f"保存索引到 {output_dir}")
        storage.save(output_dir)
        
        logger.info("索引构建完成")
    
    except Exception as e:
        logger.exception(f"构建索引时发生错误: {e}")
        sys.exit(1)


def search(args=None):
    """搜索类别"""
    if args is None:
        args = parse_search_args()
    
    # 设置日志
    logger = setup_logger("categoryvector", level=args.log_level)
    
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
            log_level=args.log_level
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
