#!/usr/bin/env python3
"""CategoryVector 主程序入口文件."""

import argparse
import json
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
from categoryvector.models import Category


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="CategoryVector - 类别向量搜索工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 构建索引子命令
    build_parser = subparsers.add_parser("build", help="构建类别向量索引")
    build_parser.add_argument("--categories", "-c", required=True, type=str,
                             help="类别数据JSON文件路径")
    build_parser.add_argument("--output", "-o", required=True, type=str,
                             help="输出索引目录")
    build_parser.add_argument("--model", "-m", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", type=str,
                             help="使用的向量模型名称")
    build_parser.add_argument("--enrich", "-e", action="store_true",
                             help="是否使用大模型丰富分类数据")
    
    # 搜索子命令
    search_parser = subparsers.add_parser("search", help="搜索类别")
    search_parser.add_argument("--index", "-i", required=True, type=str,
                              help="索引目录路径")
    search_parser.add_argument("--query", "-q", required=True, type=str,
                              help="搜索查询文本")
    search_parser.add_argument("--top-k", "-k", default=5, type=int,
                              help="返回结果数量")
    search_parser.add_argument("--threshold", "-t", default=0.6, type=float,
                              help="相似度阈值")
    search_parser.add_argument("--level", "-l", type=int,
                              help="指定搜索层级")
    
    # 通用参数
    parser.add_argument("--log-level", default="INFO", type=str,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="日志级别")
    
    return parser.parse_args()


def build_index(args):
    """构建索引."""
    logger = logging.getLogger("categoryvector")
    
    # 创建配置
    config = CategoryVectorConfig(
        model_name=args.model,
        data_dir=Path(args.categories).parent,
        log_level=args.log_level
    )
    
    # 加载类别数据
    logger.info(f"加载类别数据: {args.categories}")
    processor = CategoryProcessor(config)
    processor.load_from_json(args.categories)
    logger.info(f"加载了 {len(processor.categories)} 个类别")
    
    # 丰富分类数据
    if args.enrich:
        logger.info("使用大模型丰富分类数据")
        # TODO: 实现大模型丰富数据的逻辑
    
    # 生成向量
    logger.info(f"使用模型 {args.model} 生成向量")
    generator = VectorGenerator(model_name=args.model, config=config)
    
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


def search(args):
    """搜索类别."""
    logger = logging.getLogger("categoryvector")
    
    # 创建配置
    config = CategoryVectorConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",  # 默认模型
        data_dir=Path(args.index),
        log_level=args.log_level
    )
    
    # 加载索引和类别数据
    logger.info(f"加载索引: {args.index}")
    storage = VectorStorage(384)  # 默认维度
    storage.load(args.index)
    
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
    print("=" * 50)
    print(f"{'ID':<10} {'相似度':<10} {'路径':<30} {'描述'}")
    print("-" * 50)
    
    for category, score in results:
        description = category.description if category.description else ""
        if len(description) > 50:
            description = description[:47] + "..."
        print(f"{category.id:<10} {score:.4f}     {category.path:<30} {description}")


def main():
    """主程序入口."""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    setup_logger("categoryvector", level=args.log_level)
    logger = logging.getLogger("categoryvector")
    
    try:
        # 根据命令执行相应功能
        if args.command == "build":
            build_index(args)
        elif args.command == "search":
            search(args)
        else:
            logger.error("请指定子命令: build 或 search")
            sys.exit(1)
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()