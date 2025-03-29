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
    build_parser.add_argument("--model", "-m", default="paraphrase-multilingual-MiniLM-L12-v2", type=str,
                             help="使用的向量模型名称")
    
    # 搜索子命令
    search_parser = subparsers.add_parser("search", help="搜索类别")
    search_parser.add_argument("--index", "-i", required=True, type=str,
                              help="索引目录路径")
    search_parser.add_argument("--query", "-q", required=True, type=str,
                              help="搜索查询文本")
    search_parser.add_argument("--top-k", "-k", default=5, type=int,
                              help="返回结果数量")
    search_parser.add_argument("--model", "-m", default="paraphrase-multilingual-MiniLM-L12-v2", type=str,
                              help="使用的向量模型名称")
    
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
    
    logger.info(f"加载类别数据: {args.categories}")
    processor = CategoryProcessor(config)
    processor.load_from_json(args.categories)
    logger.info(f"加载了 {len(processor.categories)} 个类别")
    
    logger.info(f"使用模型 {args.model} 生成向量")
    generator = VectorGenerator(config)
    id_to_vector = generator.generate_vectors(processor)
    logger.info(f"生成了 {len(id_to_vector)} 个向量")
    
    logger.info("构建向量索引")
    storage = VectorStorage(config)
    storage.build_index(id_to_vector, processor)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存索引到 {output_dir}")
    storage.save(output_dir)
    
    # 保存类别数据
    categories_file = output_dir / "categories.json"
    processor.save_to_json(categories_file)
    
    logger.info("索引构建完成")


def search(args):
    """搜索类别."""
    logger = logging.getLogger("categoryvector")
    
    # 创建配置
    config = CategoryVectorConfig(
        model_name=args.model,
        data_dir=Path(args.index),
        log_level=args.log_level
    )
    
    # 加载索引和类别数据
    logger.info(f"加载索引: {args.index}")
    storage = VectorStorage(config)
    storage.load(args.index)
    
    logger.info("加载类别数据")
    processor = CategoryProcessor(config)
    processor.load_from_json(Path(args.index) / "categories.json")
    
    # 生成查询向量
    logger.info(f"查询: {args.query}")
    generator = VectorGenerator(config)
    query_vector = generator.generate_query_vector(args.query)
    
    # 搜索
    results = storage.search(query_vector, top_k=args.top_k)
    
    # 打印结果
    print(f"\n查询: {args.query}")
    print("=" * 50)
    print(f"{'ID':<10} {'相似度':<10} {'名称':<20} {'描述'}")
    print("-" * 50)
    
    for result in results:
        category = processor.get_category_by_id(result["id"])
        description = category.description if category.description else ""
        if len(description) > 50:
            description = description[:47] + "..."
        print(f"{category.id:<10} {result['score']:.4f}     {category.name:<20} {description}")


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