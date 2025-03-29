#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from pathlib import Path

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryProcessor
from categoryvector.vector_generation import VectorGenerator
from categoryvector.vector_storage import VectorStorage
from categoryvector.utils.logging_utils import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Category Vector Management Tool")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    
    # Build index subcommand
    build_parser = subparsers.add_parser("build", help="Build category index")
    build_parser.add_argument(
        "--categories",
        required=True,
        help="Path to categories JSON file"
    )
    build_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for index"
    )
    build_parser.add_argument(
        "--vector-dim",
        type=int,
        default=384,
        help="Vector dimension"
    )
    build_parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Model name for vector generation"
    )
    
    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search categories")
    search_parser.add_argument(
        "--index",
        required=True,
        help="Index directory path"
    )
    search_parser.add_argument(
        "--query",
        required=True,
        help="Search query text"
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    search_parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold"
    )
    
    return parser.parse_args()


def build_index(args=None):
    """Build category index."""
    if args is None:
        parser = argparse.ArgumentParser(description="Build category index")
        parser.add_argument(
            "--categories",
            required=True,
            help="Path to categories JSON file"
        )
        parser.add_argument(
            "--output",
            required=True,
            help="Output directory for index"
        )
        parser.add_argument(
            "--vector-dim",
            type=int,
            default=384,
            help="Vector dimension"
        )
        parser.add_argument(
            "--model",
            default="all-MiniLM-L6-v2",
            help="Model name for vector generation"
        )
        args = parser.parse_args()
    
    logger = setup_logger("categoryvector")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = CategoryVectorConfig(
        vector_dim=args.vector_dim,
        model_name=args.model,
        data_dir=output_dir
    )
    
    # Initialize components
    data_processor = CategoryProcessor(config)
    vector_generator = VectorGenerator(config)
    vector_storage = VectorStorage(config)
    
    # Load and process categories
    logger.info(f"Loading categories from: {args.categories}")
    data_processor.load_from_json(args.categories)
    categories = data_processor.get_all_categories()
    category_texts = data_processor.get_category_texts_for_embedding()
    
    # Generate vectors
    logger.info("Generating category vectors")
    vectors, names = vector_generator.generate_vectors(category_texts)
    
    # Store vectors
    logger.info(f"Saving vectors to: {args.output}")
    vector_storage.store_vectors(vectors, names)
    
    logger.info("Index building completed")


def search(args=None):
    """Search categories."""
    if args is None:
        parser = argparse.ArgumentParser(description="Search categories")
        parser.add_argument(
            "--index",
            required=True,
            help="Index directory path"
        )
        parser.add_argument(
            "--query",
            required=True,
            help="Search query text"
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of results to return"
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.6,
            help="Similarity threshold"
        )
        args = parser.parse_args()
    
    logger = setup_logger("categoryvector")
    
    # Initialize configuration
    config = CategoryVectorConfig(
        data_dir=args.index,
        top_k=args.top_k,
        similarity_threshold=args.threshold
    )
    logger.info(f"config:{config}")
    
    # Initialize components
    vector_storage = VectorStorage(config)
    vector_generator = VectorGenerator(config)
    
    # Load vectors
    logger.info(f"Loading vectors from: {args.index}")
    vectors, names, _ = vector_storage.load_vectors()
    
    # Find similar categories
    logger.info(f"Searching for similar categories to: {args.query}")
    results = vector_generator.find_similar_categories(
        vectors, names, args.query, top_k=args.top_k
    )
    
    # Print results
    print(f"\nQuery: {args.query}")
    print("=" * 50)
    print(f"{'Category':<20} {'Similarity'}")
    print("-" * 50)
    
    for category, similarity in results:
        if similarity >= args.threshold:
            print(f"{category:<20} {similarity:.4f}")


def main():
    """Main function."""
    try:
        args = parse_args()
        logger = setup_logger("categoryvector", level=args.log_level)
        
        if args.command == "build":
            build_index(args)
        elif args.command == "search":
            search(args)
        else:
            logger.error("No command specified. Use --help for help.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
