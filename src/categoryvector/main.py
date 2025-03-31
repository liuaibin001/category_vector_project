"""API服务入口点"""

import os
import sys
import argparse
from pathlib import Path

try:
    import uvicorn
except ImportError:
    print("Uvicorn未安装，请使用poetry添加依赖: poetry add uvicorn")
    sys.exit(1)

try:
    from categoryvector import get_app
    from categoryvector.utils.logging_utils import setup_logger, default_logger as logger
except ImportError:
    from . import get_app
    from .utils.logging_utils import setup_logger, default_logger as logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动Category Vector API服务")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务监听的主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务监听的端口"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数量"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    return parser.parse_args()

def main():
    """启动API服务"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logger("categoryvector_api", level=args.log_level)
    logger.info(f"正在启动Category Vector API服务于 {args.host}:{args.port}...")
    
    try:
        # 获取FastAPI应用实例
        app = get_app()
        
        # 启动服务
        if args.reload:
            logger.info("已启用热重载，这是开发模式")
            
        # 使用uvicorn启动FastAPI应用
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level.lower()
        )
    except Exception as e:
        logger.exception(f"启动服务时发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 