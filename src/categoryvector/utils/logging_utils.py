"""日志工具模块."""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
):
    """配置日志记录器.

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径，如果为None则仅输出到控制台
        rotation: 日志文件轮转策略
        retention: 日志保留策略
    """
    # 清除默认配置
    logger.remove()

    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format=DEFAULT_LOG_FORMAT,
        level=level,
        colorize=True,
    )

    # 添加文件处理器(如果指定)
    if log_file:
        logger.add(
            log_file,
            format=DEFAULT_LOG_FORMAT,
            level=level,
            rotation=rotation,
            retention=retention,
        )
    
    return logger.bind(name=name)


# 默认全局日志记录器
default_logger = setup_logger("default")
