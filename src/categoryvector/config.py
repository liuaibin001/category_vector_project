"""配置管理模块."""

import os
import toml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, field_validator
from categoryvector.utils.logging_utils import default_logger as logger


class CategoryVectorConfig(BaseModel):
    """分层类别向量配置类."""

    # 基本配置
    model_name: str = Field(
        default="all-MiniLM-L6-v2", 
        description="用于生成向量的模型名称"
    )
    vector_dim: int = Field(
        default=384, 
        description="向量维度"
    )
    data_dir: Path = Field(
        default=Path("./data"), 
        description="数据目录"
    )
    
    # 向量存储配置
    index_type: str = Field(
        default="flat", 
        description="索引类型: flat, ivf, hnsw等"
    )
    nlist: int = Field(
        default=100, 
        description="IVF索引的聚类数量"
    )
    m_factor: int = Field(
        default=16, 
        description="HNSW索引的连接数"
    )
    
    # Milvus配置
    milvus_host: str = Field(
        default="localhost", 
        description="Milvus服务器地址"
    )
    milvus_port: str = Field(
        default="19530", 
        description="Milvus服务器端口"
    )
    collection_name: str = Field(
        default="category_vectors", 
        description="Milvus集合名称"
    )
    
    # Redis配置
    redis_host: str = Field(
        default="localhost",
        description="Redis服务器地址"
    )
    redis_port: int = Field(
        default=6379,
        description="Redis服务器端口"
    )
    redis_db: int = Field(
        default=0,
        description="Redis数据库编号"
    )
    redis_password: str = Field(
        default="",
        description="Redis密码"
    )
    redis_prefix: str = Field(
        default="categoryvector:",
        description="Redis键前缀"
    )
    redis_ttl: int = Field(
        default=0,
        description="Redis键过期时间（秒），0表示永不过期"
    )
    redis_socket_timeout: int = Field(
        default=5,
        description="Redis socket超时时间（秒）"
    )
    redis_socket_connect_timeout: int = Field(
        default=5,
        description="Redis socket连接超时时间（秒）"
    )
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Redis超时时是否重试"
    )
    
    # 搜索配置
    top_k: int = Field(
        default=5, 
        description="搜索返回的结果数量"
    )
    similarity_threshold: float = Field(
        default=0.6, 
        description="相似度阈值，低于此值的结果将被过滤"
    )
    similarity_metric: str = Field(
        default="IP",
        description="相似度度量: IP (余弦相似度), L2 (欧氏距离)"
    )
    
    # 日志配置
    log_level: str = Field(
        default="INFO", 
        description="日志级别"
    )
    log_file: Optional[Path] = Field(
        default=None, 
        description="日志文件路径，None表示仅控制台输出"
    )
    
    # 数据目录配置
    output_dir: Optional[Path] = Field(
        default=None,
        description="输出目录路径，None表示使用data_dir"
    )
    
    @field_validator('data_dir', 'log_file', 'output_dir', mode='before')
    @classmethod
    def validate_path(cls, v):
        """验证路径字段."""
        if v is not None:
            return Path(v)
        return v
    
    @classmethod
    def from_toml(cls, file_path: Union[str, Path] = None) -> 'CategoryVectorConfig':
        """从TOML文件加载配置.
        
        Args:
            file_path: TOML文件路径，None表示使用默认路径
            
        Returns:
            CategoryVectorConfig实例
        """
        if file_path is None:
            file_path = Path("config.toml")
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"配置文件不存在: {file_path}，使用默认配置")
            return cls()
            
        try:
            config_data = toml.load(file_path)
            mapped_data = {}
            
            # 映射TOML节到配置字段
            if "milvus" in config_data:
                milvus_config = config_data["milvus"]
                mapped_data["milvus_host"] = milvus_config.get("host", "localhost")
                mapped_data["milvus_port"] = str(milvus_config.get("port", "19530"))
                mapped_data["collection_name"] = milvus_config.get("collection_name", "category_vectors")
                logger.info(f"从配置文件读取的 Milvus 配置: host={mapped_data['milvus_host']}, port={mapped_data['milvus_port']}")
                
            if "search" in config_data:
                search_config = config_data["search"]
                mapped_data["similarity_threshold"] = search_config.get("threshold", 0.6)
                mapped_data["top_k"] = search_config.get("top_k", 5)
                mapped_data["similarity_metric"] = search_config.get("similarity_metric", "IP")
                
            if "index" in config_data:
                index_config = config_data["index"]
                mapped_data["index_type"] = index_config.get("type", "flat")
                mapped_data["nlist"] = index_config.get("nlist", 100)
                mapped_data["m_factor"] = index_config.get("m_factor", 16)
                
            if "model" in config_data:
                model_config = config_data["model"]
                mapped_data["model_name"] = model_config.get("name", "all-MiniLM-L6-v2")
                mapped_data["vector_dim"] = model_config.get("vector_dim", 384)
                
            if "logging" in config_data:
                logging_config = config_data["logging"]
                mapped_data["log_level"] = logging_config.get("level", "INFO")
                log_file = logging_config.get("file")
                if log_file:
                    mapped_data["log_file"] = Path(log_file)
                    
            if "data" in config_data:
                data_config = config_data["data"]
                output_dir = data_config.get("output_dir")
                if output_dir:
                    mapped_data["output_dir"] = Path(output_dir)
                    
            if "redis" in config_data:
                redis_config = config_data["redis"]
                mapped_data["redis_host"] = redis_config.get("host", "localhost")
                mapped_data["redis_port"] = redis_config.get("port", 6379)
                mapped_data["redis_db"] = redis_config.get("db", 0)
                mapped_data["redis_password"] = redis_config.get("password", "")
                mapped_data["redis_prefix"] = redis_config.get("prefix", "categoryvector:")
                mapped_data["redis_ttl"] = redis_config.get("ttl", 0)
                mapped_data["redis_socket_timeout"] = redis_config.get("socket_timeout", 5)
                mapped_data["redis_socket_connect_timeout"] = redis_config.get("socket_connect_timeout", 5)
                mapped_data["redis_retry_on_timeout"] = redis_config.get("retry_on_timeout", True)
            
            # 创建配置实例
            config_instance = cls(**mapped_data)
            logger.info(f"最终配置: milvus_host={config_instance.milvus_host}, milvus_port={config_instance.milvus_port}")
            return config_instance
            
        except Exception as e:
            logger.error(f"读取配置文件失败: {e}")
            return cls()
            
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """获取Redis配置
        
        Returns:
            Dict[str, Any]: Redis配置字典
        """
        # 从配置文件加载配置
        config = cls.from_toml()
        
        return {
            'host': config.redis_host,
            'port': config.redis_port,
            'db': config.redis_db,
            'password': config.redis_password,
            'prefix': config.redis_prefix,
            'ttl': config.redis_ttl,
            'socket_timeout': config.redis_socket_timeout,
            'socket_connect_timeout': config.redis_socket_connect_timeout,
            'retry_on_timeout': config.redis_retry_on_timeout
        }

    class Config:
        """配置元数据."""
        
        validate_assignment = True
        arbitrary_types_allowed = True
