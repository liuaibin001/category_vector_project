"""配置管理模块."""

import os
import toml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, field_validator


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
        description="输出目录路径"
    )
    
    @field_validator('data_dir', 'log_file', 'output_dir', mode='before')
    @classmethod
    def validate_path(cls, v):
        """验证并转换路径."""
        if v is None:
            return v
        return Path(v) if not isinstance(v, Path) else v
    
    @classmethod
    def from_toml(cls, file_path: Union[str, Path] = None) -> 'CategoryVectorConfig':
        """从TOML文件加载配置
        
        Args:
            file_path: TOML配置文件路径，如果为None，则尝试从环境变量或默认位置读取
            
        Returns:
            CategoryVectorConfig实例
        """
        # 尝试从环境变量获取配置文件路径
        if file_path is None:
            file_path = os.environ.get("CATEGORYVECTOR_CONFIG")
            
        # 如果环境变量未设置，尝试从默认位置读取
        if file_path is None:
            # 检查项目根目录中的config.toml
            default_paths = [
                Path("./config.toml"),
                Path("./configs/config.toml"),
                Path.home() / ".config" / "categoryvector" / "config.toml"
            ]
            
            for path in default_paths:
                if path.exists():
                    file_path = path
                    break
        
        print(f"配置文件路径: {file_path}")
                    
        # 如果还是没找到配置文件，使用默认配置
        if file_path is None or not Path(file_path).exists():
            print("未找到配置文件，使用默认配置")
            return cls()
            
        # 读取TOML文件
        try:
            config_data = toml.load(file_path)
            mapped_data = {}
            
            # 映射TOML节到配置字段
            if "milvus" in config_data:
                # 正确地获取 host 和 port
                mapped_data["milvus_host"] = config_data["milvus"].get("host")
                mapped_data["milvus_port"] = config_data["milvus"].get("port")
                mapped_data["collection_name"] = config_data["milvus"].get("collection_name")
                print(f"从配置文件读取的 Milvus 配置: host={mapped_data['milvus_host']}, port={mapped_data['milvus_port']}")
                
            if "search" in config_data:
                mapped_data["similarity_threshold"] = config_data["search"].get("threshold")
                mapped_data["top_k"] = config_data["search"].get("top_k")
                mapped_data["similarity_metric"] = config_data["search"].get("similarity_metric")
                
            if "index" in config_data:
                mapped_data["index_type"] = config_data["index"].get("type")
                mapped_data["nlist"] = config_data["index"].get("nlist")
                mapped_data["m_factor"] = config_data["index"].get("m_factor")
                
            if "model" in config_data:
                mapped_data["model_name"] = config_data["model"].get("name")
                mapped_data["vector_dim"] = config_data["model"].get("vector_dim")
                
            if "logging" in config_data:
                mapped_data["log_level"] = config_data["logging"].get("level")
                log_file = config_data["logging"].get("file")
                if log_file:
                    mapped_data["log_file"] = Path(log_file)
                    
            if "data" in config_data:
                output_dir = config_data["data"].get("output_dir")
                if output_dir:
                    mapped_data["output_dir"] = Path(output_dir)
            
            # 创建配置实例，过滤掉None值
            config_instance = cls(**{k: v for k, v in mapped_data.items() if v is not None})
            print(f"最终配置: milvus_host={config_instance.milvus_host}, milvus_port={config_instance.milvus_port}")
            return config_instance
            
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return cls()
    
    class Config:
        """配置元数据."""
        
        validate_assignment = True
        arbitrary_types_allowed = True
