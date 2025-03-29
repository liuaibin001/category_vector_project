"""配置管理模块."""

from pathlib import Path
from typing import Dict, List, Optional, Union

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
        description="FAISS索引类型: flat, ivf, hnsw等"
    )
    nlist: int = Field(
        default=100, 
        description="IVF索引的聚类数量"
    )
    m_factor: int = Field(
        default=16, 
        description="HNSW索引的连接数"
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
    
    # 日志配置
    log_level: str = Field(
        default="INFO", 
        description="日志级别"
    )
    log_file: Optional[Path] = Field(
        default=None, 
        description="日志文件路径，None表示仅控制台输出"
    )
    
    @field_validator('data_dir', 'log_file', mode='before')
    @classmethod
    def validate_path(cls, v):
        """验证并转换路径."""
        if v is None:
            return v
        return Path(v) if not isinstance(v, Path) else v
    
    class Config:
        """配置元数据."""
        
        validate_assignment = True
        arbitrary_types_allowed = True
