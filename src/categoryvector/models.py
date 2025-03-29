from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Category:
    """分类数据模型"""
    id: int
    path: str
    levels: List[str]
    level_depth: int
    description: Optional[str] = None
    keywords: List[str] = None
    examples: List[str] = None
    exclusions: List[str] = None
    vector: Optional[np.ndarray] = None
    level_vectors: Dict[str, np.ndarray] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.examples is None:
            self.examples = []
        if self.exclusions is None:
            self.exclusions = []
        if self.level_vectors is None:
            self.level_vectors = {}
            
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "path": self.path,
            "levels": self.levels,
            "level_depth": self.level_depth,
            "description": self.description,
            "keywords": self.keywords,
            "examples": self.examples,
            "exclusions": self.exclusions,
            "vector": self.vector.tolist() if self.vector is not None else None,
            "level_vectors": {
                k: v.tolist() for k, v in self.level_vectors.items()
            } if self.level_vectors else {}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Category':
        """从字典创建实例"""
        if "vector" in data and data["vector"] is not None:
            data["vector"] = np.array(data["vector"])
        if "level_vectors" in data and data["level_vectors"]:
            data["level_vectors"] = {
                k: np.array(v) for k, v in data["level_vectors"].items()
            }
        return cls(**data) 