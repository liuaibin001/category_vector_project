"""数据处理模块."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from categoryvector.config import CategoryVectorConfig
from categoryvector.utils.logging_utils import default_logger as logger


class CategoryNode:
    """类别节点类，表示分类层次结构中的一个节点."""
    
    def __init__(
        self, 
        id: str, 
        name: str, 
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        level: int = 0
    ):
        """初始化类别节点.
        
        Args:
            id: 节点唯一标识符
            name: 节点名称
            description: 节点描述
            parent_id: 父节点ID
            level: 节点层级
        """
        self.id = id
        self.name = name
        self.description = description or name
        self.parent_id = parent_id
        self.level = level
        self.children: List[CategoryNode] = []
        self.vector = None
    
    def add_child(self, child_node: 'CategoryNode') -> None:
        """添加子节点.
        
        Args:
            child_node: 子节点对象
        """
        self.children.append(child_node)
    
    def to_dict(self) -> Dict:
        """将节点转换为字典.
        
        Returns:
            包含节点信息的字典
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "level": self.level,
            "children": [child.to_dict() for child in self.children]
        }
    
    def get_full_path(self, category_map: Dict[str, 'CategoryNode']) -> str:
        """获取从根节点到当前节点的完整路径.
        
        Args:
            category_map: ID到节点的映射字典
            
        Returns:
            完整路径字符串，如"根类别/子类别/当前类别"
        """
        path_parts = [self.name]
        current_parent_id = self.parent_id
        
        while current_parent_id:
            parent = category_map.get(current_parent_id)
            if parent:
                path_parts.insert(0, parent.name)
                current_parent_id = parent.parent_id
            else:
                break
                
        return " / ".join(path_parts)
    
    def get_text_for_embedding(self, category_map: Dict[str, 'CategoryNode']) -> str:
        """获取用于生成嵌入向量的文本表示.
        
        Args:
            category_map: ID到节点的映射字典
            
        Returns:
            用于生成嵌入向量的文本
        """
        full_path = self.get_full_path(category_map)
        
        # 组合路径、名称和描述以获得更丰富的语义表示
        if self.description and self.description != self.name:
            return f"{full_path}. {self.description}"
        return full_path


class CategoryProcessor:
    """类别数据处理器，负责加载和处理分类数据."""
    
    def __init__(self, config: CategoryVectorConfig):
        """初始化处理器.
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.category_map: Dict[str, CategoryNode] = {}
        self.root_categories: List[CategoryNode] = []
    
    def load_from_json(self, file_path: Union[str, Path]) -> None:
        """从JSON文件加载类别数据.
        
        Args:
            file_path: JSON文件路径
        """
        file_path = Path(file_path)
        logger.info(f"从 {file_path} 加载类别数据")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                categories_data = json.load(f)
            
            # 第一遍：创建所有节点
            for cat_data in categories_data:
                node = CategoryNode(
                    id=cat_data['id'],
                    name=cat_data['name'],
                    description=cat_data.get('description'),
                    parent_id=cat_data.get('parent_id'),
                    level=cat_data.get('level', 0)
                )
                self.category_map[node.id] = node
            
            # 第二遍：构建层次结构
            for cat_id, node in self.category_map.items():
                if node.parent_id:
                    if node.parent_id in self.category_map:
                        self.category_map[node.parent_id].add_child(node)
                    else:
                        logger.warning(f"类别 {node.id} 的父类别 {node.parent_id} 不存在")
                else:
                    self.root_categories.append(node)
            
            logger.info(f"成功加载 {len(self.category_map)} 个类别，{len(self.root_categories)} 个根类别")
        
        except Exception as e:
            logger.error(f"加载类别数据失败: {str(e)}")
            raise
    
    def get_all_categories(self) -> List[CategoryNode]:
        """获取所有类别节点.
        
        Returns:
            所有类别节点列表
        """
        return list(self.category_map.values())
    
    def get_category_by_id(self, category_id: str) -> Optional[CategoryNode]:
        """通过ID获取类别节点.
        
        Args:
            category_id: 类别ID
            
        Returns:
            类别节点，如果不存在则返回None
        """
        return self.category_map.get(category_id)
    
    def get_category_texts_for_embedding(self) -> List[Tuple[str, str]]:
        """获取所有类别的文本表示，用于生成嵌入向量.
        
        Returns:
            (类别ID, 文本表示)元组的列表
        """
        texts = []
        for category_id, node in self.category_map.items():
            text = node.get_text_for_embedding(self.category_map)
            texts.append((category_id, text))
        return texts
