"""数据处理模块."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import os
import sys
from tqdm import tqdm

try:
    from src.categoryvector.config import CategoryVectorConfig
    from src.categoryvector.utils.logging_utils import default_logger as logger
    from src.categoryvector.models import Category
except ImportError:
    from .config import CategoryVectorConfig
    from .utils.logging_utils import default_logger as logger
    from .models import Category


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
    """分类数据处理器"""
    
    def __init__(self, config: Optional[CategoryVectorConfig] = None):
        """初始化数据处理器
        
        Args:
            config: 配置对象，可选
        """
        self.config = config
        self.categories: Dict[int, Category] = {}
        
    def load_from_json(self, file_path: Path) -> None:
        """从JSON文件加载分类数据
        
        Args:
            file_path: JSON文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 处理数据
            if isinstance(data, list):
                # 处理数组格式
                logger.info(f"从文件加载 {len(data)} 个分类...")
                total = len(data)
                with tqdm(total=total, desc="加载分类", unit="类别", colour="blue", dynamic_ncols=True) as pbar:
                    for i, category_data in enumerate(data):
                        try:
                            # 确保必要字段存在
                            if not all(k in category_data for k in ['id', 'path', 'levels', 'level_depth']):
                                logger.warning(f"分类 {category_data.get('id', '未知')} 缺少必要字段，已跳过")
                                continue
                                
                            # 创建分类对象
                            category = Category.from_dict(category_data)
                            self.categories[category.id] = category
                            
                        except Exception as e:
                            logger.error(f"处理分类 {category_data.get('id', '未知')} 时发生错误: {e}")
                            continue
                        finally:
                            # 更新进度条
                            pbar.update(1)
                            pbar.set_postfix({"完成": f"{(i+1)/total*100:.1f}%"})
            else:
                # 处理字典格式 {"1": {}, "2": {}, ...}
                logger.info(f"从文件加载 {len(data)} 个分类...")
                items = list(data.items())
                total = len(items)
                with tqdm(total=total, desc="加载分类", unit="类别", colour="blue", dynamic_ncols=True) as pbar:
                    for i, (cat_id, cat_data) in enumerate(items):
                        try:
                            # 确保必要字段存在
                            if not all(k in cat_data for k in ['id', 'path', 'levels', 'level_depth']):
                                logger.warning(f"分类 {cat_id} 缺少必要字段，已跳过")
                                continue
                                
                            # 创建分类对象
                            category = Category.from_dict(cat_data)
                            self.categories[category.id] = category
                            
                        except Exception as e:
                            logger.error(f"处理分类 {cat_id} 时发生错误: {e}")
                            continue
                        finally:
                            # 更新进度条
                            pbar.update(1)
                            pbar.set_postfix({"完成": f"{(i+1)/total*100:.1f}%"})
                    
            logger.info(f"成功加载 {len(self.categories)} 个分类")
            
        except Exception as e:
            logger.error(f"加载分类数据时发生错误: {e}")
            raise
            
    def save_to_json(self, file_path: Path) -> None:
        """保存分类数据到JSON文件
        
        Args:
            file_path: 保存路径
        """
        try:
            # 转换为字典格式
            data = {
                str(cat.id): cat.to_dict()
                for cat in self.categories.values()
            }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"成功保存 {len(self.categories)} 个分类到 {file_path}")
            
        except Exception as e:
            logger.error(f"保存分类数据时发生错误: {e}")
            raise
            
    def get_category_by_id(self, category_id: int) -> Optional[Category]:
        """根据ID获取分类
        
        Args:
            category_id: 分类ID
            
        Returns:
            分类对象，如果不存在则返回None
        """
        return self.categories.get(category_id)
        
    def get_categories_by_level(self, level: int) -> List[Category]:
        """获取指定层级的所有分类
        
        Args:
            level: 目标层级
            
        Returns:
            分类对象列表
        """
        return [
            cat for cat in self.categories.values()
            if cat.level_depth == level
        ]
        
    def get_child_categories(self, parent_id: int) -> List[Category]:
        """获取指定分类的所有子分类
        
        Args:
            parent_id: 父分类ID
            
        Returns:
            子分类对象列表
        """
        parent = self.get_category_by_id(parent_id)
        if not parent:
            return []
            
        return [
            cat for cat in self.categories.values()
            if cat.level_depth == parent.level_depth + 1
            and cat.path.startswith(parent.path)
        ]
        
    def enrich_category_data(
        self,
        category: Category,
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        exclusions: Optional[List[str]] = None
    ) -> Category:
        """丰富分类数据
        
        Args:
            category: 分类对象
            description: 分类描述
            keywords: 关键词列表
            examples: 样例列表
            exclusions: 排除词列表
            
        Returns:
            更新后的分类对象
        """
        if description:
            category.description = description
        if keywords:
            category.keywords = keywords
        if examples:
            category.examples = examples
        if exclusions:
            category.exclusions = exclusions
            
        return category
