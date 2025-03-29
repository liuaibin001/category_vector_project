"""数据处理模块测试."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryProcessor, CategoryNode


class TestCategoryProcessor(unittest.TestCase):
    """测试类别处理器."""
    
    def setUp(self):
        """测试前准备."""
        # 创建临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # 创建测试配置
        self.config = CategoryVectorConfig(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            data_dir=self.data_dir,
            log_level="ERROR"
        )
        
        # 创建测试数据
        self.categories = [
            {
                "id": "1",
                "name": "电子产品",
                "description": "包括各种电子设备和数码产品",
                "parent_id": None,
                "level": 1
            },
            {
                "id": "1.1",
                "name": "手机",
                "description": "移动通信设备，包括智能手机和功能手机",
                "parent_id": "1",
                "level": 2
            },
            {
                "id": "1.2",
                "name": "电脑",
                "description": "计算机设备，包括台式机、笔记本电脑",
                "parent_id": "1",
                "level": 2
            },
            {
                "id": "1.2.1",
                "name": "笔记本电脑",
                "description": "便携式计算机",
                "parent_id": "1.2",
                "level": 3
            }
        ]
        
        # 保存测试数据
        category_file = self.data_dir / "test_categories.json"
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, ensure_ascii=False)
        
        self.test_file = category_file
    
    def tearDown(self):
        """测试后清理."""
        self.temp_dir.cleanup()
    
    def test_load_from_json(self):
        """测试从JSON加载类别数据."""
        processor = CategoryProcessor(self.config)
        processor.load_from_json(self.test_file)
        
        # 验证加载结果
        self.assertEqual(len(processor.categories), 4)
        self.assertEqual(len(processor.id_to_category), 4)
        
        # 验证类别内容
        category = processor.get_category_by_id("1")
        self.assertIsNotNone(category)
        self.assertEqual(category.name, "电子产品")
        self.assertEqual(category.level, 1)
        
        # 验证父子关系
        children = processor.get_children("1")
        self.assertEqual(len(children), 2)
        self.assertIn("1.1", [c.id for c in children])
        self.assertIn("1.2", [c.id for c in children])
    
    def test_get_category_texts(self):
        """测试获取类别文本."""
        processor = CategoryProcessor(self.config)
        processor.load_from_json(self.test_file)
        
        texts = processor.get_category_texts_for_embedding()
        self.assertEqual(len(texts), 4)
        
        # 验证文本格式
        for category_id, text in texts.items():
            category = processor.get_category_by_id(category_id)
            self.assertIn(category.name, text)
            if category.description:
                self.assertIn(category.description, text)
    
    def test_category_tree(self):
        """测试类别树结构."""
        processor = CategoryProcessor(self.config)
        processor.load_from_json(self.test_file)
        
         # 测试树结构
        tree = processor.build_category_tree()
        
        # 验证根节点
        self.assertEqual(len(tree), 1)
        root = tree[0]
        self.assertEqual(root.id, "1")
        self.assertEqual(root.name, "电子产品")
        
        # 验证子节点
        self.assertEqual(len(root.children), 2)
        
        # 验证深层节点
        computer_node = None
        for child in root.children:
            if child.id == "1.2":
                computer_node = child
                break
        
        self.assertIsNotNone(computer_node)
        self.assertEqual(len(computer_node.children), 1)
        self.assertEqual(computer_node.children[0].id, "1.2.1")
        self.assertEqual(computer_node.children[0].name, "笔记本电脑")
    
    def test_serialization(self):
        """测试序列化和反序列化."""
        processor = CategoryProcessor(self.config)
        processor.load_from_json(self.test_file)
        
        # 序列化
        output_file = self.data_dir / "serialized_categories.json"
        processor.save_to_json(output_file)
        
        # 验证文件存在
        self.assertTrue(output_file.exists())
        
        # 重新加载并验证
        new_processor = CategoryProcessor(self.config)
        new_processor.load_from_json(output_file)
        
        # 验证加载结果
        self.assertEqual(len(new_processor.categories), len(processor.categories))
        for cat_id, category in processor.id_to_category.items():
            self.assertIn(cat_id, new_processor.id_to_category)
            new_cat = new_processor.get_category_by_id(cat_id)
            self.assertEqual(new_cat.name, category.name)
            self.assertEqual(new_cat.description, category.description)


if __name__ == "__main__":
    unittest.main()