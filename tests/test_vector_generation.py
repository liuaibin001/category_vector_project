"""向量生成模块测试."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from categoryvector.config import CategoryVectorConfig
from categoryvector.data_processing import CategoryProcessor
from categoryvector.vector_generation import VectorGenerator
from categoryvector.vector_storage import VectorStorage


class TestVectorGeneration(unittest.TestCase):
    """测试向量生成."""
    
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
            }
        ]
        
        # 保存测试数据
        category_file = self.data_dir / "categories.json"
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, ensure_ascii=False)
    
    def tearDown(self):
        """测试后清理."""
        self.temp_dir.cleanup()
    
    def test_vector_generation(self):
        """测试向量生成."""
        # 加载类别数据
        processor = CategoryProcessor(self.config)
        processor.load_from_json(self.data_dir / "categories.json")
        
        # 生成向量
        generator = VectorGenerator(self.config)
        id_to_vector = generator.generate_vectors(processor)
        
        # 验证向量
        self.assertEqual(len(id_to_vector), 3)
        for cat_id, vector in id_to_vector.items():
            self.assertIn(cat_id, ["1", "1.1", "1.2"])
            self.assertIsInstance(vector, np.ndarray)
            self.assertEqual(vector.shape, (384,))  # 默认模型维度
    
    def test_query_vector_generation(self):
        """测试查询向量生成."""
        generator = VectorGenerator(self.config)
        
        # 生成查询向量
        query_vector = generator.generate_query_vector("智能手机")
        
        # 验证向量
        self.assertIsInstance(query_vector, np.ndarray)
        self.assertEqual(query_vector.shape, (384,))  # 默认模型维度
    
    def test_end_to_end(self):
        """测试端到端流程."""
        # 加载类别数据
        processor = CategoryProcessor(self.config)
        processor.load_from_json(self.data_dir / "categories.json")
        
        # 生成向量
        generator = VectorGenerator(self.config)
        id_to_vector = generator.generate_vectors(processor)
        
        # 构建索引
        storage = VectorStorage(self.config)
        storage.build_index(id_to_vector, processor)
        
        # 生成查询向量并搜索
        query_vector = generator.generate_query_vector("智能手机")
        results = storage.search(query_vector, top_k=3)
        
        # 验证结果
        self.assertEqual(len(results), 3)  # 应该返回所有3个类别
        self.assertEqual(results[0]["id"], "1.1")  # 手机应该排第一
        
        # 保存和加载索引
        index_path = self.data_dir / "index"
        storage.save(index_path)
        
        # 重新加载索引
        new_storage = VectorStorage(self.config)
        new_storage.load(index_path)
        
        # 再次搜索并验证结果
        new_results = new_storage.search(query_vector, top_k=3)
        self.assertEqual(len(new_results), 3)
        self.assertEqual(new_results[0]["id"], results[0]["id"])


if __name__ == "__main__":
    unittest.main()
