#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
from typing import List, Dict, Any

def merge_category_files(input_dir: str = "data/single_file", output_file: str = "data/categories.json"):
    """
    将input_dir目录下的所有JSON文件合并到一个output_file文件中
    
    Args:
        input_dir: 包含单个分类JSON文件的目录路径
        output_file: 输出的合并JSON文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 获取所有JSON文件路径
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 读取所有JSON文件内容
    all_categories = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                category_data = json.load(f)
                all_categories.append(category_data)
                print(f"已读取: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
    
    # 按ID排序
    all_categories.sort(key=lambda x: int(x.get('id', 0)))
    
    # 写入合并文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_categories, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成! 共 {len(all_categories)} 个分类已写入 {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='合并单个分类JSON文件到一个数组文件')
    parser.add_argument('--input', default='data/single_file', help='包含单个分类JSON文件的目录路径')
    parser.add_argument('--output', default='data/categories.json', help='输出的合并JSON文件路径')
    
    args = parser.parse_args()
    
    merge_category_files(input_dir=args.input, output_file=args.output) 