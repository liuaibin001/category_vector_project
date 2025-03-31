"""从CSV生成分类数据的工具"""

import csv
import json
import os
import requests
import time
from typing import List, Dict, Any
from pathlib import Path

class CategoryGenerator:
    def __init__(self, input_csv_path: str, output_json_path: str, api_key: str = None, 
                 begin_line: int = 0, end_line: int = None):
        """
        Initialize the CategoryGenerator with file paths and API key.
        
        Args:
            input_csv_path: Path to the input CSV file
            output_json_path: Path where the output JSON will be saved
            api_key: Deepseek API key (can be set via DEEPSEEK_API_KEY env var)
            begin_line: 开始处理的CSV行号(0-indexed)
            end_line: 结束处理的CSV行号(0-indexed)，如果为None则处理到文件末尾
        """
        self.input_csv_path = input_csv_path
        self.output_json_path = output_json_path
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Deepseek API key must be provided or set as DEEPSEEK_API_KEY environment variable")
        
        self.begin_line = begin_line
        self.end_line = end_line
        
        # 添加单文件存储目录
        self.single_files_dir = os.path.join(os.path.dirname(output_json_path), "single_file")
        # 确保单文件存储目录存在
        os.makedirs(self.single_files_dir, exist_ok=True)
        
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.categories = []
    
    def get_single_file_path(self, category_id: int) -> str:
        """获取单个分类的JSON文件路径"""
        return os.path.join(self.single_files_dir, f"{category_id}.json")
    
    def check_single_file_exists(self, category_id: int) -> bool:
        """检查单个分类的JSON文件是否已存在"""
        file_path = self.get_single_file_path(category_id)
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    def save_single_category_file(self, category_data: Dict[str, Any]) -> bool:
        """保存单个分类数据到独立的JSON文件"""
        category_id = category_data["id"]
        file_path = self.get_single_file_path(category_id)
        
        # 如果文件已存在，直接返回True
        if self.check_single_file_exists(category_id):
            return True
            
        try:
            # 使用临时文件写入，成功后再重命名，提高写入安全性
            temp_path = f"{file_path}.temp"
            with open(temp_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(category_data, jsonfile, ensure_ascii=False, indent=2)
                
            # 确保写入完整后再替换原文件
            os.replace(temp_path, file_path)
            print(f"已将ID={category_id}成功写入单独文件: {file_path}")
            return True
            
        except Exception as e:
            print(f"保存单独文件失败 ID={category_id}: {e}")
            return False
    
    def read_csv_data(self) -> List[Dict[str, Any]]:
        """Read and parse the CSV data into a list of dictionaries."""
        raw_data = []
        line_count = 0  # 当前处理的行号
        
        print(f"读取CSV: 开始行={self.begin_line}, 结束行={self.end_line if self.end_line is not None else '文件末尾'}")
        
        with open(self.input_csv_path, 'r', encoding='utf-8-sig') as csvfile:  # 使用utf-8-sig处理BOM
            for line in csvfile:
                # 如果当前行号小于开始行，跳过
                if line_count < self.begin_line:
                    line_count += 1
                    continue
                
                # 如果指定了结束行且当前行号大于结束行，结束读取
                if self.end_line is not None and line_count > self.end_line:
                    break
                
                line = line.strip()
                if not line:
                    line_count += 1
                    continue
                    
                # 手动按tab分割
                parts = line.split('\t')
                if len(parts) < 2:
                    line_count += 1
                    continue
                    
                try:
                    # 处理ID和分类路径
                    id_num = int(parts[0])
                    category_path = parts[1].replace('>>', '>')
                    levels = [level.strip() for level in category_path.split('>')]
                    
                    raw_data.append({
                        "id": id_num,
                        "path": category_path,
                        "levels": levels,
                        "level_depth": len(levels)
                    })
                except ValueError as e:
                    print(f"Error processing line {line_count}: {line}, error: {e}")
                
                line_count += 1
        
        print(f"成功读取 {len(raw_data)} 条分类数据")
        return raw_data
    
    def generate_prompt(self, category_data: Dict[str, Any]) -> str:
        """Generate a prompt for the Deepseek API based on category data."""
        return f"""
我需要为以下商品分类创建详细的数据结构:

ID: {category_data['id']}
分类路径: {category_data['path']}
分类层次: {', '.join(category_data['levels'])}
层级深度: {category_data['level_depth']}

请生成以下属性:
1. description: 一段简短的中文描述，介绍该分类的商品特点和用途 (不超过50字)
2. keywords: 与该分类相关的关键词列表 (不超过8个)
3. examples: 该分类下可能的商品示例 (不超过5个)
4. exclusions: 不属于该分类的相关商品 (不超过5个)

请按照以下JSON格式返回，保留原有ID、path、levels和level_depth:
{{
  "id": {category_data['id']},
  "path": "{category_data['path']}",
  "levels": {json.dumps(category_data['levels'], ensure_ascii=False)},
  "level_depth": {category_data['level_depth']},
  "description": "这里是分类描述...",
  "keywords": ["关键词1", "关键词2", ...],
  "examples": ["示例1", "示例2", ...],
  "exclusions": ["排除项1", "排除项2", ...]
}}

只返回JSON格式的数据，不要有其他解释文字。
"""
    
    def call_deepseek_api(self, prompt: str) -> Dict[str, Any]:
        """Call Deepseek API with the generated prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个商品分类专家，能够生成详细的商品分类描述和相关信息。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract content from response
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Try to parse the JSON from the content
            try:
                # Find JSON in the response if there's any surrounding text
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    return json.loads(json_content)
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from API response: {content}")
                return None
            
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def process_categories(self, max_categories: int = None, delay: float = 1.0):
        """
        Process all categories, call the API for each, and compile results.
        
        Args:
            max_categories: Optional limit on number of categories to process
            delay: Delay between API calls in seconds
        """
        raw_data = self.read_csv_data()
        processed_data = []
        
        # 创建输出文件目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(self.output_json_path)), exist_ok=True)
        
        if max_categories:
            raw_data = raw_data[:max_categories]
        
        total = len(raw_data)
        for i, category in enumerate(raw_data, 1):
            category_id = category['id']
            print(f"处理分类 {i}/{total}: ID={category_id}")
            
            # 检查单独文件是否存在，存在则直接使用并跳过API调用
            if self.check_single_file_exists(category_id):
                print(f"ID={category_id}已有单独文件，直接跳过")
                
                # 尝试从单文件加载数据到processed_data列表
                try:
                    with open(self.get_single_file_path(category_id), 'r', encoding='utf-8') as jsonfile:
                        category_data = json.load(jsonfile)
                        processed_data.append(category_data)
                except Exception as e:
                    print(f"读取单独文件失败: {e}，但会继续处理其他分类")
                
                continue
            
            # 单独文件不存在，调用API获取数据
            print(f"ID={category_id}未找到单独文件，调用Deepseek API...")
            prompt = self.generate_prompt(category)
            api_result = self.call_deepseek_api(prompt)
            
            if api_result:
                processed_data.append(api_result)
                self.save_single_category_file(api_result)
                print(f"成功处理分类 ID={category_id}，数据已保存到单独文件")
            else:
                # API调用失败，使用基础数据模板
                basic_category = {
                    "id": category_id,
                    "path": category["path"],
                    "levels": category["levels"],
                    "level_depth": category["level_depth"],
                    "description": f"提供{category['levels'][-1]}相关的产品和服务",
                    "keywords": [level for level in category["levels"]],
                    "examples": [f"{category['levels'][-1]}产品示例"],
                    "exclusions": [f"非{category['levels'][-1]}相关产品"]
                }
                processed_data.append(basic_category)
                self.save_single_category_file(basic_category)
                print(f"API调用失败，使用基础数据模板 ID={category_id}")
            
            # 添加延迟避免API限流
            if i < total and delay > 0:
                time.sleep(delay)
        
        print(f"处理完成: 共处理 {len(processed_data)} 个分类")
        return processed_data

def main():
    """主函数入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate detailed category data using Deepseek API')
    parser.add_argument('--input', default='data/内网分类.csv', help='Input CSV file path')
    parser.add_argument('--output', default='data/new_categories.json', help='Output JSON file path')
    parser.add_argument('--api-key', help='Deepseek API key (can also use DEEPSEEK_API_KEY env var)')
    parser.add_argument('--max', type=int, help='Maximum number of categories to process')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds')
    parser.add_argument('-b', '--begin', type=int, default=0, help='开始处理的CSV行号(0-indexed)')
    parser.add_argument('-e', '--end', type=int, help='结束处理的CSV行号(0-indexed)，不指定则处理到文件末尾')
    
    args = parser.parse_args()
    
    generator = CategoryGenerator(
        input_csv_path=args.input, 
        output_json_path=args.output,
        api_key=args.api_key,
        begin_line=args.begin,
        end_line=args.end
    )
    generator.process_categories(max_categories=args.max, delay=args.delay)

if __name__ == "__main__":
    main() 