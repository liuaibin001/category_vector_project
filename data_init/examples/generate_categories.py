"""示例：如何使用数据初始化工具生成分类数据"""

import os
from pathlib import Path
from data_init.scripts.category_generator import CategoryGenerator
from data_init.scripts.merge_categories import merge_category_files

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent.parent  # 项目根目录
    data_dir = base_dir / "data"
    
    # 确保目录存在
    data_dir.mkdir(exist_ok=True)
    (data_dir / "single_file").mkdir(exist_ok=True)
    
    # 步骤1: 从CSV生成单个分类文件
    input_csv = data_dir / "内网分类.csv"
    output_json = data_dir / "new_categories.json"
    
    # 从环境变量获取API密钥
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    print("步骤1: 从CSV生成分类数据")
    print(f"输入文件: {input_csv}")
    print(f"输出目录: {data_dir}")
    
    # 创建生成器实例
    generator = CategoryGenerator(
        input_csv_path=str(input_csv),
        output_json_path=str(output_json),
        api_key=api_key
    )
    
    # 处理分类（这里设置最大处理5个分类作为示例）
    generator.process_categories(max_categories=5, delay=1.0)
    
    # 步骤2: 合并单个分类文件
    print("\n步骤2: 合并分类文件")
    single_file_dir = data_dir / "single_file"
    categories_json = data_dir / "categories.json"
    
    print(f"合并目录: {single_file_dir}")
    print(f"输出文件: {categories_json}")
    
    # 合并文件
    categories = merge_category_files(
        input_dir=str(single_file_dir),
        output_file=str(categories_json)
    )
    
    print("\n数据初始化完成!")
    print(f"- 生成的单个分类文件位于: {single_file_dir}")
    print(f"- 合并后的分类文件位于: {categories_json}")
    print(f"- 总计处理 {len(categories)} 个分类")

if __name__ == "__main__":
    main() 