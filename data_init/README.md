# 数据初始化工具

这个包提供了用于初始化和生成分类数据的工具。主要包括两个功能：

1. 从CSV文件生成分类数据
2. 合并单个分类文件到一个数组文件

## 目录结构

```
data_init/
├── README.md           # 本文件
├── __init__.py        # 包初始化文件
├── scripts/           # 脚本目录
│   ├── category_generator.py  # 分类生成器
│   └── merge_categories.py    # 分类合并工具
├── examples/          # 示例代码
│   └── generate_categories.py # 使用示例
└── data/             # 数据目录
    ├── single_file/  # 单个分类文件存储目录
    └── categories.json # 合并后的分类文件
```

## 使用方法

### 1. 从CSV生成分类数据

```python
from data_init.scripts.category_generator import CategoryGenerator

# 创建生成器实例
generator = CategoryGenerator(
    input_csv_path="data/内网分类.csv",
    output_json_path="data/new_categories.json",
    api_key="your-api-key"  # 或者设置环境变量 DEEPSEEK_API_KEY
)

# 处理分类
generator.process_categories(max_categories=5, delay=1.0)
```

### 2. 合并分类文件

```python
from data_init.scripts.merge_categories import merge_category_files

# 合并文件
categories = merge_category_files(
    input_dir="data/single_file",
    output_file="data/categories.json"
)
```

### 3. 使用示例脚本

```bash
# 设置API密钥
export DEEPSEEK_API_KEY=your-api-key

# 运行示例脚本
python -m data_init.examples.generate_categories
```

## 命令行使用

### 生成分类数据

```bash
python -m data_init.scripts.category_generator \
    --input data/内网分类.csv \
    --output data/new_categories.json \
    --api-key your-api-key \
    --max 5 \
    --delay 1.0
```

参数说明：
- `--input`: 输入CSV文件路径
- `--output`: 输出JSON文件路径
- `--api-key`: Deepseek API密钥（也可以通过环境变量设置）
- `--max`: 最大处理分类数量
- `--delay`: API调用间隔时间（秒）
- `-b/--begin`: 开始处理的CSV行号(0-indexed)
- `-e/--end`: 结束处理的CSV行号(0-indexed)

### 合并分类文件

```bash
python -m data_init.scripts.merge_categories \
    --input data/single_file \
    --output data/categories.json
```

参数说明：
- `--input`: 包含单个分类JSON文件的目录路径
- `--output`: 输出的合并JSON文件路径

## 注意事项

1. 确保设置了 `DEEPSEEK_API_KEY` 环境变量或在调用时提供API密钥
2. 生成的单个分类文件会保存在 `single_file` 目录下
3. 合并后的分类文件会按ID排序
4. 建议设置适当的API调用延迟，避免触发限流 