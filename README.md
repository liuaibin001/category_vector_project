# CategoryVector

CategoryVector 是一个用于产品分类向量生成和搜索的Python工具包。它能够根据产品分类的文本描述生成向量表示，存储到向量数据库中，并通过相似度搜索返回最匹配的分类。

## 项目功能

- **分类向量生成**：使用Sentence Transformers模型将分类文本转换为向量表示
- **层级向量支持**：为分类的每个层级生成独立的向量表示
- **向量索引管理**：使用FAISS进行高效的向量存储和检索
- **多模式搜索**：
  - 全局搜索：在所有分类中搜索相似项
  - 层级搜索：在特定层级的分类中搜索
- **排序与筛选**：按相似度排序并返回指定数量的结果

## 数据结构

CategoryVector 使用以下数据结构表示分类：

```json
[
  {
    "id": 1,
    "path": "手机和平板>平板贴纸",
    "levels": ["手机和平板", "平板贴纸"],
    "level_depth": 2,
    "description": "适用于各种平板电脑的保护贴纸，美化平板外观，提供防刮保护",
    "keywords": ["平板", "贴纸", "保护膜", "iPad贴纸", "装饰贴"],
    "examples": ["iPad Pro 贴纸", "华为MatePad贴膜", "小米平板装饰贴"],
    "exclusions": ["手机贴纸", "手机膜"],
    "vector": [...],  // 整体向量表示
    "level_vectors": {
      "level_1": [...],  // 一级分类向量
      "level_2": [...]   // 二级分类向量
    }
  }
]
```

## 项目流程

1. **数据准备**：
   - 准备包含分类层级、描述等信息的JSON文件
   - 格式为数组格式：`[{分类1}, {分类2}, ...]`

2. **向量生成流程**：
   - 加载分类数据
   - 使用Sentence Transformers模型生成向量
   - 为每个分类生成完整向量和层级向量
   - 构建FAISS索引
   - 保存索引和分类数据

3. **搜索流程**：
   - 加载预先构建的索引
   - 将查询文本转换为向量
   - 使用FAISS进行向量相似度搜索
   - 按相似度排序并返回结果

## 安装说明

### 环境需求

- Python 3.8+
- PyTorch 2.0+
- FAISS
- Sentence Transformers

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/categoryvector.git
cd categoryvector

# 使用Poetry安装依赖
poetry install

# 或使用pip安装
pip install -r requirements.txt
```

## 使用方法

### 构建索引

将产品分类数据转换为向量并构建索引：

```bash
# 使用默认参数
python src/categoryvector/cli.py build --categories data/categories.json --output data/vectors

# 自定义参数
python src/categoryvector/cli.py build \
    --categories data/categories.json \
    --output data/vectors \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --vector-dim 384

# 使用Poetry运行（确保在项目虚拟环境中执行）
poetry run build --categories data/categories.json --output data/vectors --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --vector-dim 384
```

参数说明：
- `--categories, -c`：分类数据JSON文件路径
- `--output, -o`：索引输出目录
- `--model, -m`：使用的模型名称（默认为paraphrase-multilingual-MiniLM-L12-v2）
- `--vector-dim, -d`：向量维度（默认为384）

### 搜索分类

在已构建的索引中搜索最匹配的分类：

```bash
# 基本搜索（全局）
python src/categoryvector/cli.py search --index data/vectors --query "平板电脑贴纸"

# 自定义搜索参数
python src/categoryvector/cli.py search \
    --index data/vectors \
    --query "平板电脑贴纸" \
    --top-k 5 \
    --threshold 0.3

# 层级搜索
python src/categoryvector/cli.py search \
    --index data/vectors \
    --query "电脑配件" \
    --level 1 \
    --top-k 3

# 使用Poetry运行（方便项目依赖管理）
poetry run search --index data/vectors --query "华为笔记本电脑键盘" --threshold 0.3 --top-k 3 --level 1
```

参数说明：
- `--index, -i`：索引目录路径，存放FAISS索引和分类数据的目录
- `--query, -q`：搜索查询文本，用于寻找相似分类的关键词或短语
- `--top-k, -k`：返回结果数量（默认为5），按相似度从高到低排序
- `--threshold, -t`：相似度阈值（默认为0.6），低于此值的结果将被过滤
- `--level, -l`：指定搜索层级，例如1表示只在一级分类中搜索，2表示二级分类

当使用 `poetry run search` 命令时，它会在Poetry虚拟环境中运行，自动处理所有依赖。使用 `--level 1` 参数可以专注于搜索一级分类，对于像"华为笔记本电脑键盘"这样包含多个层级概念的查询特别有用。

## 示例说明

### 准备数据

创建一个包含以下内容的`data/categories.json`文件：

```json
[
  {
    "id": 1,
    "path": "手机和平板>平板贴纸",
    "levels": ["手机和平板", "平板贴纸"],
    "level_depth": 2,
    "description": "适用于各种平板电脑的保护贴纸，美化平板外观，提供防刮保护",
    "keywords": ["平板", "贴纸", "保护膜", "iPad贴纸", "装饰贴"],
    "examples": ["iPad Pro 贴纸", "华为MatePad贴膜", "小米平板装饰贴"],
    "exclusions": ["手机贴纸", "手机膜"]
  },
  {
    "id": 2,
    "path": "手机和平板>手机贴纸",
    "levels": ["手机和平板", "手机贴纸"],
    "level_depth": 2,
    "description": "适用于各种型号手机的装饰贴纸，多种图案可选，易贴易撕",
    "keywords": ["手机", "贴纸", "装饰膜", "iPhone贴纸", "手机装饰"],
    "examples": ["iPhone 13贴纸", "三星Galaxy贴膜", "华为P50装饰贴"],
    "exclusions": ["平板贴纸", "手机保护壳"]
  }
]
```

### 构建索引

执行以下命令构建索引：

```bash
python src/categoryvector/cli.py build --categories data/categories.json --output data/vectors
```

### 搜索分类

执行以下命令搜索分类：

```bash
python src/categoryvector/cli.py search --index data/vectors --query "平板电脑贴纸" --threshold 0.3
```

预期输出：

```
查询: 平板电脑贴纸
======================================================================
找到 2 个结果，显示前 2 个（相似度阈值 0.3）
======================================================================
排名   ID   相似度       路径                  描述
----------------------------------------------------------------------
1.    1    0.4016    手机和平板>平板贴纸      适用于各种平板电脑的保护贴纸...
2.    2    0.3287    手机和平板>手机贴纸      适用于各种型号手机的装饰贴纸...
```

## 技术实现

### 向量生成

- **完整向量**：综合分类路径、描述、关键词和示例，生成整体语义向量
- **层级向量**：为每个层级单独生成向量，支持层级搜索

### 相似度计算

- **全局搜索**：使用L2距离转换为相似度分数
- **层级搜索**：使用余弦相似度计算

## AI接入指南

如需将CategoryVector集成到AI应用中，可通过以下方式：

1. **模块导入**：
```python
from categoryvector.vector_storage import VectorStorage
from categoryvector.vector_generation import VectorGenerator
```

2. **向量生成**：
```python
generator = VectorGenerator()
query_vector = generator.generate_query_vector("平板电脑贴纸")
```

3. **向量搜索**：
```python
storage = VectorStorage(384)
storage.load("data/vectors")
results = storage.search(query_vector, top_k=5, threshold=0.3)
```

4. **结果处理**：
```python
for category, score in results:
    print(f"分类: {category.path}, 相似度: {score:.4f}")
```

## 高级功能

### 自定义相似度阈值

通过调整`threshold`参数，可以控制结果的质量。较高的阈值会返回更精确但可能更少的结果。

### 层级特定搜索

使用`--level`参数可以在特定层级搜索，例如只在一级分类或二级分类中搜索。

### 返回数量控制

使用`--top-k`参数可以控制返回的结果数量，按相似度从高到低排序。

## 故障排除

- **索引加载失败**：检查索引路径是否正确
- **搜索无结果**：尝试降低相似度阈值（`--threshold`）
- **模型加载错误**：确保正确安装所有依赖，并使用受支持的模型名称
