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

- Python 3.11+
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

### 环境准备

1. 安装Python依赖项：

```bash
pip install -e .
# 或者使用poetry
poetry install
```

2. 安装并启动Milvus服务：

```bash
# 使用Docker启动Milvus（推荐）
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.1-latest standalone
MILVUS_URL 可视化UI
docker run -d --restart=always -p 8999:3000 -e MILVUS_URL=http://192.168.3.9:19530 zilliz/attu:latest
```

更多Milvus安装选项，请参考[Milvus官方文档](https://milvus.io/docs/install_standalone-docker.md)。

### 构建索引

执行以下命令构建索引：

```bash
python src/categoryvector/cli.py build --categories data/categories.json --output data/vectors
```

也可以配置Milvus连接信息：

```bash
python src/categoryvector/cli.py build \
    --categories data/categories.json \
    --output data/vectors \
    --milvus-host localhost \
    --milvus-port 19530
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

### 向量存储

- 使用Milvus作为向量数据库，支持高效的向量相似度搜索
- 支持多种索引类型：FLAT、IVF、HNSW等
- 支持百万级以上的向量数据管理和搜索

### 向量生成

- **完整向量**：综合分类路径、描述、关键词和示例，生成整体语义向量
- **层级向量**：为每个层级单独生成向量，支持层级搜索

### 相似度计算

- **全局搜索**：使用L2距离转换为相似度分数
- **层级搜索**：使用余弦相似度计算

## 高级配置

### Milvus配置

在`config.py`中可以配置Milvus连接信息和索引类型：

```python
milvus_host: str = "localhost"  # Milvus服务器地址
milvus_port: str = "19530"      # Milvus服务器端口
collection_name: str = "category_vectors"  # 集合名称
index_type: str = "flat"        # 索引类型：flat, ivf, hnsw
```

### 索引类型

- **FLAT**：精确搜索，适合小规模数据集
- **IVF**：基于聚类的近似搜索，适合中等规模数据集
- **HNSW**：基于图的近似搜索，适合大规模数据集，搜索速度快

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

## 最新更新与最佳实践

### 索引去重功能

CategoryVector 现在支持在构建索引时自动检查并覆盖已存在的分类ID，避免重复数据：

- 插入向量前会先检查Milvus中是否存在相同ID的分类数据
- 如果存在，会先删除旧记录，再插入新记录
- 此功能确保每个分类ID只有一条最新的记录，避免查询时出现重复结果

示例代码展示了此功能的实现：

```python
# 检查是否已存在该分类ID的数据
if self.collection.num_entities > 0:
    expr = f"category_id == {category_id}"
    result = self.collection.query(expr=expr, output_fields=["pk", "category_id"])
    
    # 如果找到匹配的记录，先删除
    if result and len(result) > 0:
        pk_to_delete = [r["pk"] for r in result]
        self.collection.delete(f"pk in {pk_to_delete}")
```

### HNSW索引推荐

经过测试分析，我们推荐使用HNSW索引类型用于产品标题匹配场景，理由如下：

1. **高速查询能力**：HNSW (Hierarchical Navigable Small World) 索引基于图结构，在保持高召回率的同时提供更快的查询速度
2. **高质量搜索结果**：与其他近似索引相比，HNSW能保持较高的召回率
3. **适用于产品标题匹配**：特别适合语义相似度高的文本匹配场景，如产品标题到分类的映射

使用HNSW索引构建方法：

```bash
python -m src.categoryvector.cli build --categories data/categories.json --output data/vectors_hnsw --milvus-host 192.168.3.9 --milvus-port 19530 --index-type hnsw --verbose
```

### 产品标题匹配最佳实践

在使用CategoryVector进行产品标题匹配时，我们建议以下最佳实践：

1. **调整相似度阈值**：根据具体业务需求，合理设置相似度阈值（推荐0.3-0.6之间）
   ```bash
   python -m src.categoryvector.cli search --index data/vectors_hnsw --query "手机贴纸" --threshold 0.3
   ```

2. **使用更多样化的训练数据**：在categories.json中包含：
   - 丰富的关键词列表（keywords字段）
   - 多种典型示例（examples字段）
   - 详细的描述信息（description字段）

3. **搜索结果处理策略**：
   - 当相似度超过0.5时，可直接采用最高匹配结果
   - 当相似度在0.3-0.5之间，可返回多个候选结果供选择
   - 当所有结果相似度低于0.3时，考虑使用后备方案或进行二次查询

4. **定期更新索引**：随着产品分类体系的变化，定期重建索引以保持最新状态

### 常见问题解决

1. **无法找到匹配结果**：
   - 降低相似度阈值（--threshold参数）
   - 检查查询词是否与分类数据中的关键词相匹配
   - 确保使用了正确的多语言模型支持查询语言

2. **查询结果出现重复**：
   - 确认是否使用了最新版本的代码（含去重功能）
   - 重新构建索引以应用去重逻辑
   - 使用group_by功能（如果集成自定义Milvus客户端）

3. **性能优化建议**：
   - 对于大规模数据集（>100万条），使用HNSW索引并调整M参数（默认16）
   - 对于搜索速度要求高的场景，可考虑使用内存更大的服务器
   - 使用持久化连接模式避免重复建立Milvus连接
