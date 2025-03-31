# CategoryVector

分类向量搜索工具，支持高效的类别向量生成和相似性搜索。

## 功能

- 基于Sentence Transformers生成类别的语义向量表示
- 使用Milvus向量数据库进行高效的相似性搜索
- 支持分层搜索和按相似度阈值过滤结果
- 全面的配置系统，支持通过配置文件和命令行参数配置

## 安装

使用Poetry安装:

```bash
# 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

## 配置

CategoryVector支持通过配置文件或命令行参数进行配置。配置文件采用TOML格式，默认位置为项目根目录的`config.toml`。

### 配置文件结构

```toml
# CategoryVector 配置文件

[milvus]
host = "192.168.3.9"     # Milvus服务器地址
port = "19530"           # Milvus服务器端口
collection_name = "category_vectors"  # 集合名称

[search]
threshold = 0.3          # 相似度阈值，低于此值的结果将被过滤
top_k = 10               # 返回的最大结果数量
similarity_metric = "IP" # 相似度度量: IP (余弦相似度), L2 (欧氏距离)

[index]
type = "flat"            # 索引类型: flat, ivf, hnsw
nlist = 100              # IVF索引的聚类数量
m_factor = 16            # HNSW索引的连接数

[model]
name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 模型名称
vector_dim = 384         # 向量维度

[logging]
level = "INFO"           # DEBUG, INFO, WARNING, ERROR
file = "logs/categoryvector.log"  # 日志文件路径

[data]
output_dir = "data/vectors"  # 默认输出目录
```

### 配置优先级

1. 命令行参数优先级最高
2. 配置文件次之
3. 程序默认值最低

### 配置位置

系统按以下顺序查找配置文件:

1. 通过命令行参数`--config`指定的路径
2. 环境变量`CATEGORYVECTOR_CONFIG`指定的路径
3. 当前目录下的`config.toml`
4. 当前目录下的`configs/config.toml`
5. 用户主目录下的`.config/categoryvector/config.toml`

## 使用方法

### 构建索引

```bash
poetry run cv --help  # 显示主命令帮助
poetry run cv build --help  # 显示构建命令帮助
poetry run cv search --help  # 显示搜索命令帮助
poetry run cv update --help  # 显示更新命令帮助
```

```bash
# 基本用法
poetry run cv build --categories data/categories.json

# 带参数的用法
poetry run cv  build --categories data/valid_sample.json --output data/my_vectors --verbose
```

### 搜索分类

```bash
# 基本搜索
poetry run cv search --index data/vectors --query "手机腰包"

# 带参数的搜索
poetry run cv search --index data/vectors --query "手机腰包" --top-k 5 --threshold 0.4
```

### 更新分类

```bash
# 更新指定ID的分类
poetry run cv update --index data/vectors --category-id 2001738
```

### 启动API服务

```bash
# 基本启动
poetry run serve

# 开发模式（带热重载）
poetry run serve --reload --port 8080
```

## 向量搜索优化

本项目已进行了多项优化，提高向量搜索的准确性：

1. **使用余弦相似度**：默认使用余弦相似度(Inner Product)作为度量标准，更适合文本语义匹配
2. **强化核心产品类型权重**：核心产品类型词在向量生成时获得更高权重，提高匹配精度
3. **降低默认相似度阈值**：阈值从0.6调整为0.3，允许更宽泛的匹配结果
4. **查询词加权处理**：对查询中的关键产品词进行加权，提高查找相关产品的准确性

# Category Vector API

基于向量搜索的分类查询系统，提供REST API接口。

## 功能特点

- 基于向量相似度的分类搜索
- 分类向量的更新
- 支持多层级分类搜索
- 使用Milvus作为向量数据库
- 提供RESTful API接口

## 安装

确保已安装Python 3.8或更高版本，以及pip。

```bash
# 克隆仓库
git clone https://github.com/yourusername/categoryvector.git
cd categoryvector

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 构建索引

在使用API之前，需要先构建分类向量索引：

```bash
python -m src.categoryvector.cli build --categories data/categories.json --output data/vectors
```

### 2. 启动API服务

```bash
# 基本启动方式
python -m src.categoryvector.main

# 自定义端口
python -m src.categoryvector.main --port 8080

# 开发模式（热重载）
python -m src.categoryvector.main --reload --log-level DEBUG
```

### 3. API使用

服务启动后，可以通过HTTP请求使用API：

#### 搜索分类

**POST请求：**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/search' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "iPad保护壳",
  "top_k": 5,
  "threshold": 0.3,
  "index_dir": "data/vectors"
}'
```

**GET请求：**

```bash
curl -X 'GET' \
  'http://localhost:8000/api/search?query=iPad保护壳&top_k=5&threshold=0.3&index_dir=data/vectors'
```

**响应：**

```json
{
  "query": "iPad保护壳",
  "results": [
    {
      "id": 123,
      "path": "电子产品 / 平板电脑 / 配件 / 保护壳",
      "level_depth": 4,
      "description": "iPad平板电脑保护壳",
      "similarity": 0.85,
      "keywords": ["保护壳", "iPad壳", "平板壳"]
    },
    ...
  ],
  "total_results": 5
}
```

#### 更新分类向量

**请求：**

```bash
curl -X 'POST' \
  'http://localhost:8000/api/update' \
  -H 'Content-Type: application/json' \
  -d '{
  "category_id": 123,
  "index_dir": "data/vectors"
}'
```

**响应：**

```json
{
  "category_id": 123,
  "status": "success",
  "message": "分类ID=123的向量已成功更新"
}
```

### 4. 接口文档

启动服务后，访问以下URL查看自动生成的API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 环境变量

可以通过环境变量配置服务：

- `MILVUS_HOST` - Milvus服务器地址，默认为localhost
- `MILVUS_PORT` - Milvus服务器端口，默认为19530
- `LOG_LEVEL` - 日志级别，可选值：DEBUG, INFO, WARNING, ERROR

## 依赖项

- FastAPI
- Uvicorn
- Pydantic
- Pymilvus
- Sentence-Transformers
- NumPy

## 许可证

MIT