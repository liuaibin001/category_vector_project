"""API模块"""

from fastapi import FastAPI, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import os
from pathlib import Path
import time
from pymilvus import connections
import uvicorn

from .cli import search as cli_search, update_category as cli_update_category
from .config import CategoryVectorConfig
from .models import Category
from .vector_generation import VectorGenerator
from .vector_storage import VectorStorage
from .utils.logging_utils import setup_logger, default_logger as logger

# 创建FastAPI应用
app = FastAPI(
    title="Category Vector API",
    description="基于向量搜索的分类API",
    version="1.0.0"
)

# 定义模型
class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索关键词，必填")
    top_k: Optional[int] = Field(15, description="返回结果数量")
    threshold: Optional[float] = Field(0.1, description="相似度阈值 (0-1)")
    level: Optional[int] = Field(None, description="指定搜索层级")
    verbose: Optional[bool] = Field(False, description="是否显示详细日志")
    index_dir: Optional[str] = Field("data/vectors", description="索引目录路径")

class UpdateRequest(BaseModel):
    category_id: int = Field(..., description="要更新的分类ID，必填")
    index_dir: Optional[str] = Field("data/vectors", description="索引目录路径")
    verbose: Optional[bool] = Field(False, description="是否显示详细日志")

class CategoryResult(BaseModel):
    id: int
    path: str
    level_depth: int
    description: Optional[str] = None
    similarity: float
    keywords: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[CategoryResult]
    total_results: int
    
class UpdateResponse(BaseModel):
    category_id: int
    status: str
    message: str

# 自定义参数处理类
class SearchArgs:
    def __init__(
        self, 
        query: str,
        index: str = "data/vectors",
        top_k: int = 15,
        threshold: float = 0.1,
        level: Optional[int] = None,
        verbose: bool = False,
        log_level: str = "ERROR",
        config: Optional[str] = None,
        milvus_host: Optional[str] = None,
        milvus_port: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.query = query
        self.index = index
        self.top_k = top_k
        self.threshold = threshold
        self.level = level
        self.verbose = verbose
        self.log_level = log_level
        self.config = config
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        
class UpdateArgs:
    def __init__(
        self,
        category_id: int,
        index: str = "data/vectors",
        verbose: bool = False,
        log_level: str = "INFO",
        config: Optional[str] = None,
        vector_dim: Optional[int] = None,
        model: Optional[str] = None,
        milvus_host: Optional[str] = None,
        milvus_port: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.category_id = category_id
        self.index = index
        self.verbose = verbose
        self.log_level = log_level
        self.config = config
        self.vector_dim = vector_dim
        self.model = model
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name

# 自定义异常捕获
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"发生错误: {str(exc)}"}
    )

# 创建搜索接口
@app.post("/api/search", response_model=SearchResponse, summary="搜索分类", description="根据查询文本搜索相似分类")
async def search_api(request: SearchRequest):
    # 设置日志
    log_level = "DEBUG" if request.verbose else "INFO"
    logger = setup_logger("categoryvector", level=log_level)
    
    try:
        # 构建CLI参数
        args = SearchArgs(
            query=request.query,
            index=request.index_dir,
            top_k=request.top_k,
            threshold=request.threshold,
            level=request.level,
            verbose=request.verbose
        )
        
        # 加载配置
        config = CategoryVectorConfig.from_toml()
        
        # 命令行参数覆盖配置文件
        top_k = args.top_k or config.top_k
        threshold = args.threshold or config.similarity_threshold
        milvus_host = args.milvus_host or config.milvus_host
        milvus_port = args.milvus_port or config.milvus_port
        collection_name = args.collection_name or config.collection_name
        
        logger.info(f"开始搜索过程...")
        
        # 检查索引目录
        index_path = Path(args.index)
        if not index_path.exists():
            raise HTTPException(status_code=404, detail=f"索引目录不存在: {index_path}")
            
        # 创建配置
        search_config = CategoryVectorConfig(
            model_name=config.model_name,
            data_dir=index_path,
            log_level=log_level,
            # 添加Milvus配置
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            top_k=top_k,
            similarity_threshold=threshold
        )
        
        # 创建存储实例并检查 Milvus 连通性
        logger.info("检查 Milvus 服务器连通性...")
        storage = VectorStorage(config.vector_dim, search_config)
        try:
            storage.connect_to_milvus()
        except ConnectionError as e:
            logger.error(f"Milvus 服务器连接失败: {e}")
            raise HTTPException(status_code=503, detail=f"Milvus服务器连接失败: {str(e)}")
        
        # 初始化Milvus集合
        logger.info("初始化Milvus集合...")
        try:
            storage.setup_collection()
        except Exception as e:
            logger.error(f"初始化Milvus集合失败: {e}")
            raise HTTPException(status_code=500, detail=f"初始化Milvus集合失败: {str(e)}")
            
        # 检查集合中是否有数据
        entity_count = storage.collection.num_entities if storage.collection else 0
        if entity_count == 0:
            logger.error("Milvus集合中没有向量数据。请先运行build命令构建索引并确保数据成功保存到Milvus。")
            raise HTTPException(
                status_code=404, 
                detail="Milvus集合中没有向量数据。请先运行build命令构建索引并确保数据成功保存到Milvus。"
            )
            
        # 生成查询向量
        logger.info(f"查询: {args.query}")
        generator = VectorGenerator(model_name=search_config.model_name, config=search_config)
        query_vector = generator.generate_query_vector(args.query)
        
        # 执行搜索
        if args.level:
            # 按层级搜索
            results = storage.search_by_level(
                query_vector,
                level=args.level,
                top_k=top_k,
                threshold=threshold
            )
        else:
            # 全局搜索
            results = storage.search(
                query_vector,
                top_k=top_k,
                threshold=threshold
            )
        
        # 格式化结果
        formatted_results = []
        for category, similarity in results:
            formatted_results.append(
                CategoryResult(
                    id=category.id,
                    path=category.path,
                    level_depth=category.level_depth,
                    description=category.description,
                    keywords=category.keywords,
                    similarity=float(similarity)
                )
            )
        
        # 构建响应
        response = SearchResponse(
            query=args.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )
        
        return response
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"搜索时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索时发生错误: {str(e)}")

# 添加GET方法支持的search接口
@app.get("/api/search", response_model=SearchResponse, summary="搜索分类(GET)", description="使用GET方法搜索分类")
async def search_api_get(
    query: str = Query(..., description="搜索关键词，必填"),
    top_k: int = Query(5, description="返回结果数量"),
    threshold: float = Query(0.3, description="相似度阈值 (0-1)"),
    level: Optional[int] = Query(None, description="指定搜索层级"),
    verbose: bool = Query(False, description="是否显示详细日志"),
    index_dir: str = Query("data/vectors", description="索引目录路径")
):
    # 创建请求对象
    request = SearchRequest(
        query=query,
        top_k=top_k,
        threshold=threshold,
        level=level,
        verbose=verbose,
        index_dir=index_dir
    )
    
    # 调用POST方法处理逻辑
    return await search_api(request)

# 创建更新接口
@app.post("/api/update", response_model=UpdateResponse, summary="更新分类向量", description="更新指定ID的分类向量")
async def update_api(request: UpdateRequest):
    start_time = time.time()
    
    try:
        # 构建CLI参数
        args = UpdateArgs(
            category_id=request.category_id,
            index=request.index_dir,
            verbose=request.verbose
        )
        
        # 加载配置
        config = CategoryVectorConfig.from_toml()
        
        # 设置日志
        log_level = "DEBUG" if request.verbose else "INFO"
        logger = setup_logger("categoryvector", level=log_level)
        
        # 参数覆盖配置
        model_name = args.model or config.model_name
        vector_dim = args.vector_dim or config.vector_dim
        milvus_host = args.milvus_host or config.milvus_host
        milvus_port = args.milvus_port or config.milvus_port
        collection_name = args.collection_name or config.collection_name
        
        logger.info(f"开始更新分类ID={args.category_id}的向量...")
        
        # 创建配置
        update_config = CategoryVectorConfig(
            model_name=model_name,
            data_dir=Path(args.index),
            log_level=log_level,
            vector_dim=vector_dim,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name=collection_name,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            nlist=config.nlist,
            m_factor=config.m_factor,
            index_type=config.index_type
        )
        
        # 尝试连接 Milvus
        try:
            logger.info(f"正在连接 Milvus 服务器: {milvus_host}:{milvus_port}...")
            try:
                # 断开可能存在的连接
                connections.disconnect("default")
            except:
                pass
                
            # 尝试建立连接
            connections.connect("default", host=milvus_host, port=milvus_port)
            logger.info(f"✓ Milvus 连接成功")
        except Exception as e:
            logger.error(f"✗ Milvus 连接失败: {milvus_host}:{milvus_port}, 错误: {e}")
            raise HTTPException(status_code=503, detail=f"Milvus服务器连接失败: {str(e)}")
            
        # 创建存储实例
        storage = VectorStorage(dimension=vector_dim, config=update_config)
            
        # 加载索引目录
        index_path = Path(args.index)
        if not index_path.exists():
            logger.error(f"索引目录不存在: {index_path}")
            raise HTTPException(status_code=404, detail=f"索引目录不存在: {index_path}")
            
        # 加载现有类别数据
        logger.info(f"正在加载索引数据...")
        try:
            storage.load(index_path)
        except Exception as e:
            logger.error(f"✗ 加载索引数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"加载索引数据失败: {str(e)}")
            
        if not storage.categories:
            logger.error("✗ 索引中没有分类数据")
            raise HTTPException(status_code=404, detail="索引中没有分类数据")
            
        # 查找要更新的分类
        category_id = args.category_id
        if category_id not in storage.categories:
            logger.error(f"✗ 未找到分类ID={category_id}，无法更新")
            raise HTTPException(status_code=404, detail=f"未找到分类ID={category_id}，无法更新")
            
        category = storage.categories[category_id]
        logger.info(f"✓ 找到分类: ID={category.id}, 路径={category.path}")
        
        # 重新生成向量
        logger.info(f"正在加载模型: {update_config.model_name}...")
        generator = VectorGenerator(model_name=update_config.model_name, config=update_config)
        
        logger.info(f"为分类ID={category.id}重新生成向量...")
        try:
            # 先清除旧向量
            category.vector = None
            category.level_vectors = {}
            
            # 生成新向量
            updated_category = generator.enrich_category_vectors(category)
            logger.info(f"✓ 分类ID={category.id}的向量生成成功")
            
            # 更新到Milvus
            logger.info(f"正在更新 Milvus 中的分类向量...")
            storage.add_category(updated_category)
            logger.info(f"✓ Milvus 向量更新成功")
            
            # 保存更新后的数据
            logger.info(f"正在保存到索引目录: {index_path}...")
            storage.save(index_path)
            logger.info(f"✓ 索引数据保存完成")
            
            # 更新 Redis
            logger.info(f"正在更新 Redis...")
            from categoryvector.utils.redis_client import redis_client
            redis_key = f"categories:{category_id}"
            redis_updated = False
            redis_message = ""
            
            try:
                # 检查 Redis 客户端是否可用
                if redis_client.client is not None:
                    # 将分类对象转换为字典
                    category_dict = {
                        "id": updated_category.id,
                        "path": updated_category.path,
                        "levels": updated_category.levels,
                        "level_depth": updated_category.level_depth,
                        "description": updated_category.description,
                        "keywords": updated_category.keywords,
                        "examples": updated_category.examples,
                        "exclusions": updated_category.exclusions,
                        "vector": updated_category.vector.tolist() if hasattr(updated_category, 'vector') and updated_category.vector is not None else None,
                        "level_vectors": {
                            k: v.tolist() for k, v in updated_category.level_vectors.items()
                        } if updated_category.level_vectors else {}
                    }
                    redis_client.set(redis_key, category_dict, 60*60*24*30)
                    logger.info(f"✓ Redis 更新成功")
                    redis_updated = True
                    redis_message = "Redis更新成功"
                else:
                    logger.warning(f"✗ Redis 连接不可用，无法更新")
                    redis_message = "Redis连接不可用，无法更新"
            except Exception as e:
                logger.error(f"✗ Redis 更新失败: {e}")
                redis_message = f"Redis更新失败: {str(e)}"
            
            total_time = time.time() - start_time
            
            # 构建详细响应信息
            response_message = (
                f"分类ID={category_id}的向量已成功更新。"
                f"路径: {category.path}，"
                f"向量维度: {vector_dim}"
            )
            
            if redis_message:
                response_message += f"，{redis_message}"
                
            response_message += f"，总耗时: {total_time:.2f}秒"
            
            # 返回结果
            return UpdateResponse(
                category_id=category_id,
                status="success",
                message=response_message
            )
            
        except Exception as e:
            logger.error(f"✗ 更新分类向量时出错: {e}")
            raise HTTPException(status_code=500, detail=f"更新分类向量时出错: {str(e)}")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"更新分类时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新分类时发生错误: {str(e)}")

# 添加健康检查接口
@app.get("/health", summary="健康检查", description="检查API服务是否正常运行")
async def health_check():
    return {"status": "healthy", "message": "服务正常运行"}

# 添加服务信息接口
@app.get("/", summary="服务信息", description="获取API服务的基本信息")
async def get_info():
    return {
        "name": "Category Vector API",
        "version": "1.0.0",
        "description": "基于向量搜索的分类API服务",
        "endpoints": [
            {"path": "/api/search", "method": "POST", "description": "搜索分类"},
            {"path": "/api/search", "method": "GET", "description": "搜索分类(GET方式)"},
            {"path": "/api/update", "method": "POST", "description": "更新分类向量"},
            {"path": "/health", "method": "GET", "description": "健康检查"}
        ]
    } 