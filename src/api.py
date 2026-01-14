"""
FastAPI 服务 - 文件上传与检索 API
"""
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings

app = FastAPI(
    title="医疗文献 RAG API",
    description="文件上传、索引、检索服务",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 数据模型 ==============

class UploadResponse(BaseModel):
    """上传响应"""
    success: bool
    file_id: str
    filename: str
    message: str


class DocumentInfo(BaseModel):
    """文档信息"""
    file_id: str
    filename: str
    size_bytes: int
    status: str  # pending, indexed, failed


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int = Field(default=10, ge=1, le=100, description="返回数量")
    alpha: float = Field(default=0.5, ge=0, le=1, description="混合检索权重")
    enable_rerank: bool = Field(default=True, description="是否启用重排序")
    enable_generation: bool = Field(default=False, description="是否生成答案")


class SearchResult(BaseModel):
    """单条检索结果"""
    text: str
    source: str
    score: float
    # 元数据
    title: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None
    chunk_index: Optional[int] = None


class SearchResponse(BaseModel):
    """检索响应"""
    query: str
    results: List[SearchResult]
    answer: Optional[str] = None
    total: int


class IndexRequest(BaseModel):
    """索引请求"""
    file_ids: Optional[List[str]] = Field(default=None, description="指定文件 ID，为空则索引全部待处理文件")


class IndexResponse(BaseModel):
    """索引响应"""
    success: bool
    indexed_count: int
    failed_count: int
    message: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    healthy: bool
    services: dict


# ============== 全局状态 ==============

# 简单的文件状态追踪（生产环境应用数据库）
_file_registry: dict = {}  # file_id -> {filename, path, status}


def _get_file_path(file_id: str) -> Optional[Path]:
    """根据 file_id 获取文件路径"""
    if file_id in _file_registry:
        return Path(_file_registry[file_id]["path"])
    return None


# ============== API 端点 ==============

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    from src.health import HealthChecker

    checker = HealthChecker()
    result = checker.check_all(include_llm=False)

    return HealthResponse(
        healthy=result.healthy,
        services={s.name: {"healthy": s.healthy, "message": s.message} for s in result.services},
    )


@app.post("/upload", response_model=UploadResponse, tags=["文件管理"])
async def upload_file(file: UploadFile = File(..., description="PDF 文件")):
    """
    上传 PDF 文件

    文件将保存到 data/documents/ 目录，等待索引。
    """
    # 验证文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 生成唯一 ID
    file_id = str(uuid.uuid4())[:8]
    safe_filename = f"{file_id}_{file.filename}"
    save_path = settings.DOCUMENTS_DIR / safe_filename

    try:
        # 保存文件
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 记录文件信息
        _file_registry[file_id] = {
            "filename": file.filename,
            "path": str(save_path),
            "status": "pending",
            "size_bytes": save_path.stat().st_size,
        }

        logger.info(f"文件上传成功: {file_id} -> {safe_filename}")

        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=file.filename,
            message="上传成功，等待索引",
        )

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/files", response_model=List[DocumentInfo], tags=["文件管理"])
async def list_files():
    """列出所有已上传的文件"""
    result = []
    for file_id, info in _file_registry.items():
        result.append(
            DocumentInfo(
                file_id=file_id,
                filename=info["filename"],
                size_bytes=info.get("size_bytes", 0),
                status=info["status"],
            )
        )
    return result


@app.delete("/files/{file_id}", tags=["文件管理"])
async def delete_file(file_id: str):
    """删除指定文件"""
    if file_id not in _file_registry:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_path = Path(_file_registry[file_id]["path"])
    if file_path.exists():
        file_path.unlink()

    del _file_registry[file_id]
    logger.info(f"文件已删除: {file_id}")

    return {"success": True, "message": "文件已删除"}


@app.post("/index", response_model=IndexResponse, tags=["索引"])
async def index_documents(request: IndexRequest = None):
    """
    索引文档到向量数据库

    如果不指定 file_ids，则索引所有状态为 pending 的文件。
    """
    from src.document_loader import MinerUDocumentLoader
    from src.embedder import MedicalEmbedder
    from src.embedding_cache import EmbeddingCache
    from src.vector_store import VectorStore

    # 确定要索引的文件
    if request and request.file_ids:
        file_ids = request.file_ids
    else:
        file_ids = [fid for fid, info in _file_registry.items() if info["status"] == "pending"]

    if not file_ids:
        return IndexResponse(
            success=True,
            indexed_count=0,
            failed_count=0,
            message="没有待索引的文件",
        )

    # 初始化组件
    try:
        loader = MinerUDocumentLoader(
            backend=settings.MINERU_BACKEND,
            output_dir=str(settings.PARSED_DIR),
        )
        embedder = MedicalEmbedder(
            model_name=settings.EMBEDDING_MODEL,
            instruction=settings.MEDICAL_INSTRUCT,
            max_length=settings.MAX_SEQ_LENGTH,
        )
        cache = EmbeddingCache(cache_dir=settings.EMBEDDING_CACHE_DIR)
        vector_store = VectorStore()
    except Exception as e:
        logger.error(f"初始化组件失败: {e}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

    indexed_count = 0
    failed_count = 0

    for file_id in file_ids:
        if file_id not in _file_registry:
            failed_count += 1
            continue

        file_info = _file_registry[file_id]
        file_path = Path(file_info["path"])

        if not file_path.exists():
            file_info["status"] = "failed"
            failed_count += 1
            continue

        try:
            # 解析 + 分块
            chunks = loader.load(
                str(file_path),
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

            if not chunks:
                file_info["status"] = "failed"
                failed_count += 1
                continue

            # 生成嵌入
            texts = [c["text"] for c in chunks]
            metadatas = [c.get("metadata", {}) for c in chunks]

            embeddings = cache.get_or_compute(
                texts,
                lambda batch: embedder.encode_documents(batch, batch_size=settings.BATCH_SIZE, show_progress=False),
            )

            # 入库
            vector_store.insert(embeddings=embeddings, texts=texts, metadata=metadatas)

            file_info["status"] = "indexed"
            indexed_count += 1
            logger.info(f"索引成功: {file_id}, {len(chunks)} 个文本块")

        except Exception as e:
            logger.error(f"索引文件失败 {file_id}: {e}")
            file_info["status"] = "failed"
            failed_count += 1

    return IndexResponse(
        success=failed_count == 0,
        indexed_count=indexed_count,
        failed_count=failed_count,
        message=f"索引完成: 成功 {indexed_count}, 失败 {failed_count}",
    )


@app.post("/search", response_model=SearchResponse, tags=["检索"])
async def search(request: SearchRequest):
    """
    检索医疗文献

    支持混合检索（向量 + BM25）和可选的重排序。
    """
    from src.pipeline import MedicalRAGPipeline, RAGConfig

    try:
        config = RAGConfig(
            alpha=request.alpha,
            final_top_k=request.top_k,
            enable_generation=request.enable_generation,
            stream_output=False,
        )
        pipeline = MedicalRAGPipeline(config=config)

        result = pipeline.query(
            query=request.query,
            enable_generation=request.enable_generation,
        )

        # 转换结果格式
        search_results = []
        for doc in result.documents:
            entity = doc.get("entity", doc)
            
            # 解析 keywords JSON
            keywords = entity.get("keywords", "")
            if keywords and isinstance(keywords, str):
                try:
                    import json
                    keywords = json.loads(keywords)
                except:
                    keywords = []
            
            search_results.append(
                SearchResult(
                    text=entity.get("original_text") or entity.get("text", ""),
                    source=entity.get("source", "未知"),
                    score=doc.get("rerank_score", doc.get("score", 0)),
                    title=entity.get("title"),
                    year=entity.get("year"),
                    doi=entity.get("doi"),
                    keywords=keywords if keywords else None,
                    chunk_index=entity.get("chunk_index"),
                )
            )

        return SearchResponse(
            query=request.query,
            results=search_results,
            answer=result.answer,
            total=len(search_results),
        )

    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.get("/search", response_model=SearchResponse, tags=["检索"])
async def search_get(
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(default=10, ge=1, le=100),
    alpha: float = Query(default=0.5, ge=0, le=1),
):
    """GET 方式检索（简化参数）"""
    return await search(
        SearchRequest(query=q, top_k=top_k, alpha=alpha, enable_generation=False)
    )


# ============== 启动入口 ==============

def create_app():
    """创建应用实例（用于 uvicorn）"""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
