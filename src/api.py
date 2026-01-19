"""
FastAPI 服务 - 文件上传与检索 API
"""
import shutil
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings
from src.auth import require_api_key
from src.database import Database, DocumentStatus, get_database, init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化数据库
    init_database()
    logger.info("数据库已初始化")
    yield
    # 关闭时清理（如有需要）


app = FastAPI(
    title="医疗文献 RAG API",
    description="文件上传、索引、检索服务",
    version="1.0.0",
    lifespan=lifespan,
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


# ============== Paper2Figure 数据模型 ==============

class Paper2FigureRequest(BaseModel):
    """Paper2Figure 请求"""
    content: Optional[str] = Field(default=None, description="论文文本内容")
    file_id: Optional[str] = Field(default=None, description="已上传的文件 ID")
    figure_type: str = Field(default="auto", description="图表类型: auto, architecture, roadmap, flowchart, experiment")
    title: Optional[str] = Field(default=None, description="图表标题")
    output_formats: List[str] = Field(default=["html", "pptx"], description="输出格式: html, pptx, svg")


class Paper2FigureResponse(BaseModel):
    """Paper2Figure 响应"""
    success: bool
    title: str
    figure_type: str
    mermaid_code: str
    outputs: dict = Field(default_factory=dict, description="生成的文件路径")
    message: str


# ============== Paper2PPT 数据模型 ==============

class Paper2PPTRequest(BaseModel):
    """Paper2PPT 请求"""
    content: Optional[str] = Field(default=None, description="论文文本内容")
    file_id: Optional[str] = Field(default=None, description="已上传的文件 ID")
    title: Optional[str] = Field(default=None, description="PPT 标题")
    style: str = Field(default="academic", description="风格: academic, business, modern, colorful")


class Paper2PPTResponse(BaseModel):
    """Paper2PPT 响应"""
    success: bool
    title: str
    slide_count: int
    output_path: str
    message: str


# ============== PPTPolish 数据模型 ==============

class PPTPolishRequest(BaseModel):
    """美化请求"""
    file_id: Optional[str] = Field(default=None, description="已上传的 PPT 文件 ID")
    pptx_path: Optional[str] = Field(default=None, description="PPT 文件路径")
    mode: str = Field(default="full", description="美化模式: content, style, full")
    color_scheme: str = Field(default="academic_blue", description="配色方案")
    font_scheme: str = Field(default="professional", description="字体方案")
    add_page_numbers: bool = Field(default=True, description="是否添加页码")


class PPTPolishResponse(BaseModel):
    """美化响应"""
    success: bool
    original_path: str
    output_path: str
    changes: List[str]
    suggestions: List[str]
    message: str


# ============== 数据库依赖 ==============

def get_db() -> Database:
    """获取数据库实例"""
    return get_database()


# ============== 核心组件缓存 ==============

@lru_cache(maxsize=1)
def get_embedder():
    from src.embedder import MedicalEmbedder

    return MedicalEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        instruction=settings.MEDICAL_INSTRUCT,
        max_length=settings.MAX_SEQ_LENGTH,
        expected_dim=settings.EMBEDDING_DIM,
    )


@lru_cache(maxsize=1)
def get_vector_store():
    from src.vector_store import VectorStore

    return VectorStore(
        uri=settings.MILVUS_URI,
        collection_name=settings.COLLECTION_NAME,
        dim=settings.EMBEDDING_DIM,
    )


@lru_cache(maxsize=1)
def get_pipeline():
    from src.pipeline import MedicalRAGPipeline

    return MedicalRAGPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


# ============== API 端点 ==============

@app.get("/health", response_model=HealthResponse, tags=["系统"])
def health_check():
    """健康检查"""
    from src.health import HealthChecker

    checker = HealthChecker()
    result = checker.check_all(include_llm=False)

    return HealthResponse(
        healthy=result.healthy,
        services={s.name: {"healthy": s.healthy, "message": s.message} for s in result.services},
    )


@app.post("/upload", response_model=UploadResponse, tags=["文件管理"])
def upload_file(
    file: UploadFile = File(..., description="PDF 文件"),
    db: Database = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
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

        # 记录文件信息到数据库
        doc = db.create_document(
            file_id=file_id,
            filename=safe_filename,
            original_filename=file.filename,
            path=str(save_path),
            size_bytes=save_path.stat().st_size,
        )

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
def list_files(
    db: Database = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """列出所有已上传的文件"""
    docs = db.list_documents()
    return [
        DocumentInfo(
            file_id=doc.id,
            filename=doc.original_filename or doc.filename,
            size_bytes=doc.size_bytes or 0,
            status=doc.status.value if doc.status else "pending",
        )
        for doc in docs
    ]


@app.delete("/files/{file_id}", tags=["文件管理"])
def delete_file(
    file_id: str,
    db: Database = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """删除指定文件"""
    doc = db.get_document(file_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文件不存在")

    # 删除物理文件
    if doc.path:
        file_path = Path(doc.path)
        if file_path.exists():
            file_path.unlink()

    # 从数据库删除
    db.delete_document(file_id)
    logger.info(f"文件已删除: {file_id}")

    return {"success": True, "message": "文件已删除"}


@app.post("/index", response_model=IndexResponse, tags=["索引"])
def index_documents(
    request: IndexRequest = None,
    db: Database = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """
    索引文档到向量数据库

    如果不指定 file_ids，则索引所有状态为 pending 的文件。
    """
    from src.document_loader import MinerUDocumentLoader
    from src.embedding_cache import EmbeddingCache

    # 确定要索引的文件
    if request and request.file_ids:
        docs = [db.get_document(fid) for fid in request.file_ids]
        docs = [d for d in docs if d is not None]
    else:
        docs = db.list_documents(status=DocumentStatus.PENDING)

    if not docs:
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
        embedder = get_embedder()
        cache = EmbeddingCache(
            cache_dir=settings.EMBEDDING_CACHE_DIR,
            model_id=embedder.model_id,
        )
        vector_store = get_vector_store()
    except Exception as e:
        logger.error(f"初始化组件失败: {e}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

    indexed_count = 0
    failed_count = 0

    for doc in docs:
        # 更新状态为 INDEXING
        db.update_document_status(doc.id, DocumentStatus.INDEXING)
        
        file_path = Path(doc.path)

        if not file_path.exists():
            db.update_document_status(doc.id, DocumentStatus.FAILED, error_message="文件不存在")
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
                db.update_document_status(doc.id, DocumentStatus.FAILED, error_message="解析失败，无文本块")
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

            # 更新状态和 chunk 数量
            db.update_document_status(doc.id, DocumentStatus.INDEXED, chunk_count=len(chunks))
            indexed_count += 1
            logger.info(f"索引成功: {doc.id}, {len(chunks)} 个文本块")

        except Exception as e:
            logger.error(f"索引文件失败 {doc.id}: {e}")
            db.update_document_status(doc.id, DocumentStatus.FAILED, error_message=str(e))
            failed_count += 1

    return IndexResponse(
        success=failed_count == 0,
        indexed_count=indexed_count,
        failed_count=failed_count,
        message=f"索引完成: 成功 {indexed_count}, 失败 {failed_count}",
    )


@app.post("/search", response_model=SearchResponse, tags=["检索"])
def search(
    request: SearchRequest,
    api_key: str = Depends(require_api_key),
):
    """
    检索医疗文献

    支持混合检索（向量 + BM25）和可选的重排序。
    """

    try:
        pipeline = get_pipeline()

        result = pipeline.query(
            query=request.query,
            alpha=request.alpha,
            final_top_k=request.top_k,
            enable_generation=request.enable_generation,
            enable_rerank=request.enable_rerank,
            stream=False,
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
def search_get(
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(default=10, ge=1, le=100),
    alpha: float = Query(default=0.5, ge=0, le=1),
    enable_rerank: bool = Query(default=True),
    api_key: str = Depends(require_api_key),
):
    """​GET 方式检索（简化参数）"""
    return search(
        SearchRequest(
            query=q,
            top_k=top_k,
            alpha=alpha,
            enable_rerank=enable_rerank,
            enable_generation=False,
        ),
        api_key=api_key,
    )


# ============== Paper2Figure 端点 ==============

@app.post("/paper2figure", response_model=Paper2FigureResponse, tags=["Paper2Figure"])
def generate_figure(
    request: Paper2FigureRequest,
    db: Database = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """
    从论文内容生成图表

    支持生成：
    - 模型架构图 (architecture)
    - 技术路线图 (roadmap)
    - 流程图 (flowchart)
    - 实验数据图 (experiment)
    - 自动检测 (auto)

    输出格式：HTML、PPTX、SVG
    """
    from src.paper2figure import Paper2Figure, FigureType, FigureRenderer

    # 获取论文内容
    content = request.content

    if not content and request.file_id:
        # 从已上传文件获取内容
        doc = db.get_document(request.file_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文件不存在")

        # 加载已解析的文本
        parsed_dir = settings.PARSED_DIR / doc.filename.replace(".pdf", "")
        if parsed_dir.exists():
            # 尝试读取解析结果
            md_files = list(parsed_dir.glob("*.md"))
            if md_files:
                content = md_files[0].read_text(encoding="utf-8")

        if not content:
            # 如果没有解析结果，尝试直接解析 PDF
            from src.document_loader import MinerUDocumentLoader
            loader = MinerUDocumentLoader()
            chunks = loader.load(doc.path, chunk_size=10000, chunk_overlap=0)
            if chunks:
                content = "\n\n".join(c["text"] for c in chunks)

    if not content:
        raise HTTPException(status_code=400, detail="请提供论文内容或文件 ID")

    try:
        # 初始化 Paper2Figure
        p2f = Paper2Figure()

        # 解析图表类型
        figure_type = FigureType(request.figure_type) if request.figure_type in [e.value for e in FigureType] else FigureType.AUTO

        # 生成图表
        result = p2f.generate(content, figure_type, request.title)

        # 渲染输出
        renderer = FigureRenderer(output_dir=settings.DATA_DIR / "paper2figure")
        outputs = renderer.render_all(result, formats=request.output_formats)

        return Paper2FigureResponse(
            success=True,
            title=result.title,
            figure_type=result.figure_type.value,
            mermaid_code=result.mermaid_code,
            outputs=outputs,
            message="图表生成成功",
        )

    except Exception as e:
        logger.error(f"Paper2Figure 失败: {e}")
        raise HTTPException(status_code=500, detail=f"图表生成失败: {str(e)}")


@app.post("/paper2figure/from-text", response_model=Paper2FigureResponse, tags=["Paper2Figure"])
def generate_figure_from_text(
    content: str = Query(..., min_length=100, description="论文文本内容"),
    figure_type: str = Query(default="auto", description="图表类型"),
    title: Optional[str] = Query(default=None, description="图表标题"),
    api_key: str = Depends(require_api_key),
):
    """从文本内容生成图表（简化接口）"""
    return generate_figure(
        Paper2FigureRequest(content=content, figure_type=figure_type, title=title),
        db=get_db(),
        api_key=api_key,
    )


# ============== Paper2PPT 端点 ==============

@app.post("/paper2ppt", response_model=Paper2PPTResponse, tags=["Paper2PPT"])
def generate_ppt(
    request: Paper2PPTRequest,
    db: Database = Depends(get_db),
    api_key: str = Depends(require_api_key),
):
    """
    从论文内容生成完整 PPT 演示文稿

    支持风格：
    - academic: 学术风格（默认）
    - business: 商务风格
    - modern: 现代简约
    - colorful: 多彩活泼
    """
    from src.paper2figure import Paper2PPT, PPTStyle

    # 获取论文内容
    content = request.content

    if not content and request.file_id:
        doc = db.get_document(request.file_id)
        if not doc:
            raise HTTPException(status_code=404, detail="文件不存在")

        # 加载已解析的文本
        parsed_dir = settings.PARSED_DIR / doc.filename.replace(".pdf", "")
        if parsed_dir.exists():
            md_files = list(parsed_dir.glob("*.md"))
            if md_files:
                content = md_files[0].read_text(encoding="utf-8")

        if not content:
            from src.document_loader import MinerUDocumentLoader
            loader = MinerUDocumentLoader()
            chunks = loader.load(doc.path, chunk_size=10000, chunk_overlap=0)
            if chunks:
                content = "\n\n".join(c["text"] for c in chunks)

    if not content:
        raise HTTPException(status_code=400, detail="请提供论文内容或文件 ID")

    try:
        p2ppt = Paper2PPT()

        # 解析风格
        style = PPTStyle(request.style) if request.style in [e.value for e in PPTStyle] else PPTStyle.ACADEMIC

        # 生成 PPT
        output_dir = settings.DATA_DIR / "paper2ppt"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_title = "".join(c for c in (request.title or "presentation") if c.isalnum() or c in " _-")[:50]
        output_path = output_dir / f"{safe_title}.pptx"

        ppt_content = p2ppt.analyze(content, request.title)
        result_path = p2ppt.generate_pptx(ppt_content, output_path, style)

        return Paper2PPTResponse(
            success=True,
            title=ppt_content.title,
            slide_count=len(ppt_content.slides),
            output_path=result_path,
            message="PPT 生成成功",
        )

    except Exception as e:
        logger.error(f"Paper2PPT 失败: {e}")
        raise HTTPException(status_code=500, detail=f"PPT 生成失败: {str(e)}")


# ============== PPTPolish 端点 ==============

@app.post("/ppt-polish", response_model=PPTPolishResponse, tags=["PPTPolish"])
def polish_ppt(
    request: PPTPolishRequest,
    api_key: str = Depends(require_api_key),
):
    """
    美化 PPT

    功能：
    - 内容优化：精炼文字、统一风格
    - 样式美化：调整配色、字体
    - 添加页码

    配色方案：academic_blue, modern_green, elegant_purple, business_navy, warm_orange, minimal_gray
    字体方案：professional, elegant, modern
    """
    from src.paper2figure import PPTPolish, PolishMode

    # 确定 PPT 文件路径
    pptx_path = request.pptx_path

    if not pptx_path and request.file_id:
        # 从 paper2ppt 输出目录查找
        output_dir = settings.DATA_DIR / "paper2ppt"
        # 简化：查找最新的 pptx 文件
        pptx_files = list(output_dir.glob("*.pptx"))
        if pptx_files:
            pptx_path = str(max(pptx_files, key=lambda p: p.stat().st_mtime))

    if not pptx_path:
        raise HTTPException(status_code=400, detail="请提供 PPT 文件路径或文件 ID")

    from pathlib import Path
    if not Path(pptx_path).exists():
        raise HTTPException(status_code=404, detail=f"PPT 文件不存在: {pptx_path}")

    try:
        polisher = PPTPolish()

        # 解析美化模式
        mode = PolishMode(request.mode) if request.mode in [e.value for e in PolishMode] else PolishMode.FULL

        # 执行美化
        result = polisher.polish(
            pptx_path,
            mode=mode,
            color_scheme=request.color_scheme,
            font_scheme=request.font_scheme,
            add_numbers=request.add_page_numbers,
        )

        return PPTPolishResponse(
            success=True,
            original_path=result.original_path,
            output_path=result.output_path,
            changes=result.changes,
            suggestions=result.suggestions,
            message="PPT 美化完成",
        )

    except Exception as e:
        logger.error(f"PPTPolish 失败: {e}")
        raise HTTPException(status_code=500, detail=f"PPT 美化失败: {str(e)}")


@app.get("/ppt-polish/schemes", tags=["PPTPolish"])
def list_polish_schemes(
    api_key: str = Depends(require_api_key),
):
    """列出可用的配色和字体方案"""
    from src.paper2figure import PPTPolish

    return {
        "color_schemes": PPTPolish.list_color_schemes(),
        "font_schemes": PPTPolish.list_font_schemes(),
    }


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
