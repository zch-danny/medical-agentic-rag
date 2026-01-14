"""
医疗文献嵌入系统配置
支持环境变量和 .env 文件
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent.parent


def _get_env(key: str, default: str = None, cast_type: type = str):
    """获取环境变量，支持类型转换"""
    value = os.getenv(key, default)
    if value is None:
        return None
    if cast_type == bool:
        return value.lower() in ("true", "1", "yes")
    return cast_type(value)


# ===================
# 模型配置
# ===================
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
RERANKER_MODEL = _get_env("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")
EMBEDDING_DIM = _get_env("EMBEDDING_DIM", "4096", int)
MAX_SEQ_LENGTH = _get_env("MAX_SEQ_LENGTH", "8192", int)

# ===================
# 分块配置
# ===================
# CHUNK_SIZE: 字符数（不是 token）
# 医学文献建议 512 字符，约等于 200-300 中文 token
# 太大会稀释语义，太小会丢失上下文
CHUNK_SIZE = _get_env("CHUNK_SIZE", "512", int)
CHUNK_OVERLAP = _get_env("CHUNK_OVERLAP", "64", int)

# ===================
# 检索配置
# ===================
TOP_K = _get_env("TOP_K", "50", int)  # Hybrid 检索候选数
RERANK_TOP_K = _get_env("RERANK_TOP_K", "10", int)  # 重排后返回数
HYBRID_ALPHA = _get_env("HYBRID_ALPHA", "0.7", float)  # 语义向量权重

# ===================
# 医疗领域指令
# ===================
MEDICAL_INSTRUCT = _get_env(
    "MEDICAL_INSTRUCT",
    "Given a medical or clinical question, retrieve relevant passages from medical literature, "
    "clinical guidelines, or research papers that provide evidence-based answers about diagnosis, "
    "treatment, symptoms, or prognosis"
)
RERANKER_INSTRUCT = _get_env(
    "RERANKER_INSTRUCT",
    "Given a medical question, determine if the document contains clinically relevant, "
    "evidence-based information that directly answers or supports the question"
)

# ===================
# 数据目录
# ===================
DATA_DIR = Path(_get_env("DATA_DIR", str(BASE_DIR / "data")))
DOCUMENTS_DIR = DATA_DIR / "documents"
PARSED_DIR = DATA_DIR / "parsed"
CACHE_DIR = DATA_DIR / "cache"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"

# ===================
# 数据库配置
# ===================
DATABASE_URL = _get_env("DATABASE_URL", f"sqlite:///{DATA_DIR}/app.db")

# ===================
# API 认证配置
# ===================
API_KEYS = _get_env("API_KEYS", "")  # 逗号分隔多个 Key
API_AUTH_ENABLED = _get_env("API_AUTH_ENABLED", "false", bool)
API_AUTH_USE_DB = _get_env("API_AUTH_USE_DB", "false", bool)  # 是否从数据库读取 Key

# ===================
# Milvus 配置
# ===================
MILVUS_URI = _get_env("MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION_NAME = _get_env("COLLECTION_NAME", "medical_literature")

# ===================
# LLM API 配置 (用于答案生成)
# ===================
LLM_API_KEY = _get_env("LLM_API_KEY", "")
LLM_BASE_URL = _get_env("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = _get_env("LLM_MODEL", "deepseek-chat")

# ===================
# MinerU 配置
# ===================
MINERU_BACKEND = _get_env("MINERU_BACKEND", "hybrid-auto-engine")

# ===================
# 性能配置
# ===================
BATCH_SIZE = _get_env("BATCH_SIZE", "4", int)
RERANKER_BATCH_SIZE = _get_env("RERANKER_BATCH_SIZE", "8", int)
RERANKER_TIMEOUT = _get_env("RERANKER_TIMEOUT", "2.0", float)
NUM_WORKERS = _get_env("NUM_WORKERS", "2", int)

# ===================
# 日志配置
# ===================
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"

# ===================
# PubMed API 配置
# ===================
PUBMED_API_KEY = _get_env("PUBMED_API_KEY", "")  # NCBI API Key（可选）
PUBMED_EMAIL = _get_env("PUBMED_EMAIL", "")  # 联系邮箱（推荐）
PUBMED_MAX_RESULTS = _get_env("PUBMED_MAX_RESULTS", "20", int)
PUBMED_TIMEOUT = _get_env("PUBMED_TIMEOUT", "30.0", float)

# ===================
# 缓存配置
# ===================
CACHE_DIR = DATA_DIR / "cache"
CACHE_LRU_SIZE = _get_env("CACHE_LRU_SIZE", "1000", int)
CACHE_TTL_DEFAULT = _get_env("CACHE_TTL_DEFAULT", "300.0", float)  # 5分钟
CACHE_PERSISTENT_DIR = DATA_DIR / "cache" / "persistent"

# ===================
# 异步配置
# ===================
ASYNC_MAX_CONCURRENCY = _get_env("ASYNC_MAX_CONCURRENCY", "10", int)
ASYNC_DEFAULT_TIMEOUT = _get_env("ASYNC_DEFAULT_TIMEOUT", "30.0", float)
ASYNC_BATCH_SIZE = _get_env("ASYNC_BATCH_SIZE", "10", int)

# ===================
# 兼容字段别名
# ===================
# 历史/脚本中可能使用 MILVUS_COLLECTION
MILVUS_COLLECTION = COLLECTION_NAME

# ===================
# 创建必要目录
# ===================
for _dir in [DOCUMENTS_DIR, PARSED_DIR, CACHE_DIR, EMBEDDING_CACHE_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ===================
# settings 对象（兼容 pipeline / scripts 中的 from config.settings import settings）
# ===================
from types import SimpleNamespace

settings = SimpleNamespace(
    # paths
    BASE_DIR=BASE_DIR,
    DATA_DIR=DATA_DIR,
    DOCUMENTS_DIR=DOCUMENTS_DIR,
    PARSED_DIR=PARSED_DIR,
    CACHE_DIR=CACHE_DIR,
    EMBEDDING_CACHE_DIR=EMBEDDING_CACHE_DIR,
    LOG_DIR=LOG_DIR,
    # database
    DATABASE_URL=DATABASE_URL,
    # api auth
    API_KEYS=API_KEYS,
    API_AUTH_ENABLED=API_AUTH_ENABLED,
    API_AUTH_USE_DB=API_AUTH_USE_DB,
    # models
    EMBEDDING_MODEL=EMBEDDING_MODEL,
    RERANKER_MODEL=RERANKER_MODEL,
    EMBEDDING_DIM=EMBEDDING_DIM,
    MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
    MEDICAL_INSTRUCT=MEDICAL_INSTRUCT,
    RERANKER_INSTRUCT=RERANKER_INSTRUCT,
    # chunking
    CHUNK_SIZE=CHUNK_SIZE,
    CHUNK_OVERLAP=CHUNK_OVERLAP,
    # retrieval
    TOP_K=TOP_K,
    RERANK_TOP_K=RERANK_TOP_K,
    HYBRID_ALPHA=HYBRID_ALPHA,
    # milvus
    MILVUS_URI=MILVUS_URI,
    COLLECTION_NAME=COLLECTION_NAME,
    MILVUS_COLLECTION=MILVUS_COLLECTION,
    # llm
    LLM_API_KEY=LLM_API_KEY,
    LLM_BASE_URL=LLM_BASE_URL,
    LLM_MODEL=LLM_MODEL,
    # mineru
    MINERU_BACKEND=MINERU_BACKEND,
    # perf
    BATCH_SIZE=BATCH_SIZE,
    RERANKER_BATCH_SIZE=RERANKER_BATCH_SIZE,
    RERANKER_TIMEOUT=RERANKER_TIMEOUT,
    NUM_WORKERS=NUM_WORKERS,
    # logging
    LOG_LEVEL=LOG_LEVEL,
    # pubmed
    PUBMED_API_KEY=PUBMED_API_KEY,
    PUBMED_EMAIL=PUBMED_EMAIL,
    PUBMED_MAX_RESULTS=PUBMED_MAX_RESULTS,
    PUBMED_TIMEOUT=PUBMED_TIMEOUT,
    # cache
    CACHE_LRU_SIZE=CACHE_LRU_SIZE,
    CACHE_TTL_DEFAULT=CACHE_TTL_DEFAULT,
    CACHE_PERSISTENT_DIR=CACHE_PERSISTENT_DIR,
    # async
    ASYNC_MAX_CONCURRENCY=ASYNC_MAX_CONCURRENCY,
    ASYNC_DEFAULT_TIMEOUT=ASYNC_DEFAULT_TIMEOUT,
    ASYNC_BATCH_SIZE=ASYNC_BATCH_SIZE,
)
