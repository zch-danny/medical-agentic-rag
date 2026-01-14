"""
高级功能模块

包含:
- PubMed API 集成 (pubmed.py)
- 结构化信息提取 (extractor.py)
- 缓存管理 (cache.py)
- 异步工具 (async_utils.py)
"""

from .pubmed import PubMedClient, PubMedArticle, search_pubmed
from .extractor import MedicalExtractor, ExtractedInfo
from .cache import CacheManager, LRUCache, TTLCache
from .async_utils import AsyncExecutor, batch_process, run_concurrent

__all__ = [
    # PubMed
    "PubMedClient",
    "PubMedArticle",
    "search_pubmed",
    # Extractor
    "MedicalExtractor",
    "ExtractedInfo",
    # Cache
    "CacheManager",
    "LRUCache",
    "TTLCache",
    # Async
    "AsyncExecutor",
    "batch_process",
    "run_concurrent",
]
