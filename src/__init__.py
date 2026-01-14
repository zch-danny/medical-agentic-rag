from .document_loader import DocumentLoader, MinerUDocumentLoader
from .embedder import MedicalEmbedder
from .embedding_cache import EmbeddingCache
from .vector_store import VectorStore
from .reranker import Qwen3Reranker, Reranker
from .retriever import MedicalRetriever
from .generator import AnswerGenerator, GenerationConfig
from .pipeline import MedicalRAGPipeline, RAGConfig, RAGResult
from .health import HealthChecker, HealthStatus, SystemHealth

# LlamaIndex Adapters (延迟导入，避免强制依赖 llama-index)
try:
    from .adapters import (
        MedicalLlamaRetriever,
        MedicalRetrieverTool,
        MedicalGeneratorTool,
        create_medical_tools,
    )
    _HAS_LLAMA_INDEX = True
except ImportError:
    _HAS_LLAMA_INDEX = False

__all__ = [
    "DocumentLoader",
    "MinerUDocumentLoader",
    "MedicalEmbedder",
    "EmbeddingCache",
    "VectorStore",
    "Qwen3Reranker",
    "Reranker",
    "MedicalRetriever",
    "AnswerGenerator",
    "GenerationConfig",
    "MedicalRAGPipeline",
    "RAGConfig",
    "RAGResult",
    "HealthChecker",
    "HealthStatus",
    "SystemHealth",
]

# 仅在安装了 llama-index 时导出适配器
if _HAS_LLAMA_INDEX:
    __all__.extend([
        "MedicalLlamaRetriever",
        "MedicalRetrieverTool",
        "MedicalGeneratorTool",
        "create_medical_tools",
    ])
