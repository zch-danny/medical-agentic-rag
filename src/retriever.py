"""
检索器 - 整合嵌入、向量检索和重排序
"""
from typing import Dict, List, Optional

from loguru import logger

from .embedder import MedicalEmbedder
from .reranker import Qwen3Reranker
from .vector_store import VectorStore


class MedicalRetriever:
    """医疗文献检索器 (嵌入 + 向量检索 + 重排序)"""

    def __init__(
        self,
        config,
        lazy_load: bool = False,
        embedder: Optional[MedicalEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Qwen3Reranker] = None,
    ):
        """
        Args:
            config: 配置模块（如 config.settings）
            lazy_load: 是否延迟加载模型 (首次检索时加载)
            embedder/vector_store/reranker: 可注入，便于测试或自定义
        """
        self.config = config
        self._embedder = embedder
        self._reranker = reranker

        # 向量库默认立即初始化（可通过注入替换）
        self.vector_store = vector_store or VectorStore(
            config.MILVUS_URI,
            config.COLLECTION_NAME,
            config.EMBEDDING_DIM,
        )

        if not lazy_load and (self._embedder is None or self._reranker is None):
            self._load_models()

    def _load_models(self):
        """加载嵌入和重排序模型"""
        if self._embedder is None:
            self._embedder = MedicalEmbedder(
                self.config.EMBEDDING_MODEL,
                self.config.MEDICAL_INSTRUCT,
                self.config.MAX_SEQ_LENGTH
            )
        if self._reranker is None:
            self._reranker = Qwen3Reranker(
                self.config.RERANKER_MODEL,
                instruction=self.config.RERANKER_INSTRUCT
            )

    @property
    def embedder(self) -> MedicalEmbedder:
        if self._embedder is None:
            self._load_models()
        return self._embedder

    @property
    def reranker(self) -> Qwen3Reranker:
        if self._reranker is None:
            self._load_models()
        return self._reranker

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        alpha: Optional[float] = None,
        candidate_top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        检索医疗文献

        Args:
            query: 查询文本
            top_k: 返回结果数量 (默认使用配置)
            use_rerank: 是否使用重排序
            alpha: 混合检索权重，覆盖配置
            candidate_top_k: 粗召回数量，覆盖配置

        Returns:
            检索结果列表，包含 text, source, score 等字段
        """
        top_k_val = top_k if top_k is not None else self.config.RERANK_TOP_K
        alpha_val = alpha if alpha is not None else getattr(self.config, "HYBRID_ALPHA", 0.7)

        # 1. 查询嵌入
        logger.debug(f"查询: {query[:50]}...")
        query_emb = self.embedder.encode_query(query)

        # 2. 向量检索
        default_candidate = self.config.TOP_K if use_rerank else top_k_val
        retrieve_k = candidate_top_k if candidate_top_k is not None else default_candidate
        candidates = self.vector_store.hybrid_search(
            query_embedding=query_emb,
            query_text=query,
            top_k=retrieve_k,
            alpha=alpha_val,
        )
        logger.debug(f"向量检索返回 {len(candidates)} 条候选")

        if not candidates:
            return []

        # 3. 重排序 (可选)
        if use_rerank and len(candidates) > 1:
            results = self.reranker.rerank(query, candidates, top_k_val)
            logger.debug(f"重排序后返回 {len(results)} 条结果")
        else:
            results = candidates[:top_k_val]

        return results

    # 兼容别名
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict]:
        return self.search(query=query, top_k=top_k, **kwargs)

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            "collection": self.config.COLLECTION_NAME,
            "vector_count": self.vector_store.count(),
            "embedding_model": self.config.EMBEDDING_MODEL,
            "reranker_model": self.config.RERANKER_MODEL,
        }

    def close(self):
        """释放资源"""
        close_fn = getattr(self.vector_store, "close", None)
        if callable(close_fn):
            close_fn()
