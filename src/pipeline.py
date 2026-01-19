"""
端到端 RAG 管道 - 整合检索与生成
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from loguru import logger

from config import settings as cfg
from src.embedder import MedicalEmbedder
from src.generator import AnswerGenerator
from src.reranker import Qwen3Reranker
from src.vector_store import VectorStore


@dataclass
class RAGConfig:
    """RAG 管道配置"""

    # 检索参数
    alpha: float = 0.5  # 混合检索权重 (0=纯BM25, 1=纯向量)
    candidate_top_k: int = 50  # 粗召回数量
    final_top_k: int = 10  # 重排序后返回数量
    enable_rerank: bool = True  # 是否启用重排序

    # 生成参数
    enable_generation: bool = True
    stream_output: bool = True


@dataclass
class RAGResult:
    """RAG 结果封装"""

    query: str
    documents: List[Dict] = field(default_factory=list)
    answer: Optional[str] = None
    answer_stream: Optional[Iterator[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalRAGPipeline:
    """
    医疗文献 RAG 管道

    整合:
    - 向量嵌入 (Qwen3-Embedding)
    - 混合检索 (Dense + Sparse via Milvus)
    - 重排序 (Qwen3-Reranker)
    - 答案生成 (DeepSeek/OpenAI)
    """

    def __init__(
        self,
        embedder: Optional[MedicalEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Qwen3Reranker] = None,
        generator: Optional[AnswerGenerator] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.config = config or RAGConfig()

        # 延迟初始化组件
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._generator = generator

        logger.info("MedicalRAGPipeline 初始化完成")

    @property
    def embedder(self) -> MedicalEmbedder:
        if self._embedder is None:
            self._embedder = MedicalEmbedder(
                model_name=cfg.EMBEDDING_MODEL,
                instruction=cfg.MEDICAL_INSTRUCT,
                max_length=cfg.MAX_SEQ_LENGTH,
            )
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                uri=cfg.MILVUS_URI,
                collection_name=cfg.COLLECTION_NAME,
                dim=cfg.EMBEDDING_DIM,
            )
        return self._vector_store

    @property
    def reranker(self) -> Qwen3Reranker:
        if self._reranker is None:
            self._reranker = Qwen3Reranker(
                model_name=cfg.RERANKER_MODEL,
                instruction=cfg.RERANKER_INSTRUCT,
                max_length=cfg.MAX_SEQ_LENGTH,
                batch_size=cfg.RERANKER_BATCH_SIZE,
            )
        return self._reranker

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator()
        return self._generator

    def retrieve(
        self,
        query: str,
        alpha: Optional[float] = None,
        candidate_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
    ) -> List[Dict]:
        """
        执行检索流程: Query Embedding → Hybrid Search → Rerank

        Args:
            query: 查询文本
            alpha: 混合检索权重，覆盖默认配置
            candidate_top_k: 粗召回数量，覆盖默认配置
            final_top_k: 最终返回数量，覆盖默认配置
            enable_rerank: 是否启用重排序，覆盖默认配置

        Returns:
            重排序后的文档列表
        """
        alpha = alpha if alpha is not None else self.config.alpha
        candidate_top_k = candidate_top_k or self.config.candidate_top_k
        final_top_k = final_top_k or self.config.final_top_k
        enable_rerank = enable_rerank if enable_rerank is not None else self.config.enable_rerank

        logger.info(
            f"检索开始: query={query[:50]}..., alpha={alpha}, "
            f"candidate_top_k={candidate_top_k}, final_top_k={final_top_k}, "
            f"enable_rerank={enable_rerank}"
        )

        # Step 1: Query Embedding
        query_vector = self.embedder.encode_query(query)
        logger.debug("Query embedding 完成")

        # Step 2: Hybrid Search (Dense + Sparse)
        candidates = self.vector_store.hybrid_search(
            query_embedding=query_vector,
            query_text=query,
            top_k=candidate_top_k,
            alpha=alpha,
        )
        logger.debug(f"混合检索返回 {len(candidates)} 个候选")

        if not candidates:
            logger.warning("混合检索未返回结果")
            return []

        # Step 3: Rerank（可选）
        if enable_rerank and len(candidates) > 1:
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=final_top_k,
            )
            logger.info(f"重排序完成，返回 {len(reranked)} 个文档")
            return reranked

        return candidates[:final_top_k]

    def query(
        self,
        query: str,
        alpha: Optional[float] = None,
        candidate_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        enable_generation: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
        stream: Optional[bool] = None,
    ) -> RAGResult:
        """
        完整 RAG 查询

        Args:
            query: 用户查询
            alpha: 混合检索权重
            candidate_top_k: 粗召回数量
            final_top_k: 最终返回数量
            enable_generation: 是否生成答案，覆盖默认配置
            enable_rerank: 是否启用重排序，覆盖默认配置
            stream: 是否流式输出，覆盖默认配置

        Returns:
            RAGResult 包含文档和答案
        """
        enable_gen = enable_generation if enable_generation is not None else self.config.enable_generation
        stream_out = stream if stream is not None else self.config.stream_output

        # 检索文档
        documents = self.retrieve(
            query=query,
            alpha=alpha,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
            enable_rerank=enable_rerank,
        )

        alpha_val = alpha if alpha is not None else self.config.alpha
        candidate_val = candidate_top_k if candidate_top_k is not None else self.config.candidate_top_k
        final_val = final_top_k if final_top_k is not None else self.config.final_top_k
        rerank_val = enable_rerank if enable_rerank is not None else self.config.enable_rerank

        result = RAGResult(
            query=query,
            documents=documents,
            metadata={
                "alpha": alpha_val,
                "candidate_top_k": candidate_val,
                "final_top_k": final_val,
                "enable_rerank": rerank_val,
                "num_retrieved": len(documents),
            },
        )

        # 生成答案
        if enable_gen and documents:
            try:
                if stream_out:
                    result.answer_stream = self.generator.generate_stream(query, documents)
                else:
                    result.answer = self.generator.generate_sync(query, documents)
            except Exception as e:
                logger.error(f"答案生成失败: {e}")
                result.answer = f"答案生成失败: {e}"
                result.answer_stream = None
        elif enable_gen and not documents:
            result.answer = "未找到相关文献，无法回答该问题。"

        return result

    def close(self):
        """释放资源"""
        if self._vector_store is not None:
            self._vector_store.close()
        logger.info("Pipeline 资源已释放")
