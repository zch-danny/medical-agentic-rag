"""
LlamaIndex Retriever 适配器

将现有的 MedicalRetriever 封装为 LlamaIndex 的 BaseRetriever
"""

from typing import Any, Dict, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from loguru import logger


class MedicalLlamaRetriever(BaseRetriever):
    """
    将 MedicalRetriever 封装为 LlamaIndex Retriever

    复用现有的:
    - MedicalEmbedder (Qwen3-Embedding)
    - VectorStore (Milvus Hybrid Search)
    - Qwen3Reranker
    """

    def __init__(
        self,
        retriever=None,
        top_k: int = 10,
        alpha: float = 0.7,
        use_rerank: bool = True,
        lazy_load: bool = True,
    ):
        """
        Args:
            retriever: 已有的 MedicalRetriever 实例，如果为 None 则自动创建
            top_k: 返回结果数量
            alpha: 混合检索权重 (0=纯BM25, 1=纯向量)
            use_rerank: 是否使用重排序
            lazy_load: 是否延迟加载模型
        """
        super().__init__()

        self._retriever = retriever
        self._top_k = top_k
        self._alpha = alpha
        self._use_rerank = use_rerank
        self._lazy_load = lazy_load

        if self._retriever is None and not lazy_load:
            self._init_retriever()

    def _init_retriever(self):
        """初始化内部 Retriever"""
        if self._retriever is not None:
            return

        from config import settings as cfg
        from src.retriever import MedicalRetriever

        logger.info("初始化 MedicalRetriever...")
        self._retriever = MedicalRetriever(
            config=cfg,
            lazy_load=self._lazy_load,
        )

    @property
    def retriever(self):
        """延迟获取 retriever"""
        if self._retriever is None:
            self._init_retriever()
        return self._retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        执行检索，返回 LlamaIndex 格式的结果

        Args:
            query_bundle: LlamaIndex 查询包装

        Returns:
            NodeWithScore 列表
        """
        query_str = query_bundle.query_str

        # 调用现有的检索方法
        results = self.retriever.search(
            query=query_str,
            top_k=self._top_k,
            use_rerank=self._use_rerank,
            alpha=self._alpha,
        )

        # 转换为 LlamaIndex 格式
        nodes_with_scores = []
        for result in results:
            # 提取实体信息
            entity = result.get("entity", result)

            # 创建 TextNode
            text = entity.get("original_text") or entity.get("text", "")
            node = TextNode(
                text=text,
                metadata={
                    "source": entity.get("source", ""),
                    "path": entity.get("path", ""),
                    "title": entity.get("title", ""),
                    "year": entity.get("year", ""),
                    "doi": entity.get("doi", ""),
                    "keywords": entity.get("keywords", ""),
                    "chunk_index": entity.get("chunk_index", 0),
                },
            )

            # 获取分数
            score = self._get_score(result)

            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        logger.debug(f"检索完成，返回 {len(nodes_with_scores)} 个结果")
        return nodes_with_scores

    def _get_score(self, result: Dict) -> float:
        """从检索结果中提取分数"""
        # 优先使用 rerank_score
        score = result.get("rerank_score")
        if score is not None:
            return float(score)

        # 其次使用 score
        score = result.get("score")
        if score is not None:
            return float(score)

        # 最后使用 distance（取反）
        distance = result.get("distance")
        if distance is not None:
            return -float(distance)

        return 0.0

    def retrieve_with_details(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索并返回详细信息（原始格式）

        方便需要完整元数据的场景
        """
        return self.retriever.search(
            query=query,
            top_k=top_k or self._top_k,
            alpha=alpha or self._alpha,
            use_rerank=use_rerank if use_rerank is not None else self._use_rerank,
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return self.retriever.get_stats()

    def close(self):
        """释放资源"""
        if self._retriever is not None:
            self._retriever.close()
