"""
向量存储 - Milvus 2.5+ Hybrid Search (Dense + BM25)
"""
from typing import Dict, List

import jieba
from loguru import logger
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
)
from tenacity import retry, stop_after_attempt, wait_exponential


def tokenize_text(text: str) -> str:
    """
    智能分词：中文用 jieba，英文按空格/标点
    
    支持中英混合文本
    """
    import re
    
    result = []
    # 按中文和非中文分段
    segments = re.split(r'([\u4e00-\u9fff]+)', text)
    
    for seg in segments:
        if not seg.strip():
            continue
        # 如果是中文，用 jieba
        if re.search(r'[\u4e00-\u9fff]', seg):
            result.extend(jieba.cut(seg))
        else:
            # 英文/数字，按空格和标点分割
            tokens = re.split(r'[\s\-_.,;:!?()\[\]{}"\'/]+', seg)
            result.extend([t.lower() for t in tokens if t.strip()])
    
    return " ".join(result)


# 兼容旧名称
def tokenize_chinese(text: str) -> str:
    return tokenize_text(text)


class VectorStore:
    """
    Milvus 向量存储

    支持:
    - Dense 向量检索 (COSINE)
    - BM25 稀疏检索 (内置分词)
    - Hybrid Search (加权融合)
    """

    def __init__(self, uri: str = None, collection_name: str = None, dim: int = None):
        """
        Args:
            uri: Milvus 连接 URI
            collection_name: 集合名称
            dim: 向量维度
        """
        # 延迟导入，避免循环依赖
        from config import settings as cfg

        self.uri = uri or cfg.MILVUS_URI
        self.collection_name = collection_name or cfg.COLLECTION_NAME
        self.dim = dim or cfg.EMBEDDING_DIM

        logger.info(f"连接 Milvus: {self.uri}")
        self.client = MilvusClient(uri=self.uri)

        self._ensure_collection()

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        if self.collection_name in self.client.list_collections():
            logger.info(f"集合已存在: {self.collection_name}")
            return

        logger.info(f"创建集合: {self.collection_name}")

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="original_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
            # 元数据字段
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="year", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1024),  # JSON 字符串
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="indexed_at", dtype=DataType.VARCHAR, max_length=32),
            # 向量字段
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        schema = CollectionSchema(fields=fields, description="医疗文献向量库")

        # 添加 BM25 函数（自动从 text 生成 sparse_vector）
        bm25_function = Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # 创建索引
        index_params = self.client.prepare_index_params()

        # Dense 向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # Sparse 向量索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        logger.info(f"集合创建完成: {self.collection_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def insert(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: List[Dict],
        use_jieba: bool = True,
    ) -> int:
        """
        插入数据

        Args:
            embeddings: 嵌入向量列表
            texts: 文本列表
            metadata: 元数据列表
            use_jieba: 是否使用 jieba 分词

        Returns:
            插入数量
        """
        data = []
        for emb, text, meta in zip(embeddings, texts, metadata):
            # 对文本进行 jieba 分词（用于 BM25）
            processed_text = tokenize_chinese(text) if use_jieba else text
            
            # 处理 keywords（列表转 JSON 字符串）
            keywords = meta.get("keywords", [])
            if isinstance(keywords, list):
                import json
                keywords = json.dumps(keywords, ensure_ascii=False)

            data.append({
                "dense_vector": emb,
                "text": processed_text,
                "original_text": text,
                "source": meta.get("source", ""),
                "path": meta.get("path", ""),
                "title": meta.get("title", "")[:500],
                "year": meta.get("year", "")[:10],
                "doi": meta.get("doi", "")[:250],
                "keywords": keywords[:1000] if keywords else "",
                "chunk_index": meta.get("chunk_index", 0),
                "indexed_at": meta.get("indexed_at", "")[:32],
            })

        try:
            self.client.insert(collection_name=self.collection_name, data=data)
            logger.debug(f"成功插入 {len(data)} 条数据")
            return len(data)
        except Exception as e:
            logger.error(f"Insert 失败: {e}")
            raise

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 20,
        alpha: float = 0.7,
        use_jieba: bool = True,
    ) -> List[Dict]:
        """
        混合检索（Dense + BM25）

        Args:
            query_embedding: 查询向量
            query_text: 查询文本
            top_k: 返回数量
            alpha: 语义向量权重 (0-1)
            use_jieba: 是否对查询进行 jieba 分词

        Returns:
            检索结果列表
        """
        # 对查询文本进行 jieba 分词（与入库时保持一致）
        processed_query = tokenize_chinese(query_text) if use_jieba else query_text

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=top_k * 2,
        )

        sparse_req = AnnSearchRequest(
            data=[processed_query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k * 2,
        )

        ranker = WeightedRanker(alpha, 1 - alpha)

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=top_k,
            output_fields=["text", "source", "path", "original_text", "title", "year", "doi", "keywords", "chunk_index"],
        )

        return results[0] if results else []

    def count(self) -> int:
        """获取集合中的文档数量"""
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)

    def delete_collection(self):
        """删除集合"""
        if self.collection_name in self.client.list_collections():
            self.client.drop_collection(self.collection_name)
            logger.info(f"已删除集合: {self.collection_name}")

    # 兼容别名
    def drop_collection(self):
        """delete_collection 的别名"""
        return self.delete_collection()

    def close(self):
        """关闭客户端连接"""
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
