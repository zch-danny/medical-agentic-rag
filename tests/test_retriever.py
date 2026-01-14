"""
检索模块单元测试
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMedicalRetriever:
    """MedicalRetriever 测试"""

    @pytest.fixture
    def dummy_config(self):
        # 最小配置（测试用）
        from types import SimpleNamespace

        return SimpleNamespace(
            MILVUS_URI="http://localhost:19530",
            COLLECTION_NAME="test_collection",
            EMBEDDING_DIM=4096,
            TOP_K=50,
            RERANK_TOP_K=10,
            HYBRID_ALPHA=0.7,
            EMBEDDING_MODEL="dummy",
            MEDICAL_INSTRUCT="dummy",
            MAX_SEQ_LENGTH=8192,
            RERANKER_MODEL="dummy",
            RERANKER_INSTRUCT="dummy",
        )

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.encode_query.return_value = [0.1] * 4096
        return embedder

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.hybrid_search.return_value = [
            {
                "id": 1,
                "distance": 0.5,
                "entity": {
                    "text": "测试文本1",
                    "original_text": "这是测试文本1的完整内容",
                    "source": "test.pdf",
                },
            },
            {
                "id": 2,
                "distance": 0.7,
                "entity": {
                    "text": "测试文本2",
                    "original_text": "这是测试文本2的完整内容",
                    "source": "test.pdf",
                },
            },
        ]
        return store

    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()

        def mock_rerank(query, documents, top_k=10):
            # 模拟重排序：添加 rerank_score 并返回
            for i, doc in enumerate(documents[:top_k]):
                doc["rerank_score"] = 0.9 - i * 0.1
            return documents[:top_k]

        reranker.rerank.side_effect = mock_rerank
        return reranker

    def test_retrieve_returns_documents(self, dummy_config, mock_embedder, mock_vector_store, mock_reranker):
        """测试基本检索功能"""
        from src.retriever import MedicalRetriever

        retriever = MedicalRetriever(
            config=dummy_config,
            lazy_load=True,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            reranker=mock_reranker,
        )

        results = retriever.retrieve("测试查询", top_k=2)

        assert len(results) == 2
        mock_embedder.encode_query.assert_called_once_with("测试查询")
        mock_vector_store.hybrid_search.assert_called_once()
        mock_reranker.rerank.assert_called_once()

    def test_retrieve_empty_results(self, dummy_config, mock_embedder, mock_reranker):
        """测试空结果处理"""
        from src.retriever import MedicalRetriever

        empty_store = MagicMock()
        empty_store.hybrid_search.return_value = []

        retriever = MedicalRetriever(
            config=dummy_config,
            lazy_load=True,
            embedder=mock_embedder,
            vector_store=empty_store,
            reranker=mock_reranker,
        )

        results = retriever.retrieve("测试查询")

        assert results == []

    def test_retrieve_with_alpha(self, dummy_config, mock_embedder, mock_vector_store, mock_reranker):
        """测试 alpha 参数传递"""
        from src.retriever import MedicalRetriever

        retriever = MedicalRetriever(
            config=dummy_config,
            lazy_load=True,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            reranker=mock_reranker,
        )

        retriever.retrieve("测试查询", alpha=0.7)

        # 验证 alpha 参数被传递
        call_kwargs = mock_vector_store.hybrid_search.call_args[1]
        assert call_kwargs.get("alpha") == 0.7


class TestEmbeddingCache:
    """EmbeddingCache 测试"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        return tmp_path / "cache"

    def test_cache_set_and_get(self, temp_cache_dir):
        """测试缓存读写"""
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir=str(temp_cache_dir))

        embedding = [0.1, 0.2, 0.3]
        cache.set("test_text", embedding)

        result = cache.get("test_text")
        assert result == embedding

    def test_cache_miss(self, temp_cache_dir):
        """测试缓存未命中"""
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir=str(temp_cache_dir))

        result = cache.get("nonexistent")
        assert result is None

    def test_cache_persistence(self, temp_cache_dir):
        """测试缓存持久化"""
        from src.embedding_cache import EmbeddingCache

        # 创建并写入（set 会落盘）
        cache1 = EmbeddingCache(cache_dir=str(temp_cache_dir))
        cache1.set("persistent_key", [1.0, 2.0, 3.0])

        # 重新加载
        cache2 = EmbeddingCache(cache_dir=str(temp_cache_dir))
        result = cache2.get("persistent_key")

        assert result == [1.0, 2.0, 3.0]


class TestAnswerGenerator:
    """AnswerGenerator 测试"""

    @pytest.fixture
    def mock_openai_client(self):
        with patch("src.generator.OpenAI") as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance

            # Mock 非流式响应
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "这是生成的回答"
            client_instance.chat.completions.create.return_value = response

            yield client_instance

    def test_generate_sync(self, mock_openai_client):
        """测试同步生成"""
        from src.generator import AnswerGenerator

        with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
            generator = AnswerGenerator()

            documents = [
                {
                    "entity": {
                        "text": "测试内容",
                        "source": "test.pdf",
                    },
                    "score": 0.9,
                }
            ]

            result = generator.generate_sync("测试问题", documents)

            assert result == "这是生成的回答"
            mock_openai_client.chat.completions.create.assert_called_once()

    def test_generate_empty_documents(self, mock_openai_client):
        """测试空文档处理"""
        from src.generator import AnswerGenerator

        with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
            generator = AnswerGenerator()

            result = generator.generate_sync("测试问题", [])

            assert "未找到相关文献" in result


class TestRAGPipeline:
    """MedicalRAGPipeline 集成测试"""

    @pytest.fixture
    def mock_components(self):
        embedder = MagicMock()
        embedder.encode_query.return_value = [0.1] * 4096

        vector_store = MagicMock()
        vector_store.hybrid_search.return_value = [
            {"id": 1, "entity": {"text": "内容", "source": "test.pdf"}, "distance": 0.5}
        ]

        reranker = MagicMock()
        reranker.rerank.return_value = [
            {"id": 1, "entity": {"text": "内容", "source": "test.pdf"}, "rerank_score": 0.9}
        ]

        generator = MagicMock()
        generator.generate_sync.return_value = "测试回答"

        return embedder, vector_store, reranker, generator

    def test_pipeline_query(self, mock_components):
        """测试完整查询流程"""
        from src.pipeline import MedicalRAGPipeline, RAGConfig

        embedder, vector_store, reranker, generator = mock_components

        config = RAGConfig(enable_generation=True, stream_output=False)
        pipeline = MedicalRAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            generator=generator,
            config=config,
        )

        result = pipeline.query("测试问题")

        assert len(result.documents) == 1
        assert result.answer == "测试回答"

    def test_pipeline_retrieve_only(self, mock_components):
        """测试仅检索模式"""
        from src.pipeline import MedicalRAGPipeline, RAGConfig

        embedder, vector_store, reranker, generator = mock_components

        config = RAGConfig(enable_generation=False)
        pipeline = MedicalRAGPipeline(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            generator=generator,
            config=config,
        )

        result = pipeline.query("测试问题")

        assert len(result.documents) == 1
        assert result.answer is None
        generator.generate_sync.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
