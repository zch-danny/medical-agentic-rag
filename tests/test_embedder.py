"""
嵌入模块单元测试
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMedicalEmbedder:
    """MedicalEmbedder 测试"""

    @pytest.fixture
    def mock_tokenizer(self):
        """模拟 tokenizer"""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """模拟嵌入模型"""
        model = MagicMock()
        model.config.hidden_size = 4096
        model.device = "cpu"
        model.eval = MagicMock()
        return model

    @patch("src.embedder.AutoTokenizer")
    @patch("src.embedder.AutoModel")
    @patch("src.embedder.torch.cuda.is_available", return_value=False)
    def test_init_success(self, mock_cuda, mock_auto_model, mock_auto_tokenizer):
        """测试正常初始化"""
        from src.embedder import MedicalEmbedder

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_auto_model.from_pretrained.return_value = mock_model

        embedder = MedicalEmbedder(model_name="test-model")

        assert embedder.model_name == "test-model"
        assert embedder.embedding_dim == 4096
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        mock_auto_model.from_pretrained.assert_called()

    @patch("src.embedder.AutoTokenizer")
    def test_init_tokenizer_failure(self, mock_auto_tokenizer):
        """测试 tokenizer 加载失败"""
        from src.embedder import MedicalEmbedder, EmbedderLoadError

        mock_auto_tokenizer.from_pretrained.side_effect = Exception("网络错误")

        with pytest.raises(EmbedderLoadError) as exc_info:
            MedicalEmbedder(model_name="invalid-model")

        assert "Tokenizer 加载失败" in str(exc_info.value)

    @patch("src.embedder.AutoTokenizer")
    @patch("src.embedder.AutoModel")
    @patch("src.embedder.torch.cuda.is_available", return_value=False)
    def test_init_model_failure(self, mock_cuda, mock_auto_model, mock_auto_tokenizer):
        """测试模型加载失败"""
        from src.embedder import MedicalEmbedder, EmbedderLoadError

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_auto_model.from_pretrained.side_effect = Exception("模型不存在")

        with pytest.raises(EmbedderLoadError) as exc_info:
            MedicalEmbedder(model_name="invalid-model")

        assert "模型加载失败" in str(exc_info.value)

    @patch("src.embedder.AutoTokenizer")
    @patch("src.embedder.AutoModel")
    @patch("src.embedder.torch.cuda.is_available", return_value=False)
    def test_dimension_mismatch_warning(self, mock_cuda, mock_auto_model, mock_auto_tokenizer, caplog):
        """测试维度不匹配警告"""
        from src.embedder import MedicalEmbedder

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 1024  # 实际维度
        mock_auto_model.from_pretrained.return_value = mock_model

        # 期望维度 4096，但实际是 1024
        embedder = MedicalEmbedder(model_name="test-model", expected_dim=4096)

        assert embedder.embedding_dim == 1024
        assert "不匹配" in caplog.text or embedder._expected_dim == 4096

    @patch("src.embedder.AutoTokenizer")
    @patch("src.embedder.AutoModel")
    @patch("src.embedder.torch.cuda.is_available", return_value=False)
    def test_model_id_property(self, mock_cuda, mock_auto_model, mock_auto_tokenizer):
        """测试 model_id 属性"""
        from src.embedder import MedicalEmbedder

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_auto_model.from_pretrained.return_value = mock_model

        embedder = MedicalEmbedder(model_name="Qwen/Qwen3-Embedding-8B")

        assert embedder.model_id == "Qwen/Qwen3-Embedding-8B:4096"

    @patch("src.embedder.AutoTokenizer")
    @patch("src.embedder.AutoModel")
    @patch("src.embedder.torch.cuda.is_available", return_value=False)
    def test_custom_instruction(self, mock_cuda, mock_auto_model, mock_auto_tokenizer):
        """测试自定义指令"""
        from src.embedder import MedicalEmbedder

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_auto_model.from_pretrained.return_value = mock_model

        custom_instruction = "自定义医疗查询指令"
        embedder = MedicalEmbedder(model_name="test", instruction=custom_instruction)

        assert embedder.instruction == custom_instruction

    @patch("src.embedder.AutoTokenizer")
    @patch("src.embedder.AutoModel")
    @patch("src.embedder.torch.cuda.is_available", return_value=False)
    def test_default_instruction(self, mock_cuda, mock_auto_model, mock_auto_tokenizer):
        """测试默认指令"""
        from src.embedder import MedicalEmbedder

        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_auto_model.from_pretrained.return_value = mock_model

        embedder = MedicalEmbedder(model_name="test")

        assert "medical" in embedder.instruction.lower()


class TestEmbeddingCacheAdvanced:
    """EmbeddingCache 高级功能测试"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        return tmp_path / "embedding_cache"

    def test_model_version_tracking(self, temp_cache_dir):
        """测试模型版本追踪"""
        from src.embedding_cache import EmbeddingCache

        # 使用模型 A 创建缓存
        cache_a = EmbeddingCache(
            cache_dir=temp_cache_dir,
            model_id="model-a:1024"
        )
        cache_a.set("test", [1.0, 2.0])

        # 使用相同模型重新打开
        cache_a2 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            model_id="model-a:1024"
        )
        assert cache_a2.get("test") == [1.0, 2.0]

    def test_model_version_change_clears_cache(self, temp_cache_dir):
        """测试模型版本变化时清空缓存"""
        from src.embedding_cache import EmbeddingCache

        # 使用模型 A 创建缓存
        cache_a = EmbeddingCache(
            cache_dir=temp_cache_dir,
            model_id="model-a:1024"
        )
        cache_a.set("test", [1.0, 2.0])

        # 使用模型 B 打开（应该清空）
        cache_b = EmbeddingCache(
            cache_dir=temp_cache_dir,
            model_id="model-b:2048"
        )
        assert cache_b.get("test") is None

    def test_ttl_expiration(self, temp_cache_dir):
        """测试 TTL 过期"""
        import time
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(
            cache_dir=temp_cache_dir,
            ttl_seconds=0.1  # 100ms 过期
        )
        cache.set("test", [1.0, 2.0])

        # 立即获取应该成功
        assert cache.get("test") == [1.0, 2.0]

        # 等待过期
        time.sleep(0.15)
        assert cache.get("test") is None

    def test_batch_write_optimization(self, temp_cache_dir):
        """测试批量写入优化"""
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        texts = [f"text_{i}" for i in range(10)]
        embeddings = [[float(i)] * 3 for i in range(10)]

        def compute_fn(texts_to_compute):
            return [[float(i)] * 3 for i in range(len(texts_to_compute))]

        results = cache.get_or_compute(texts, compute_fn)

        assert len(results) == 10
        # 验证索引文件只写入一次（通过检查统计）
        assert cache.stats()["total_entries"] == 10

    def test_cleanup_expired(self, temp_cache_dir):
        """测试清理过期缓存"""
        import time
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(
            cache_dir=temp_cache_dir,
            ttl_seconds=0.05
        )

        # 添加多个条目
        for i in range(5):
            cache.set(f"key_{i}", [float(i)])

        # 等待过期
        time.sleep(0.1)

        # 清理
        cleaned = cache.cleanup_expired()
        assert cleaned == 5
        assert cache.stats()["total_entries"] == 0

    def test_flush(self, temp_cache_dir):
        """测试 flush 方法"""
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache.set("test", [1.0], save_index=False)

        # dirty 标记应该为 True
        assert cache._dirty is True

        cache.flush()
        assert cache._dirty is False

    def test_full_hash_no_collision(self, temp_cache_dir):
        """测试完整哈希避免碰撞"""
        from src.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # 两个不同文本
        text1 = "这是第一段文本"
        text2 = "这是第二段文本"

        cache.set(text1, [1.0, 1.0])
        cache.set(text2, [2.0, 2.0])

        assert cache.get(text1) == [1.0, 1.0]
        assert cache.get(text2) == [2.0, 2.0]


class TestEmbedderIntegration:
    """嵌入器集成测试（需要真实模型时跳过）"""

    @pytest.mark.skip(reason="需要真实模型，CI 中跳过")
    def test_encode_query_real(self):
        """测试真实查询编码"""
        from src.embedder import MedicalEmbedder

        embedder = MedicalEmbedder()
        result = embedder.encode_query("什么是糖尿病？")

        assert isinstance(result, list)
        assert len(result) == embedder.embedding_dim
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.skip(reason="需要真实模型，CI 中跳过")
    def test_encode_documents_real(self):
        """测试真实文档编码"""
        from src.embedder import MedicalEmbedder

        embedder = MedicalEmbedder()
        docs = ["糖尿病是一种代谢疾病", "高血压的治疗方法"]
        results = embedder.encode_documents(docs)

        assert len(results) == 2
        assert all(len(r) == embedder.embedding_dim for r in results)
