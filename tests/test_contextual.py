"""
Contextual Retrieval 模块测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.contextual import (
    ContextualConfig,
    ContextualEnricher,
    create_contextual_enricher,
    MEDICAL_CONTEXT_PROMPT_TEMPLATE,
    QWEN3_MEDICAL_CONTEXT_PROMPT,
    QWEN3_SYSTEM_PROMPT,
)


# ============== ContextualConfig 测试 ==============

class TestContextualConfig:
    """配置类测试"""
    
    def test_default_config(self):
        config = ContextualConfig()
        assert config.model == "qwen3-8b"
        assert config.max_context_tokens == 150
        assert config.temperature == 0.0
        assert config.enabled is True
    
    def test_custom_config(self):
        config = ContextualConfig(
            model="gpt-4",
            max_context_tokens=200,
            max_workers=8,
        )
        assert config.model == "gpt-4"
        assert config.max_context_tokens == 200
        assert config.max_workers == 8


# ============== ContextualEnricher 测试 ==============

class TestContextualEnricher:
    """上下文增强器测试"""
    
    @pytest.fixture
    def enricher_no_client(self):
        """无 LLM 客户端的增强器"""
        return ContextualEnricher(
            api_key=None,
            base_url=None,
            model="qwen3-8b",
        )
    
    @pytest.fixture
    def mock_enricher(self):
        """带 Mock 客户端的增强器"""
        with patch("src.contextual.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "这是一篇关于糖尿病治疗的研究论文，主要讨论了药物疗法。"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            enricher = ContextualEnricher(
                api_key="test-key",
                base_url="http://localhost:8000/v1",
                model="qwen3-8b",
            )
            enricher._client = mock_client
            return enricher
    
    def test_init_without_credentials(self, enricher_no_client):
        """测试无凭证初始化"""
        assert enricher_no_client._client is None
        assert enricher_no_client.is_available is False
    
    def test_qwen3_detection(self):
        """测试 Qwen3 模型检测"""
        enricher = ContextualEnricher(
            model="qwen3-8b",
            use_qwen3_optimization=True,
        )
        assert enricher._is_qwen3 is True
        assert enricher.system_prompt == QWEN3_SYSTEM_PROMPT
        assert enricher.prompt_template == QWEN3_MEDICAL_CONTEXT_PROMPT
    
    def test_qwen3_detection_disabled(self):
        """测试禁用 Qwen3 优化"""
        enricher = ContextualEnricher(
            model="qwen3-8b",
            use_qwen3_optimization=False,
        )
        assert enricher._is_qwen3 is False
        assert enricher.system_prompt is None
        assert enricher.prompt_template == MEDICAL_CONTEXT_PROMPT_TEMPLATE
    
    def test_non_qwen3_model(self):
        """测试非 Qwen3 模型"""
        enricher = ContextualEnricher(
            model="gpt-4",
            use_qwen3_optimization=True,
        )
        assert enricher._is_qwen3 is False
        assert enricher.system_prompt is None
    
    def test_truncate_document(self, enricher_no_client):
        """测试文档截断"""
        long_doc = "A" * 50000
        truncated = enricher_no_client._truncate_document(long_doc)
        assert len(truncated) < len(long_doc)
        assert "[...文档中间部分已省略...]" in truncated
    
    def test_truncate_short_document(self, enricher_no_client):
        """测试短文档不截断"""
        short_doc = "短文档内容"
        result = enricher_no_client._truncate_document(short_doc)
        assert result == short_doc
    
    def test_enrich_chunk_unavailable(self, enricher_no_client):
        """测试不可用时返回原文"""
        context, enriched = enricher_no_client.enrich_chunk("文档", "文本块")
        assert context == ""
        assert enriched == "文本块"
    
    def test_enrich_chunk_with_mock(self, mock_enricher):
        """测试带 Mock 的上下文生成"""
        context, enriched = mock_enricher.enrich_chunk(
            document="完整文档内容...",
            chunk="这是一个文本块",
        )
        assert context != ""
        assert context in enriched
        assert "这是一个文本块" in enriched
    
    def test_enrich_chunks_unavailable(self, enricher_no_client):
        """测试批量处理不可用"""
        chunks = ["chunk1", "chunk2", "chunk3"]
        results = enricher_no_client.enrich_chunks("doc", chunks, show_progress=False)
        
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["original"] == chunks[i]
            assert r["context"] == ""
            assert r["enriched"] == chunks[i]
    
    def test_enrich_chunks_with_mock(self, mock_enricher):
        """测试带 Mock 的批量处理"""
        chunks = ["chunk1", "chunk2"]
        results = mock_enricher.enrich_chunks("文档", chunks, show_progress=False)
        
        assert len(results) == 2
        for r in results:
            assert r["context"] != ""
            assert r["context"] in r["enriched"]


# ============== 工厂函数测试 ==============

class TestCreateContextualEnricher:
    """工厂函数测试"""
    
    def test_create_with_defaults(self):
        enricher = create_contextual_enricher()
        assert isinstance(enricher, ContextualEnricher)
    
    def test_create_with_qwen3_optimization(self):
        enricher = create_contextual_enricher(
            model="qwen3-8b",
            use_qwen3_optimization=True,
        )
        assert enricher._is_qwen3 is True
    
    def test_create_without_qwen3_optimization(self):
        enricher = create_contextual_enricher(
            model="qwen3-8b",
            use_qwen3_optimization=False,
        )
        assert enricher._is_qwen3 is False


# ============== Prompt 模板测试 ==============

class TestPromptTemplates:
    """提示词模板测试"""
    
    def test_medical_prompt_has_placeholders(self):
        """测试医学提示词包含占位符"""
        assert "{document}" in MEDICAL_CONTEXT_PROMPT_TEMPLATE
        assert "{chunk}" in MEDICAL_CONTEXT_PROMPT_TEMPLATE
    
    def test_qwen3_prompt_has_placeholders(self):
        """测试 Qwen3 提示词包含占位符"""
        assert "{document}" in QWEN3_MEDICAL_CONTEXT_PROMPT
        assert "{chunk}" in QWEN3_MEDICAL_CONTEXT_PROMPT
    
    def test_qwen3_prompt_has_no_think(self):
        """测试 Qwen3 提示词包含 /no_think"""
        assert "/no_think" in QWEN3_MEDICAL_CONTEXT_PROMPT
    
    def test_prompt_formatting(self):
        """测试提示词格式化"""
        doc = "测试文档"
        chunk = "测试块"
        
        formatted = QWEN3_MEDICAL_CONTEXT_PROMPT.format(document=doc, chunk=chunk)
        assert doc in formatted
        assert chunk in formatted


# ============== 清理思考标签测试 ==============

class TestThinkTagCleaning:
    """测试 <think> 标签清理"""
    
    def test_clean_think_tags(self):
        """测试清理思考标签"""
        import re
        
        content_with_think = """<think>
这是思考过程...
</think>
这是实际输出"""
        
        cleaned = re.sub(r"<think>.*?</think>", "", content_with_think, flags=re.DOTALL).strip()
        assert "<think>" not in cleaned
        assert "这是实际输出" in cleaned
    
    def test_no_think_tags(self):
        """测试无思考标签"""
        import re
        
        content = "这是正常输出"
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        assert cleaned == content


# ============== 集成测试 ==============

class TestContextualIntegration:
    """集成测试"""
    
    def test_explicit_model_param(self):
        """测试显式传入模型参数"""
        enricher = ContextualEnricher(model="custom-model")
        assert enricher.model == "custom-model"
    
    def test_default_model(self):
        """测试默认模型"""
        enricher = ContextualEnricher()
        # 默认使用配置中的 model，优先级高于环境变量
        assert enricher.model == "qwen3-8b"
