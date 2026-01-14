"""
分块器模块测试
"""
import sys
import importlib.util
from pathlib import Path

import pytest

# 直接加载 chunker.py，避免触发 src/__init__.py
ROOT_DIR = Path(__file__).parent.parent
chunker_path = ROOT_DIR / "src" / "chunker.py"
spec = importlib.util.spec_from_file_location("chunker", chunker_path)
chunker_module = importlib.util.module_from_spec(spec)
sys.modules["chunker"] = chunker_module
spec.loader.exec_module(chunker_module)

# 从加载的模块中导入
ChunkConfig = chunker_module.ChunkConfig
ChunkingStrategy = chunker_module.ChunkingStrategy
Chunk = chunker_module.Chunk
MarkdownChunker = chunker_module.MarkdownChunker
RecursiveChunker = chunker_module.RecursiveChunker
FixedSizeChunker = chunker_module.FixedSizeChunker
SemanticChunker = chunker_module.SemanticChunker
create_chunker = chunker_module.create_chunker


# ============== 测试数据 ==============

SAMPLE_MARKDOWN = """# 糖尿病概述

糖尿病是一种以高血糖为特征的代谢性疾病。

## 分类

### 1型糖尿病
1型糖尿病是由于胰岛β细胞被破坏，导致胰岛素绝对缺乏。

### 2型糖尿病
2型糖尿病是由于胰岛素抵抗和相对胰岛素分泌不足引起。

## 治疗

治疗方法包括：
- 饮食控制
- 运动疗法
- 药物治疗
"""

SAMPLE_PLAIN_TEXT = """糖尿病是一种常见的慢性代谢性疾病。它的主要特征是血糖水平持续偏高。
长期高血糖会导致多种并发症，包括心血管疾病、肾病、视网膜病变等。
治疗糖尿病需要综合管理，包括饮食控制、运动和必要的药物治疗。"""


# ============== ChunkConfig 测试 ==============

class TestChunkConfig:
    """配置类测试"""
    
    def test_default_config(self):
        config = ChunkConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.min_chunk_size == 100
        assert config.strategy == ChunkingStrategy.MARKDOWN
    
    def test_custom_config(self):
        config = ChunkConfig(chunk_size=1000, chunk_overlap=100)
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100


# ============== MarkdownChunker 测试 ==============

class TestMarkdownChunker:
    """Markdown 分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkConfig(chunk_size=200, min_chunk_size=20)
        return MarkdownChunker(config)
    
    def test_chunk_by_headers(self, chunker):
        """测试按标题分块"""
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        assert len(chunks) > 0
        # 每个 chunk 不应超过 max_chunk_size
        for chunk in chunks:
            assert len(chunk.text) <= chunker.config.max_chunk_size
    
    def test_preserves_content(self, chunker):
        """测试内容完整性"""
        chunks = chunker.chunk(SAMPLE_MARKDOWN)
        # 关键内容应该被保留
        combined = " ".join([c.text for c in chunks])
        assert "糖尿病" in combined
        assert "1型糖尿病" in combined or "型糖尿病" in combined
    
    def test_empty_input(self, chunker):
        """测试空输入"""
        chunks = chunker.chunk("")
        assert chunks == []
    
    def test_chunk_has_metadata(self, chunker):
        """测试 chunk 包含元数据"""
        chunks = chunker.chunk(SAMPLE_MARKDOWN, metadata={"source": "test"})
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.index >= 0
            assert "source" in chunk.metadata


# ============== RecursiveChunker 测试 ==============

class TestRecursiveChunker:
    """递归分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        return RecursiveChunker(config)
    
    def test_recursive_split(self, chunker):
        """测试递归分割"""
        text = "段落一。\n\n段落二。\n\n段落三。"
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
    
    def test_long_text(self, chunker):
        """测试长文本分块"""
        long_text = "这是一段很长的文本。" * 50
        chunks = chunker.chunk(long_text)
        assert len(chunks) > 1
        # 验证分块数量合理
        assert len(chunks) >= 2


# ============== FixedSizeChunker 测试 ==============

class TestFixedSizeChunker:
    """固定大小分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        return FixedSizeChunker(config)
    
    def test_fixed_size(self, chunker):
        """测试固定大小分块"""
        text = "A" * 200
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
    
    def test_short_text(self, chunker):
        """测试短文本"""
        text = "短文本测试内容"
        chunks = chunker.chunk(text)
        # 短文本可能因 min_chunk_size 限制而无块
        assert len(chunks) <= 1


# ============== 工厂函数测试 ==============

class TestCreateChunker:
    """工厂函数测试"""
    
    def test_create_markdown_chunker(self):
        chunker = create_chunker(ChunkingStrategy.MARKDOWN)
        assert isinstance(chunker, MarkdownChunker)
    
    def test_create_recursive_chunker(self):
        chunker = create_chunker(ChunkingStrategy.RECURSIVE)
        assert isinstance(chunker, RecursiveChunker)
    
    def test_create_fixed_chunker(self):
        chunker = create_chunker(ChunkingStrategy.FIXED)
        assert isinstance(chunker, FixedSizeChunker)
    
    def test_create_semantic_chunker(self):
        chunker = create_chunker(ChunkingStrategy.SEMANTIC)
        assert isinstance(chunker, SemanticChunker)
    
    def test_custom_config(self):
        config = ChunkConfig(chunk_size=1000)
        chunker = create_chunker(ChunkingStrategy.MARKDOWN, config)
        assert chunker.config.chunk_size == 1000


# ============== Chunk 数据类测试 ==============

class TestChunk:
    """Chunk 数据类测试"""
    
    def test_chunk_creation(self):
        chunk = Chunk(
            text="测试文本",
            index=0,
            start_char=0,
            end_char=4,
            metadata={"key": "value"}
        )
        assert chunk.text == "测试文本"
        assert chunk.index == 0
        assert chunk.metadata["key"] == "value"
    
    def test_chunk_to_dict(self):
        chunk = Chunk(
            text="测试文本",
            index=1,
            start_char=10,
            end_char=14,
            metadata={"source": "test"}
        )
        d = chunk.to_dict()
        assert d["text"] == "测试文本"
        assert d["chunk_index"] == 1
        assert d["chunk_start"] == 10
        assert d["chunk_end"] == 14
        assert d["source"] == "test"


# ============== 集成测试 ==============

class TestChunkerIntegration:
    """集成测试"""
    
    def test_medical_document_chunking(self):
        """测试医学文档分块"""
        medical_doc = """
# 高血压诊疗指南

## 定义
高血压是指以体循环动脉血压增高为主要特征的临床综合征。

## 分级
- 1级高血压：收缩压140-159mmHg 和/或 舒张压90-99mmHg
- 2级高血压：收缩压160-179mmHg 和/或 舒张压100-109mmHg
- 3级高血压：收缩压≥180mmHg 和/或 舒张压≥110mmHg

## 治疗原则
1. 生活方式干预
2. 药物治疗
3. 定期随访
"""
        config = ChunkConfig(chunk_size=200, min_chunk_size=30)
        chunker = create_chunker(ChunkingStrategy.MARKDOWN, config)
        chunks = chunker.chunk(medical_doc)
        
        assert len(chunks) >= 1
        # 验证关键信息被保留
        combined = " ".join([c.text for c in chunks])
        assert "高血压" in combined
