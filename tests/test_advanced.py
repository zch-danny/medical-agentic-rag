"""
测试 advanced 模块

包含:
- PubMed API 测试
- 信息提取器测试
- 缓存测试
- 异步工具测试
"""

import asyncio
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import pytest


# 使用 importlib 直接加载模块，避免 src/__init__.py 中的 torch 依赖
def load_module(name: str, path: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 加载模块
SRC_DIR = Path(__file__).parent.parent / "src" / "advanced"

extractor_module = load_module("extractor", SRC_DIR / "extractor.py")
cache_module = load_module("cache", SRC_DIR / "cache.py")
async_module = load_module("async_utils", SRC_DIR / "async_utils.py")
pubmed_module = load_module("pubmed", SRC_DIR / "pubmed.py")

# 从模块导入类
MedicalExtractor = extractor_module.MedicalExtractor
ExtractedInfo = extractor_module.ExtractedInfo
EntityType = extractor_module.EntityType
extract_medical_info = extractor_module.extract_medical_info

LRUCache = cache_module.LRUCache
TTLCache = cache_module.TTLCache
CacheManager = cache_module.CacheManager

AsyncExecutor = async_module.AsyncExecutor
RateLimiter = async_module.RateLimiter
TaskResult = async_module.TaskResult
BatchResult = async_module.BatchResult

PubMedArticle = pubmed_module.PubMedArticle
PubMedClient = pubmed_module.PubMedClient


# ============== 信息提取器测试 ==============

class TestMedicalExtractor:
    """测试医疗信息提取器"""
    
    def test_extract_diseases(self):
        """测试疾病提取"""
        text = "患者诊断为2型糖尿病合并高血压，既往有冠心病病史。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        assert len(info.diseases) >= 2
        disease_names = [d.normalized for d in info.diseases]
        assert "糖尿病" in disease_names
        assert "高血压" in disease_names
    
    def test_extract_symptoms(self):
        """测试症状提取"""
        text = "患者主诉头痛、乏力、发热3天，伴有咳嗽、咳痰。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        assert len(info.symptoms) >= 3
        symptom_names = [s.normalized for s in info.symptoms]
        assert "头痛" in symptom_names
        assert "乏力" in symptom_names
        assert "发热" in symptom_names
    
    def test_extract_medications(self):
        """测试药物提取"""
        text = "处方：二甲双胍500mg tid，阿司匹林100mg qd，奥美拉唑20mg bid。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        assert len(info.medications) >= 3
        med_names = [m.normalized for m in info.medications]
        assert "二甲双胍" in med_names
        assert "阿司匹林" in med_names
        assert "奥美拉唑" in med_names
    
    def test_extract_examinations(self):
        """测试检查提取"""
        text = "建议完善血常规、肝功能、CT检查和心电图。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        assert len(info.examinations) >= 3
        exam_names = [e.normalized for e in info.examinations]
        assert "血常规" in exam_names
        assert "肝功能" in exam_names
    
    def test_extract_treatments(self):
        """测试治疗方案提取"""
        text = "治疗方案：行冠脉搭桥手术，术后康复治疗，必要时化疗。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        assert len(info.treatments) >= 2
        treatment_names = [t.normalized for t in info.treatments]
        assert "搭桥手术" in treatment_names or "手术治疗" in treatment_names
    
    def test_to_json(self):
        """测试 JSON 输出"""
        text = "患者诊断为高血压，建议服用氨氯地平。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        json_str = info.to_json()
        assert isinstance(json_str, str)
        assert "diseases" in json_str
        assert "medications" in json_str
    
    def test_summary(self):
        """测试统计摘要"""
        text = "糖尿病患者出现头痛症状，处方二甲双胍。"
        extractor = MedicalExtractor()
        info = extractor.extract(text)
        
        summary = info.summary
        assert isinstance(summary, dict)
        assert "diseases" in summary
        assert "symptoms" in summary
        assert "medications" in summary
    
    def test_convenience_function(self):
        """测试便捷函数"""
        text = "高血压患者，服用缬沙坦。"
        info = extract_medical_info(text)
        
        assert isinstance(info, ExtractedInfo)
        assert len(info.diseases) >= 1


# ============== 缓存测试 ==============

class TestLRUCache:
    """测试 LRU 缓存"""
    
    def test_basic_operations(self):
        """测试基本操作"""
        cache = LRUCache(max_size=3)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") is None
    
    def test_eviction(self):
        """测试淘汰策略"""
        cache = LRUCache(max_size=2)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # 应该淘汰 "a"
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
    
    def test_update_order(self):
        """测试访问顺序更新"""
        cache = LRUCache(max_size=2)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # 访问 a，使其变为最近使用
        cache.set("c", 3)  # 应该淘汰 b
        
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
    
    def test_delete(self):
        """测试删除"""
        cache = LRUCache()
        cache.set("a", 1)
        
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("nonexistent") is False
    
    def test_clear(self):
        """测试清空"""
        cache = LRUCache()
        cache.set("a", 1)
        cache.set("b", 2)
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get("a") is None
    
    def test_stats(self):
        """测试统计"""
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.get("a")
        cache.get("b")  # miss
        
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestTTLCache:
    """测试 TTL 缓存"""
    
    def test_basic_operations(self):
        """测试基本操作"""
        cache = TTLCache(default_ttl=10.0)
        
        cache.set("a", 1)
        assert cache.get("a") == 1
    
    def test_expiration(self):
        """测试过期"""
        cache = TTLCache(default_ttl=0.1)
        
        cache.set("a", 1)
        assert cache.get("a") == 1
        
        time.sleep(0.2)
        assert cache.get("a") is None
    
    def test_custom_ttl(self):
        """测试自定义 TTL"""
        cache = TTLCache(default_ttl=10.0)
        
        cache.set("a", 1, ttl=0.1)
        assert cache.get("a") == 1
        
        time.sleep(0.2)
        assert cache.get("a") is None


class TestCacheManager:
    """测试缓存管理器"""
    
    def test_default_backend(self):
        """测试默认后端"""
        cache = CacheManager()
        
        cache.set("key", "value")
        assert cache.get("key") == "value"
    
    def test_namespace(self):
        """测试命名空间"""
        cache = CacheManager()
        
        cache.set("key", "value1", namespace="ns1")
        cache.set("key", "value2", namespace="ns2")
        
        assert cache.get("key", namespace="ns1") == "value1"
        assert cache.get("key", namespace="ns2") == "value2"
    
    def test_cached_decorator(self):
        """测试缓存装饰器"""
        cache = CacheManager()
        call_count = 0
        
        @cache.cached(ttl=10.0)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_func(5)
        result2 = expensive_func(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # 第二次应该从缓存获取


# ============== 异步工具测试 ==============

class TestRateLimiter:
    """测试限流器"""
    
    @pytest.mark.asyncio
    async def test_basic_acquire(self):
        """测试基本获取"""
        limiter = RateLimiter(rate=10, burst=5)
        
        # 应该能够立即获取
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        assert elapsed < 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """测试速率限制"""
        limiter = RateLimiter(rate=10, burst=2)
        
        # 快速获取 3 次，第 3 次应该等待
        await limiter.acquire()
        await limiter.acquire()
        
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        assert elapsed >= 0.05  # 应该等待一段时间


class TestAsyncExecutor:
    """测试异步执行器"""
    
    @pytest.mark.asyncio
    async def test_run_concurrent(self):
        """测试并发执行"""
        executor = AsyncExecutor(max_concurrency=5)
        
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2
        
        coros = [task(i) for i in range(5)]
        result = await executor.run_concurrent(coros)
        
        assert len(result.results) == 5
        assert result.success_rate == 1.0
        assert set(result.successful_results) == {0, 2, 4, 6, 8}
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """测试超时"""
        executor = AsyncExecutor(max_concurrency=5, default_timeout=0.1)
        
        async def slow_task():
            await asyncio.sleep(1.0)
            return "done"
        
        result = await executor.run_concurrent([slow_task()])
        
        assert len(result.results) == 1
        assert result.results[0].error is not None
    
    @pytest.mark.asyncio
    async def test_retry(self):
        """测试重试"""
        executor = AsyncExecutor()
        attempt_count = 0
        
        async def flaky_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = await executor.run_with_retry(
            flaky_task,
            max_retries=3,
            retry_delay=0.01,
        )
        
        assert result.success
        assert result.value == "success"
        assert result.retries == 2


class TestTaskResult:
    """测试任务结果"""
    
    def test_success(self):
        """测试成功结果"""
        result = TaskResult(value="data")
        
        assert result.success
        assert result.get_or_raise() == "data"
    
    def test_failure(self):
        """测试失败结果"""
        result = TaskResult(error=ValueError("test error"))
        
        assert not result.success
        with pytest.raises(ValueError):
            result.get_or_raise()


class TestBatchResult:
    """测试批处理结果"""
    
    def test_success_rate(self):
        """测试成功率"""
        results = [
            TaskResult(value=1),
            TaskResult(value=2),
            TaskResult(error=ValueError("fail")),
        ]
        batch = BatchResult(results=results)
        
        assert batch.success_rate == pytest.approx(2/3)
        assert len(batch.successful_results) == 2
        assert len(batch.failed_results) == 1


# ============== PubMed 测试 ==============

class TestPubMedArticle:
    """测试 PubMed 文章数据结构"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        article = PubMedArticle(
            pmid="12345678",
            title="Test Article",
            abstract="This is a test abstract.",
            authors=["Author A", "Author B"],
        )
        
        d = article.to_dict()
        assert d["pmid"] == "12345678"
        assert d["title"] == "Test Article"
        assert len(d["authors"]) == 2
    
    def test_to_document(self):
        """测试转换为文档格式"""
        article = PubMedArticle(
            pmid="12345678",
            title="Test Article",
            abstract="This is a test abstract.",
            authors=["Author A", "Author B", "Author C", "Author D"],
            journal="Test Journal",
        )
        
        doc = article.to_document()
        assert "content" in doc
        assert "metadata" in doc
        assert doc["metadata"]["source"] == "pubmed"
        assert "..." in doc["metadata"]["authors"]  # 超过 3 个作者应该截断


@pytest.mark.skip(reason="需要网络连接")
class TestPubMedClient:
    """测试 PubMed 客户端（需要网络）"""
    
    @pytest.mark.asyncio
    async def test_search(self):
        """测试搜索"""
        client = PubMedClient()
        try:
            pmids = await client.search("diabetes treatment", max_results=5)
            assert len(pmids) > 0
        finally:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_fetch_article(self):
        """测试获取文章"""
        client = PubMedClient()
        try:
            # 使用一个已知存在的 PMID
            article = await client.fetch_article("25693016")
            assert article is not None
            assert article.pmid == "25693016"
        finally:
            await client.close()


# ============== 运行测试 ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
