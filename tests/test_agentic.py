"""
Agentic 组件测试

测试 QueryRouter, QueryRewriter, ResultEvaluator
"""

import pytest
from src.agentic import (
    QueryRouter, RouteDecision, QueryType,
    QueryRewriter, RewriteResult,
    ResultEvaluator, EvaluationResult,
)
from src.agentic.result_evaluator import EvaluationDecision


class TestQueryRouter:
    """测试 QueryRouter"""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_hybrid_default(self, router):
        """默认使用混合检索"""
        result = router.route("糖尿病的治疗方法有哪些？")
        assert result.query_type == QueryType.HYBRID

    def test_bm25_for_exact_terms(self, router):
        """精确术语使用 BM25"""
        result = router.route("阿司匹林的剂量是多少mg？")
        assert result.query_type == QueryType.BM25
        assert result.suggested_alpha < 0.5

    def test_vector_for_semantic(self, router):
        """语义问题使用向量检索"""
        result = router.route("糖尿病为什么会导致视力下降？机制是什么？")
        assert result.query_type == QueryType.VECTOR
        assert result.suggested_alpha > 0.7

    def test_direct_for_simple_definition(self, router):
        """简单定义问题直接回答"""
        result = router.route("什么是BMI？")
        assert result.query_type == QueryType.DIRECT

    def test_web_for_recent_info(self, router):
        """最新信息使用联网搜索"""
        result = router.route("2024年糖尿病指南有什么更新？")
        assert result.query_type == QueryType.WEB

    def test_medical_abbreviations(self, router):
        """医学缩写识别"""
        result = router.route("T2DM患者的HbA1c控制目标")
        assert result.query_type == QueryType.BM25

    def test_confidence_score(self, router):
        """置信度在有效范围"""
        result = router.route("任意查询")
        assert 0 <= result.confidence <= 1

    def test_suggested_alpha(self, router):
        """建议的 alpha 在有效范围"""
        result = router.route("任意查询")
        assert 0 <= result.suggested_alpha <= 1


class TestQueryRewriter:
    """测试 QueryRewriter"""

    @pytest.fixture
    def rewriter(self):
        return QueryRewriter()

    def test_no_rewrite_needed(self, rewriter):
        """不需要改写的查询"""
        result = rewriter.rewrite("糖尿病的症状")
        # 可能会有同义词扩展，但主查询不变
        assert "糖尿病" in result.rewritten_query

    def test_term_standardization(self, rewriter):
        """术语标准化"""
        result = rewriter.rewrite("hypertension怎么治疗")
        assert "高血压" in result.rewritten_query or "hypertension" in result.rewritten_query

    def test_follow_up_completion(self, rewriter):
        """追问补全"""
        result = rewriter.rewrite(
            "怎么预防？",
            previous_query="糖尿病的症状有哪些"
        )
        assert "糖尿病" in result.rewritten_query
        assert "预防" in result.rewritten_query

    def test_synonym_expansion(self, rewriter):
        """同义词扩展"""
        result = rewriter.rewrite("高血压的治疗")
        # 应该扩展出英文或其他同义词
        assert len(result.expanded_terms) > 0 or "高血压" in result.rewritten_query

    def test_ambiguity_detection(self, rewriter):
        """歧义检测"""
        result = rewriter.rewrite("糖尿病的治疗")
        # 应该提示可能的类型
        if "建议明确" in result.reason:
            assert "1型" in result.reason or "2型" in result.reason

    def test_rewrite_result_fields(self, rewriter):
        """检查返回字段完整性"""
        result = rewriter.rewrite("测试查询")
        assert isinstance(result, RewriteResult)
        assert result.original_query == "测试查询"
        assert isinstance(result.expanded_terms, list)
        assert isinstance(result.is_modified, bool)


class TestResultEvaluator:
    """测试 ResultEvaluator"""

    @pytest.fixture
    def evaluator(self):
        return ResultEvaluator()

    @pytest.fixture
    def sample_docs_good(self):
        """高质量检索结果"""
        return [
            {
                "entity": {
                    "original_text": "糖尿病的治疗包括饮食控制、运动和药物治疗。" * 5,
                    "source": "内科学教材.pdf",
                },
                "rerank_score": 0.95,
            },
            {
                "entity": {
                    "original_text": "2型糖尿病首选口服降糖药，如二甲双胍。" * 5,
                    "source": "临床指南.pdf",
                },
                "rerank_score": 0.88,
            },
            {
                "entity": {
                    "original_text": "糖尿病患者需要定期监测血糖和糖化血红蛋白。" * 5,
                    "source": "护理手册.pdf",
                },
                "rerank_score": 0.82,
            },
        ]

    @pytest.fixture
    def sample_docs_poor(self):
        """低质量检索结果"""
        return [
            {
                "entity": {
                    "original_text": "高血压的治疗方法...",
                    "source": "其他文档.pdf",
                },
                "rerank_score": 0.3,
            },
        ]

    def test_sufficient_results(self, evaluator, sample_docs_good):
        """充分结果评估"""
        result = evaluator.evaluate("糖尿病的治疗方法", sample_docs_good)
        assert result.decision == EvaluationDecision.SUFFICIENT
        assert result.relevance_score > 0.6

    def test_insufficient_results(self, evaluator, sample_docs_poor):
        """不充分结果评估"""
        result = evaluator.evaluate("糖尿病的治疗方法", sample_docs_poor)
        assert result.decision in [EvaluationDecision.PARTIAL, EvaluationDecision.INSUFFICIENT]

    def test_empty_results(self, evaluator):
        """空结果评估"""
        result = evaluator.evaluate("任意查询", [])
        assert result.decision == EvaluationDecision.INSUFFICIENT
        assert result.relevance_score == 0.0

    def test_best_docs_indices(self, evaluator, sample_docs_good):
        """最佳文档索引"""
        result = evaluator.evaluate("糖尿病", sample_docs_good)
        assert len(result.best_docs_indices) > 0
        assert all(0 <= i < len(sample_docs_good) for i in result.best_docs_indices)

    def test_filter_relevant_docs(self, evaluator, sample_docs_good):
        """过滤相关文档"""
        filtered = evaluator.filter_relevant_docs(sample_docs_good, min_score=0.8)
        assert len(filtered) >= 2

    def test_evaluation_result_fields(self, evaluator, sample_docs_good):
        """检查返回字段完整性"""
        result = evaluator.evaluate("测试", sample_docs_good)
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.relevance_score <= 1
        assert 0 <= result.sufficiency_score <= 1
        assert isinstance(result.suggestions, list)


class TestIntegration:
    """集成测试：组件协同工作"""

    def test_router_rewriter_flow(self):
        """路由 -> 改写流程"""
        router = QueryRouter()
        rewriter = QueryRewriter()

        query = "DM的治疗"

        # 路由决策
        route_result = router.route(query)

        # 改写查询
        rewrite_result = rewriter.rewrite(query)

        # 改写后应该包含标准术语
        assert "糖尿病" in rewrite_result.rewritten_query or "DM" in rewrite_result.rewritten_query

    def test_full_agentic_flow(self):
        """完整 Agentic 流程"""
        router = QueryRouter()
        rewriter = QueryRewriter()
        evaluator = ResultEvaluator()

        query = "高血压怎么预防？"

        # 1. 路由
        route = router.route(query)
        assert route.query_type in QueryType

        # 2. 改写
        rewrite = rewriter.rewrite(query)
        final_query = rewrite.rewritten_query

        # 3. 模拟检索结果
        mock_results = [
            {
                "entity": {"original_text": "高血压的预防包括..." * 10, "source": "doc1.pdf"},
                "rerank_score": 0.85,
            },
            {
                "entity": {"original_text": "控制盐摄入是预防高血压的重要措施..." * 10, "source": "doc2.pdf"},
                "rerank_score": 0.78,
            },
        ]

        # 4. 评估
        eval_result = evaluator.evaluate(final_query, mock_results)

        # 应该有合理的评估结果
        assert eval_result.decision in EvaluationDecision
        assert len(eval_result.best_docs_indices) > 0
