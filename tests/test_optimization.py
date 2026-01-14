"""
DSPy 优化模块测试

测试 Signatures, Modules, Metrics 和 Optimizer
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
import importlib.util
project_root = str(Path(__file__).parent.parent)

# 直接加载模块文件，避免触发 src/__init__.py
def load_module_direct(name: str, file_path: str):
    """直接从文件加载模块"""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# 加载 metrics 模块
metrics_path = Path(project_root) / "src" / "optimization" / "metrics.py"
metrics_module = load_module_direct("src.optimization.metrics", str(metrics_path))

# 导出需要的类和函数
semantic_f1 = metrics_module.semantic_f1
citation_accuracy = metrics_module.citation_accuracy
factual_consistency = metrics_module.factual_consistency
EvaluationResult = metrics_module.EvaluationResult
MedicalQAMetrics = metrics_module.MedicalQAMetrics
_simple_token_overlap = metrics_module._simple_token_overlap
_tokenize = metrics_module._tokenize


class TestMetrics:
    """评估指标测试"""
    
    def test_tokenize_chinese(self):
        """测试中文分词"""
        text = "糖尿病的症状"
        tokens = _tokenize(text)
        assert "糖" in tokens
        assert "尿" in tokens
        assert "病" in tokens
    
    def test_tokenize_english(self):
        """测试英文分词"""
        text = "diabetes symptoms"
        tokens = _tokenize(text)
        assert "d" in tokens or "diabetes" in "".join(tokens)
    
    def test_simple_token_overlap(self):
        """测试简单词重叠"""
        text1 = "糖尿病的症状包括多饮多尿"
        text2 = "糖尿病主要症状是多饮多尿多食"
        
        score = _simple_token_overlap(text1, text2)
        assert 0 < score <= 1
    
    def test_simple_token_overlap_identical(self):
        """测试相同文本"""
        text = "糖尿病"
        score = _simple_token_overlap(text, text)
        assert score == 1.0
    
    def test_simple_token_overlap_empty(self):
        """测试空文本"""
        score = _simple_token_overlap("", "测试")
        assert score == 0.0
    
    def test_semantic_f1_basic(self):
        """测试语义 F1 基本功能"""
        prediction = "糖尿病的主要症状包括多饮、多尿、多食"
        ground_truth = "糖尿病的症状有多饮多尿多食体重下降"
        
        result = semantic_f1(prediction, ground_truth)
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.score <= 1
        assert isinstance(result.passed, bool)
    
    def test_semantic_f1_empty_input(self):
        """测试空输入"""
        result = semantic_f1("", "测试")
        assert result.score == 0.0
        assert result.passed is False
    
    def test_citation_accuracy_valid(self):
        """测试有效引用"""
        answer = "根据文献[1]，糖尿病的症状包括多饮多尿[2]。"
        documents = ["文献1内容", "文献2内容", "文献3内容"]
        
        result = citation_accuracy(answer, documents)
        
        assert result.score == 1.0
        assert result.passed is True
        assert result.details["valid_citations"] == 2
    
    def test_citation_accuracy_invalid(self):
        """测试无效引用"""
        answer = "根据文献[5]，糖尿病的症状..."
        documents = ["文献1", "文献2"]
        
        result = citation_accuracy(answer, documents)
        
        assert result.score == 0.0
        assert "5" in result.details["invalid_citations"]
    
    def test_citation_accuracy_no_citations(self):
        """测试无引用"""
        answer = "糖尿病的症状包括多饮多尿"
        documents = ["文献1"]
        
        result = citation_accuracy(answer, documents)
        
        assert result.details.get("no_citations") is True
    
    def test_factual_consistency_basic(self):
        """测试事实一致性基本功能"""
        answer = "糖尿病的治疗方法包括药物治疗和生活方式干预"
        context = "糖尿病是一种代谢疾病，治疗方法包括药物治疗、饮食控制等"
        
        result = factual_consistency(answer, context)
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.score <= 1
    
    def test_factual_consistency_empty(self):
        """测试空输入"""
        result = factual_consistency("", "测试")
        assert result.score == 0.0


class TestMedicalQAMetrics:
    """综合评估指标测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        metrics = MedicalQAMetrics()
        
        assert metrics.weights["semantic_f1"] == 0.4
        assert metrics.weights["citation_accuracy"] == 0.3
        assert metrics.weights["factual_consistency"] == 0.3
    
    def test_init_custom_weights(self):
        """测试自定义权重"""
        weights = {
            "semantic_f1": 0.5,
            "citation_accuracy": 0.25,
            "factual_consistency": 0.25,
        }
        metrics = MedicalQAMetrics(weights=weights)
        
        assert metrics.weights["semantic_f1"] == 0.5
    
    def test_evaluate_with_ground_truth(self):
        """测试带标准答案的评估"""
        metrics = MedicalQAMetrics()
        
        result = metrics.evaluate(
            prediction="糖尿病症状包括多饮多尿",
            ground_truth="糖尿病的主要症状是多饮多尿多食",
        )
        
        assert "semantic_f1" in result
        assert "overall" in result
        assert 0 <= result["overall"]["score"] <= 1
    
    def test_evaluate_with_context(self):
        """测试带文献的评估"""
        metrics = MedicalQAMetrics()
        
        result = metrics.evaluate(
            prediction="糖尿病需要治疗",
            context="糖尿病是需要长期治疗的疾病",
        )
        
        assert "factual_consistency" in result
    
    def test_evaluate_with_documents(self):
        """测试带文档列表的评估"""
        metrics = MedicalQAMetrics()
        
        result = metrics.evaluate(
            prediction="根据[1]，糖尿病...",
            documents=["糖尿病相关文献"],
        )
        
        assert "citation_accuracy" in result


class TestTrainingData:
    """训练数据测试"""
    
    def test_training_data_exists(self):
        """测试训练数据文件存在"""
        data_path = Path(project_root) / "data" / "training" / "qa_pairs.json"
        assert data_path.exists(), f"训练数据文件不存在: {data_path}"
    
    def test_training_data_format(self):
        """测试训练数据格式"""
        data_path = Path(project_root) / "data" / "training" / "qa_pairs.json"
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # 检查第一条数据的格式
        first_item = data[0]
        assert "question" in first_item
        assert "context" in first_item
        assert "answer" in first_item
    
    def test_training_data_content(self):
        """测试训练数据内容"""
        data_path = Path(project_root) / "data" / "training" / "qa_pairs.json"
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 检查所有数据
        for item in data:
            assert len(item["question"]) > 0, "问题不能为空"
            assert len(item["context"]) > 0, "上下文不能为空"
            assert len(item["answer"]) > 0, "答案不能为空"


class TestSignatures:
    """Signatures 测试（需要 DSPy）"""
    
    @pytest.mark.skipif(True, reason="需要 DSPy 环境")
    def test_medical_qa_signature(self):
        """测试 MedicalQA 签名"""
        from src.optimization.signatures import MedicalQA
        
        # 检查字段
        assert hasattr(MedicalQA, "context")
        assert hasattr(MedicalQA, "question")
        assert hasattr(MedicalQA, "answer")
    
    @pytest.mark.skipif(True, reason="需要 DSPy 环境")
    def test_query_rewrite_signature(self):
        """测试 QueryRewrite 签名"""
        from src.optimization.signatures import QueryRewrite
        
        assert hasattr(QueryRewrite, "original_query")
        assert hasattr(QueryRewrite, "conversation_history")
        assert hasattr(QueryRewrite, "rewritten_query")


class TestModules:
    """Modules 测试（需要 DSPy）"""
    
    @pytest.mark.skipif(True, reason="需要 DSPy 环境")
    def test_optimized_rag_init(self):
        """测试 OptimizedRAG 初始化"""
        from src.optimization.modules import OptimizedRAG
        
        module = OptimizedRAG(with_citation=False)
        assert module.with_citation is False
    
    @pytest.mark.skipif(True, reason="需要 DSPy 环境")
    def test_optimized_rewriter_init(self):
        """测试 OptimizedRewriter 初始化"""
        from src.optimization.modules import OptimizedRewriter
        
        module = OptimizedRewriter(with_reasoning=True)
        assert module.with_reasoning is True


class TestOptimizer:
    """Optimizer 测试（需要 DSPy）"""
    
    @pytest.mark.skipif(True, reason="需要 DSPy 环境")
    def test_optimization_config_default(self):
        """测试默认优化配置"""
        from src.optimization.optimizer import OptimizationConfig
        
        config = OptimizationConfig()
        
        assert config.optimizer_type == "BootstrapFewShot"
        assert config.max_bootstrapped_demos == 4
        assert config.max_labeled_demos == 8
    
    @pytest.mark.skipif(True, reason="需要 DSPy 环境")
    def test_training_example(self):
        """测试训练样本"""
        from src.optimization.optimizer import TrainingExample
        
        example = TrainingExample(
            question="测试问题",
            context="测试上下文",
            answer="测试答案",
        )
        
        dspy_example = example.to_dspy_example()
        assert dspy_example.question == "测试问题"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
