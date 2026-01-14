"""
DSPy 优化模块

提供基于 DSPy 的提示词优化和答案质量提升功能：
- Signatures: 定义任务的输入输出规范
- Modules: 可优化的 DSPy 模块
- Metrics: 评估指标
- Optimizer: 优化器封装
"""

from .signatures import (
    MedicalQA,
    QueryRewrite,
    RelevanceEval,
    AnswerEval,
)
from .modules import (
    OptimizedRAG,
    OptimizedRewriter,
    OptimizedEvaluator,
    ChainOfThoughtRAG,
)
from .metrics import (
    semantic_f1,
    citation_accuracy,
    factual_consistency,
    MedicalQAMetrics,
)
from .optimizer import (
    DSPyOptimizer,
    OptimizationConfig,
    load_compiled_module,
    save_compiled_module,
)

__all__ = [
    # Signatures
    "MedicalQA",
    "QueryRewrite",
    "RelevanceEval",
    "AnswerEval",
    # Modules
    "OptimizedRAG",
    "OptimizedRewriter",
    "OptimizedEvaluator",
    "ChainOfThoughtRAG",
    # Metrics
    "semantic_f1",
    "citation_accuracy",
    "factual_consistency",
    "MedicalQAMetrics",
    # Optimizer
    "DSPyOptimizer",
    "OptimizationConfig",
    "load_compiled_module",
    "save_compiled_module",
]
