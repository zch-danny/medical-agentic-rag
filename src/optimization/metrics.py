"""
评估指标模块

提供用于评估 RAG 系统性能的指标：
- 语义 F1：答案与标准答案的语义相似度
- 引用准确率：引用是否来自检索文档
- 事实一致性：答案是否与文献一致
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger


@dataclass
class EvaluationResult:
    """评估结果"""
    score: float                    # 主要分数 (0-1)
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    passed: bool = False            # 是否通过阈值


def semantic_f1(
    prediction: str,
    ground_truth: str,
    embedder=None,
    threshold: float = 0.7,
) -> EvaluationResult:
    """
    计算语义 F1 分数
    
    使用 embedding 计算预测答案与标准答案的语义相似度。
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        embedder: Embedding 模型（可选）
        threshold: 通过阈值
        
    Returns:
        EvaluationResult
    """
    if not prediction or not ground_truth:
        return EvaluationResult(
            score=0.0,
            details={"error": "Empty input"},
            passed=False,
        )
    
    # 如果没有提供 embedder，使用简单的词重叠计算
    if embedder is None:
        score = _simple_token_overlap(prediction, ground_truth)
    else:
        try:
            # 使用 embedding 计算余弦相似度
            pred_emb = embedder.embed(prediction)
            truth_emb = embedder.embed(ground_truth)
            score = _cosine_similarity(pred_emb, truth_emb)
        except Exception as e:
            logger.warning(f"Embedding 计算失败，使用词重叠: {e}")
            score = _simple_token_overlap(prediction, ground_truth)
    
    return EvaluationResult(
        score=score,
        details={
            "prediction_length": len(prediction),
            "ground_truth_length": len(ground_truth),
        },
        passed=score >= threshold,
    )


def _simple_token_overlap(text1: str, text2: str) -> float:
    """简单的词重叠计算"""
    # 中文分词（简单实现）
    tokens1 = set(_tokenize(text1))
    tokens2 = set(_tokenize(text2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    precision = len(intersection) / len(tokens1) if tokens1 else 0
    recall = len(intersection) / len(tokens2) if tokens2 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _tokenize(text: str) -> List[str]:
    """简单分词"""
    # 移除标点
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词（对中文按字符，对英文按空格）
    tokens = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符
            tokens.append(char)
        elif char.isalnum():
            tokens.append(char.lower())
    return tokens


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    import math
    
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def citation_accuracy(
    answer: str,
    documents: List[str],
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    计算引用准确率
    
    检查答案中的引用是否来自提供的文档。
    
    Args:
        answer: 生成的答案（可能包含 [1], [2] 等引用）
        documents: 检索到的文档列表
        threshold: 通过阈值
        
    Returns:
        EvaluationResult
    """
    # 提取引用编号
    citations = re.findall(r'\[(\d+)\]', answer)
    
    if not citations:
        # 没有引用，检查答案内容是否来自文档
        return EvaluationResult(
            score=0.5,  # 中性分数
            details={"no_citations": True},
            passed=True,
        )
    
    valid_citations = 0
    invalid_citations = []
    
    for cite in citations:
        cite_idx = int(cite) - 1  # 转为 0-indexed
        if 0 <= cite_idx < len(documents):
            valid_citations += 1
        else:
            invalid_citations.append(cite)
    
    total_citations = len(citations)
    score = valid_citations / total_citations if total_citations > 0 else 0
    
    return EvaluationResult(
        score=score,
        details={
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
        },
        passed=score >= threshold,
    )


def factual_consistency(
    answer: str,
    context: str,
    llm=None,
    threshold: float = 0.7,
) -> EvaluationResult:
    """
    检查答案与文献的事实一致性
    
    使用 LLM 或规则判断答案中的陈述是否与文献一致。
    
    Args:
        answer: 生成的答案
        context: 参考文献内容
        llm: LLM 实例（可选，用于高级检查）
        threshold: 通过阈值
        
    Returns:
        EvaluationResult
    """
    if not answer or not context:
        return EvaluationResult(
            score=0.0,
            details={"error": "Empty input"},
            passed=False,
        )
    
    # 简单实现：检查答案中的关键词是否出现在文献中
    answer_keywords = set(_extract_keywords(answer))
    context_keywords = set(_extract_keywords(context))
    
    if not answer_keywords:
        return EvaluationResult(
            score=0.5,
            details={"no_keywords": True},
            passed=True,
        )
    
    # 计算答案关键词在文献中出现的比例
    matched = answer_keywords & context_keywords
    coverage = len(matched) / len(answer_keywords)
    
    # 检查是否有明显的矛盾关键词（简单启发式）
    contradictions = _check_contradictions(answer, context)
    
    # 如果有矛盾，降低分数
    if contradictions:
        coverage *= 0.5
    
    return EvaluationResult(
        score=coverage,
        details={
            "matched_keywords": list(matched),
            "answer_keywords": list(answer_keywords),
            "contradictions": contradictions,
        },
        passed=coverage >= threshold,
    )


def _extract_keywords(text: str) -> List[str]:
    """提取关键词"""
    # 医学相关关键词模式
    medical_patterns = [
        r'[\u4e00-\u9fff]+病',  # XX病
        r'[\u4e00-\u9fff]+症',  # XX症
        r'[\u4e00-\u9fff]+炎',  # XX炎
        r'[\u4e00-\u9fff]+癌',  # XX癌
        r'\d+[\.\d]*\s*(mg|ml|g|%|毫克|毫升|克)',  # 数值+单位
    ]
    
    keywords = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, text)
        keywords.extend(matches)
    
    # 添加常见医学术语
    common_terms = [
        '治疗', '诊断', '症状', '预防', '药物', '手术',
        '检查', '禁忌', '副作用', '剂量', '疗程',
    ]
    for term in common_terms:
        if term in text:
            keywords.append(term)
    
    return keywords


def _check_contradictions(answer: str, context: str) -> List[str]:
    """检查是否有矛盾（简单启发式）"""
    contradictions = []
    
    # 检查否定词使用不一致
    negation_patterns = [
        (r'不能', r'能够?'),
        (r'禁止', r'建议'),
        (r'避免', r'推荐'),
    ]
    
    for neg_pattern, pos_pattern in negation_patterns:
        if re.search(neg_pattern, answer) and re.search(pos_pattern, context):
            contradictions.append(f"Answer uses '{neg_pattern}' but context has positive")
        elif re.search(pos_pattern, answer) and re.search(neg_pattern, context):
            contradictions.append(f"Answer uses '{pos_pattern}' but context has negative")
    
    return contradictions


class MedicalQAMetrics:
    """
    医疗问答综合评估指标
    
    组合多个指标进行综合评估。
    """
    
    def __init__(
        self,
        embedder=None,
        llm=None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            embedder: Embedding 模型
            llm: LLM 实例
            weights: 各指标权重
        """
        self.embedder = embedder
        self.llm = llm
        self.weights = weights or {
            "semantic_f1": 0.4,
            "citation_accuracy": 0.3,
            "factual_consistency": 0.3,
        }
    
    def evaluate(
        self,
        prediction: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None,
        documents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        综合评估
        
        Args:
            prediction: 预测答案
            ground_truth: 标准答案（可选）
            context: 参考文献（可选）
            documents: 检索文档列表（可选）
            
        Returns:
            评估结果字典
        """
        results = {}
        total_score = 0.0
        total_weight = 0.0
        
        # 语义 F1（需要标准答案）
        if ground_truth:
            sf1_result = semantic_f1(
                prediction,
                ground_truth,
                self.embedder,
            )
            results["semantic_f1"] = {
                "score": sf1_result.score,
                "passed": sf1_result.passed,
                "details": sf1_result.details,
            }
            total_score += sf1_result.score * self.weights["semantic_f1"]
            total_weight += self.weights["semantic_f1"]
        
        # 引用准确率（需要文档列表）
        if documents:
            ca_result = citation_accuracy(prediction, documents)
            results["citation_accuracy"] = {
                "score": ca_result.score,
                "passed": ca_result.passed,
                "details": ca_result.details,
            }
            total_score += ca_result.score * self.weights["citation_accuracy"]
            total_weight += self.weights["citation_accuracy"]
        
        # 事实一致性（需要参考文献）
        if context:
            fc_result = factual_consistency(
                prediction,
                context,
                self.llm,
            )
            results["factual_consistency"] = {
                "score": fc_result.score,
                "passed": fc_result.passed,
                "details": fc_result.details,
            }
            total_score += fc_result.score * self.weights["factual_consistency"]
            total_weight += self.weights["factual_consistency"]
        
        # 计算综合分数
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0.0
        
        results["overall"] = {
            "score": overall_score,
            "passed": overall_score >= 0.6,
        }
        
        return results
    
    def __call__(
        self,
        example,
        prediction,
        trace=None,
    ) -> float:
        """
        DSPy 评估函数接口
        
        用于 DSPy 优化器的评估函数。
        
        Args:
            example: DSPy Example 对象
            prediction: DSPy Prediction 对象
            trace: 执行追踪（可选）
            
        Returns:
            评估分数 (0-1)
        """
        # 从 example 获取标准答案和上下文
        ground_truth = getattr(example, "answer", None)
        context = getattr(example, "context", None)
        
        # 从 prediction 获取预测答案
        pred_answer = getattr(prediction, "answer", str(prediction))
        
        # 执行评估
        results = self.evaluate(
            prediction=pred_answer,
            ground_truth=ground_truth,
            context=context,
        )
        
        return results["overall"]["score"]


def create_dspy_metric(
    metric_type: str = "semantic_f1",
    threshold: float = 0.7,
    **kwargs,
) -> Callable:
    """
    创建 DSPy 兼容的评估函数
    
    Args:
        metric_type: 指标类型
        threshold: 通过阈值
        **kwargs: 额外参数
        
    Returns:
        评估函数
    """
    def metric_fn(example, prediction, trace=None):
        ground_truth = getattr(example, "answer", "")
        pred_answer = getattr(prediction, "answer", str(prediction))
        
        if metric_type == "semantic_f1":
            result = semantic_f1(pred_answer, ground_truth, threshold=threshold)
        elif metric_type == "citation_accuracy":
            documents = getattr(example, "documents", [])
            result = citation_accuracy(pred_answer, documents, threshold=threshold)
        elif metric_type == "factual_consistency":
            context = getattr(example, "context", "")
            result = factual_consistency(pred_answer, context, threshold=threshold)
        else:
            result = EvaluationResult(score=0.0, passed=False)
        
        return result.score
    
    return metric_fn
