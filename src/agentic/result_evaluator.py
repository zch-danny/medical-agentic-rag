"""
结果评估器 - 评估检索结果质量，决定下一步行动

评估维度：
- 相关性：文档是否与问题相关
- 充分性：信息是否足够回答问题
- 质量：文档的可信度和专业度

决策输出：
- SUFFICIENT: 信息充分，可生成答案
- PARTIAL: 部分相关，建议改写查询重试
- INSUFFICIENT: 不相关，尝试其他检索策略
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class EvaluationDecision(Enum):
    """评估决策"""
    SUFFICIENT = "sufficient"       # 充分，可生成答案
    PARTIAL = "partial"             # 部分相关，建议重试
    INSUFFICIENT = "insufficient"   # 不相关，换策略


@dataclass
class EvaluationResult:
    """评估结果"""
    decision: EvaluationDecision
    relevance_score: float      # 相关性得分 0-1
    sufficiency_score: float    # 充分性得分 0-1
    reason: str                 # 评估原因
    suggestions: List[str]      # 改进建议
    best_docs_indices: List[int]  # 最相关文档的索引


class ResultEvaluator:
    """
    结果评估器
    
    评估检索结果的质量，决定是否需要重试或改变策略
    """
    
    # 相关性阈值
    RELEVANCE_THRESHOLD_HIGH = 0.7
    RELEVANCE_THRESHOLD_LOW = 0.4
    
    # 充分性阈值
    SUFFICIENCY_THRESHOLD = 0.5
    
    # 最小相关文档数
    MIN_RELEVANT_DOCS = 2
    
    def __init__(
        self,
        llm=None,
        use_llm_evaluation: bool = False,
        relevance_threshold: float = 0.5,
    ):
        """
        Args:
            llm: LLM 实例，用于语义评估（可选）
            use_llm_evaluation: 是否使用 LLM 进行评估
            relevance_threshold: 相关性阈值
        """
        self._llm = llm
        self._use_llm_evaluation = use_llm_evaluation
        self._relevance_threshold = relevance_threshold
    
    def evaluate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        scores: Optional[List[float]] = None,
    ) -> EvaluationResult:
        """
        评估检索结果
        
        Args:
            query: 原始查询
            documents: 检索到的文档列表
            scores: 文档得分列表（可选）
            
        Returns:
            EvaluationResult
        """
        if not documents:
            return EvaluationResult(
                decision=EvaluationDecision.INSUFFICIENT,
                relevance_score=0.0,
                sufficiency_score=0.0,
                reason="未检索到任何文档",
                suggestions=["尝试更宽泛的查询", "检查向量库是否有数据"],
                best_docs_indices=[],
            )
        
        # 提取分数
        if scores is None:
            scores = [self._extract_score(doc) for doc in documents]
        
        # 计算相关性得分
        relevance_score = self._calc_relevance_score(query, documents, scores)
        
        # 计算充分性得分
        sufficiency_score = self._calc_sufficiency_score(query, documents, scores)
        
        # 找出最相关的文档
        best_indices = self._find_best_docs(documents, scores)
        
        # 做出决策
        decision, reason, suggestions = self._make_decision(
            relevance_score, sufficiency_score, len(best_indices)
        )
        
        return EvaluationResult(
            decision=decision,
            relevance_score=relevance_score,
            sufficiency_score=sufficiency_score,
            reason=reason,
            suggestions=suggestions,
            best_docs_indices=best_indices,
        )
    
    def _extract_score(self, doc: Dict) -> float:
        """从文档中提取分数"""
        # 优先 rerank_score
        score = doc.get("rerank_score")
        if score is not None:
            return float(score)
        
        score = doc.get("score")
        if score is not None:
            return float(score)
        
        distance = doc.get("distance")
        if distance is not None:
            return max(0, 1 - float(distance))
        
        return 0.5  # 默认中等分数
    
    def _calc_relevance_score(
        self,
        query: str,
        documents: List[Dict],
        scores: List[float],
    ) -> float:
        """
        计算整体相关性得分
        
        基于:
        - 文档得分分布
        - 高分文档数量
        - 查询词覆盖率
        """
        if not scores:
            return 0.0
        
        # 1. 平均分数（带权重，前几名权重高）
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # 2. 高分文档比例
        high_score_count = sum(1 for s in scores if s > self._relevance_threshold)
        high_score_ratio = high_score_count / len(scores)
        
        # 3. 查询词覆盖（简单检查）
        query_terms = set(query.lower().split())
        coverage = 0.0
        for doc in documents[:3]:  # 只看前 3 个
            text = self._get_doc_text(doc).lower()
            matches = sum(1 for term in query_terms if term in text)
            coverage = max(coverage, matches / max(len(query_terms), 1))
        
        # 综合得分
        relevance = 0.5 * weighted_avg + 0.3 * high_score_ratio + 0.2 * coverage
        return min(relevance, 1.0)
    
    def _calc_sufficiency_score(
        self,
        query: str,
        documents: List[Dict],
        scores: List[float],
    ) -> float:
        """
        计算信息充分性得分
        
        基于:
        - 高质量文档数量
        - 文档内容长度
        - 信息多样性
        """
        if not documents:
            return 0.0
        
        # 1. 足够数量的高分文档
        relevant_count = sum(1 for s in scores if s > self._relevance_threshold)
        count_score = min(relevant_count / self.MIN_RELEVANT_DOCS, 1.0)
        
        # 2. 内容长度（太短可能信息不足）
        total_length = sum(len(self._get_doc_text(doc)) for doc in documents[:5])
        length_score = min(total_length / 1000, 1.0)  # 假设 1000 字足够
        
        # 3. 来源多样性
        sources = set()
        for doc in documents[:5]:
            entity = doc.get("entity", doc)
            source = entity.get("source", "")
            if source:
                sources.add(source)
        diversity_score = min(len(sources) / 3, 1.0)  # 3 个不同来源算多样
        
        # 综合得分
        sufficiency = 0.4 * count_score + 0.4 * length_score + 0.2 * diversity_score
        return sufficiency
    
    def _get_doc_text(self, doc: Dict) -> str:
        """获取文档文本"""
        entity = doc.get("entity", doc)
        return entity.get("original_text") or entity.get("text", "")
    
    def _find_best_docs(
        self,
        documents: List[Dict],
        scores: List[float],
    ) -> List[int]:
        """找出最相关的文档索引"""
        best = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            if score >= self._relevance_threshold:
                best.append(i)
        
        # 至少返回前 3 个（如果有的话）
        if len(best) < 3:
            for i in range(min(3, len(documents))):
                if i not in best:
                    best.append(i)
        
        return sorted(best)[:5]  # 最多 5 个
    
    def _make_decision(
        self,
        relevance_score: float,
        sufficiency_score: float,
        relevant_doc_count: int,
    ) -> tuple:
        """做出评估决策"""
        suggestions = []
        
        # 高相关性 + 高充分性 = 可以生成答案
        if relevance_score >= self.RELEVANCE_THRESHOLD_HIGH and sufficiency_score >= self.SUFFICIENCY_THRESHOLD:
            return (
                EvaluationDecision.SUFFICIENT,
                f"检索结果相关性高({relevance_score:.2f})且信息充分({sufficiency_score:.2f})",
                [],
            )
        
        # 中等相关性 = 部分相关，可以尝试改写
        if relevance_score >= self.RELEVANCE_THRESHOLD_LOW:
            suggestions = [
                "尝试添加同义词或相关术语",
                "明确查询的具体方面（如症状、治疗、预防）",
            ]
            if sufficiency_score < self.SUFFICIENCY_THRESHOLD:
                suggestions.append("增加检索数量(top_k)")
            
            return (
                EvaluationDecision.PARTIAL,
                f"检索结果部分相关({relevance_score:.2f})，建议优化查询",
                suggestions,
            )
        
        # 低相关性 = 不相关，需要换策略
        suggestions = [
            "尝试不同的检索策略（如从向量改为关键词）",
            "检查查询是否在知识库覆盖范围内",
            "尝试更通用的查询词",
        ]
        
        return (
            EvaluationDecision.INSUFFICIENT,
            f"检索结果相关性低({relevance_score:.2f})，建议更换检索策略",
            suggestions,
        )
    
    def evaluate_with_llm(
        self,
        query: str,
        documents: List[Dict],
    ) -> EvaluationResult:
        """
        使用 LLM 进行深度评估
        
        Args:
            query: 查询
            documents: 文档列表
            
        Returns:
            EvaluationResult
        """
        if self._llm is None:
            logger.warning("LLM 未配置，回退到规则评估")
            return self.evaluate(query, documents)
        
        # 准备文档摘要
        doc_summaries = []
        for i, doc in enumerate(documents[:5]):
            text = self._get_doc_text(doc)[:200]
            doc_summaries.append(f"[{i+1}] {text}...")
        
        docs_text = "\n".join(doc_summaries)
        
        prompt = f"""评估以下检索结果是否能回答用户问题。

用户问题：{query}

检索到的文档：
{docs_text}

请评估：
1. 相关性（0-1）：文档是否与问题相关
2. 充分性（0-1）：信息是否足够回答问题
3. 决策：SUFFICIENT（充分）/ PARTIAL（部分相关）/ INSUFFICIENT（不相关）
4. 原因：简要说明

格式：
相关性：X.X
充分性：X.X
决策：XXX
原因：XXX
"""
        
        try:
            response = self._llm.complete(prompt).text
            
            relevance = 0.5
            sufficiency = 0.5
            decision = EvaluationDecision.PARTIAL
            reason = "LLM 评估"
            
            for line in response.strip().split("\n"):
                if "相关性" in line:
                    try:
                        relevance = float(line.split("：")[-1].strip())
                    except:
                        pass
                elif "充分性" in line:
                    try:
                        sufficiency = float(line.split("：")[-1].strip())
                    except:
                        pass
                elif "决策" in line:
                    dec_text = line.split("：")[-1].strip().upper()
                    for d in EvaluationDecision:
                        if d.value.upper() in dec_text:
                            decision = d
                            break
                elif "原因" in line:
                    reason = line.split("：", 1)[-1].strip()
            
            scores = [self._extract_score(doc) for doc in documents]
            best_indices = self._find_best_docs(documents, scores)
            
            return EvaluationResult(
                decision=decision,
                relevance_score=relevance,
                sufficiency_score=sufficiency,
                reason=reason,
                suggestions=[],
                best_docs_indices=best_indices,
            )
            
        except Exception as e:
            logger.error(f"LLM 评估失败: {e}")
            return self.evaluate(query, documents)
    
    def filter_relevant_docs(
        self,
        documents: List[Dict],
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        过滤出相关文档
        
        Args:
            documents: 文档列表
            min_score: 最低分数阈值
            
        Returns:
            过滤后的文档列表
        """
        threshold = min_score if min_score is not None else self._relevance_threshold
        
        filtered = []
        for doc in documents:
            score = self._extract_score(doc)
            if score >= threshold:
                filtered.append(doc)
        
        return filtered
