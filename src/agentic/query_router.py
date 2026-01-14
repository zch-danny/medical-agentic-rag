"""
查询路由器 - 根据查询类型选择最优检索策略

路由决策：
- VECTOR: 语义相似检索（适合概念性问题）
- BM25: 精确关键词匹配（适合术语查询）
- HYBRID: 混合检索（默认，适合复杂问题）
- DIRECT: 直接回答（通用知识，无需检索）
- WEB: 联网搜索（最新信息，本地库不足）
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from loguru import logger


class QueryType(Enum):
    """查询类型"""
    VECTOR = "vector"       # 语义向量检索
    BM25 = "bm25"           # 关键词精确匹配
    HYBRID = "hybrid"       # 混合检索（默认）
    DIRECT = "direct"       # 直接回答，无需检索
    WEB = "web"             # 联网搜索


@dataclass
class RouteDecision:
    """路由决策结果"""
    query_type: QueryType
    confidence: float       # 置信度 0-1
    reason: str             # 决策原因
    suggested_alpha: float  # 建议的混合检索权重
    

class QueryRouter:
    """
    查询路由器
    
    根据查询特征选择最优检索策略
    """
    
    # 医学术语和缩写（用于识别精确匹配需求）
    MEDICAL_TERMS = {
        # 常见疾病缩写
        "dm", "t2dm", "t1dm", "htn", "copd", "ckd", "chf", "mi", "cad",
        "acs", "dvt", "pe", "ards", "aki", "uti", "tb", "hiv", "aids",
        # 药物缩写
        "nsaid", "ace", "arb", "ccb", "bb", "ppi", "ssri", "snri",
        # 检查缩写
        "ct", "mri", "ecg", "ekg", "eeg", "emg", "pet", "spect",
        # 指标缩写
        "bp", "hr", "rr", "bmi", "hba1c", "ldl", "hdl", "tg", "alt", "ast",
        "bun", "cr", "gfr", "wbc", "rbc", "hgb", "plt", "inr", "pt", "ptt",
    }
    
    # 需要最新信息的关键词
    RECENT_KEYWORDS = {
        "最新", "2024", "2025", "2026", "新冠", "covid", "疫情",
        "最近", "近期", "刚刚", "新发现", "新研究", "新指南",
    }
    
    # 通用知识问题模式（可直接回答）
    GENERAL_PATTERNS = [
        r"^什么是.{2,10}[？?]?$",  # "什么是BMI？"
        r"^.{2,10}是什么[？?]?$",  # "BMI是什么？"
        r"^.{2,6}的定义",           # "高血压的定义"
        r"^.{2,6}的概念",           # "糖尿病的概念"
    ]
    
    # 精确查询模式（需要 BM25）
    EXACT_PATTERNS = [
        r"剂量|用量|用法",
        r"禁忌|禁忌症",
        r"适应症",
        r"不良反应|副作用",
        r"半衰期|药代动力学",
        r"参考值|正常值|正常范围",
        r"诊断标准|诊断依据",
        r"分期|分级|分类",
    ]
    
    def __init__(
        self,
        llm=None,
        use_llm_routing: bool = False,
        default_alpha: float = 0.7,
    ):
        """
        Args:
            llm: LLM 实例，用于复杂路由决策（可选）
            use_llm_routing: 是否使用 LLM 进行路由（更准确但更慢）
            default_alpha: 默认混合检索权重
        """
        self._llm = llm
        self._use_llm_routing = use_llm_routing
        self._default_alpha = default_alpha
        
        # 编译正则
        self._general_patterns = [re.compile(p) for p in self.GENERAL_PATTERNS]
        self._exact_patterns = [re.compile(p) for p in self.EXACT_PATTERNS]
        
    def route(self, query: str, context: Optional[str] = None) -> RouteDecision:
        """
        对查询进行路由决策
        
        Args:
            query: 用户查询
            context: 对话上下文（可选）
            
        Returns:
            RouteDecision 包含路由类型和原因
        """
        query_lower = query.lower().strip()
        
        # 1. 检查是否需要最新信息
        if self._needs_recent_info(query_lower):
            return RouteDecision(
                query_type=QueryType.WEB,
                confidence=0.8,
                reason="查询涉及最新信息，建议联网搜索",
                suggested_alpha=0.5,
            )
        
        # 2. 检查是否是通用知识问题
        if self._is_general_knowledge(query):
            return RouteDecision(
                query_type=QueryType.DIRECT,
                confidence=0.7,
                reason="通用知识问题，可直接回答",
                suggested_alpha=0.5,
            )
        
        # 3. 检查是否需要精确匹配
        exact_score = self._calc_exact_match_score(query_lower)
        if exact_score > 0.6:
            return RouteDecision(
                query_type=QueryType.BM25,
                confidence=exact_score,
                reason="查询包含需要精确匹配的医学术语或指标",
                suggested_alpha=0.3,  # 更偏向 BM25
            )
        
        # 4. 检查是否是纯语义问题
        semantic_score = self._calc_semantic_score(query_lower)
        if semantic_score > 0.7:
            return RouteDecision(
                query_type=QueryType.VECTOR,
                confidence=semantic_score,
                reason="概念性/语义性问题，适合向量检索",
                suggested_alpha=0.9,  # 更偏向向量
            )
        
        # 5. 默认使用混合检索
        return RouteDecision(
            query_type=QueryType.HYBRID,
            confidence=0.8,
            reason="复杂查询，使用混合检索获得最佳效果",
            suggested_alpha=self._default_alpha,
        )
    
    def _needs_recent_info(self, query: str) -> bool:
        """检查是否需要最新信息"""
        for keyword in self.RECENT_KEYWORDS:
            if keyword in query:
                return True
        return False
    
    def _is_general_knowledge(self, query: str) -> bool:
        """检查是否是通用知识问题"""
        for pattern in self._general_patterns:
            if pattern.search(query):
                # 进一步检查是否是简单定义问题
                if len(query) < 20:
                    return True
        return False
    
    def _calc_exact_match_score(self, query: str) -> float:
        """计算精确匹配得分"""
        score = 0.0
        
        # 检查医学缩写
        words = set(re.findall(r'\b\w+\b', query))
        medical_term_count = len(words & self.MEDICAL_TERMS)
        if medical_term_count > 0:
            score += 0.3 * min(medical_term_count, 2)
        
        # 检查精确模式
        for pattern in self._exact_patterns:
            if pattern.search(query):
                score += 0.3
                break
        
        # 检查数字（可能是剂量、指标等）
        if re.search(r'\d+\s*(mg|ml|g|kg|mmol|umol|%)', query):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calc_semantic_score(self, query: str) -> float:
        """计算语义检索得分"""
        score = 0.5  # 基础分
        
        # 问题类关键词增加语义得分
        semantic_keywords = ["为什么", "如何", "怎么", "机制", "原理", "关系", "影响", "作用"]
        for kw in semantic_keywords:
            if kw in query:
                score += 0.15
        
        # 长查询更适合语义
        if len(query) > 30:
            score += 0.1
        
        # 缺少具体数字/术语的查询更适合语义
        if not re.search(r'\d+', query) and not (set(re.findall(r'\b\w+\b', query.lower())) & self.MEDICAL_TERMS):
            score += 0.1
        
        return min(score, 1.0)
    
    def route_with_llm(self, query: str, context: Optional[str] = None) -> RouteDecision:
        """
        使用 LLM 进行更准确的路由决策（较慢）
        
        Args:
            query: 用户查询
            context: 对话上下文
            
        Returns:
            RouteDecision
        """
        if self._llm is None:
            logger.warning("LLM 未配置，回退到规则路由")
            return self.route(query, context)
        
        prompt = f"""分析以下医疗查询，选择最合适的检索策略：

查询：{query}
{"上下文：" + context if context else ""}

可选策略：
1. VECTOR - 语义向量检索：适合概念性问题，如"糖尿病的发病机制是什么？"
2. BM25 - 关键词精确匹配：适合查找具体术语、剂量、指标，如"阿司匹林的常用剂量"
3. HYBRID - 混合检索：适合复杂问题，兼顾语义和关键词
4. DIRECT - 直接回答：简单定义问题，无需检索，如"什么是BMI？"
5. WEB - 联网搜索：需要最新信息，如"2024年糖尿病指南更新了什么？"

请只返回策略名称（VECTOR/BM25/HYBRID/DIRECT/WEB）和原因，格式：
策略：XXX
原因：XXX
"""
        
        try:
            response = self._llm.complete(prompt).text
            
            # 解析响应
            strategy = "HYBRID"  # 默认
            reason = "LLM 决策"
            
            for line in response.strip().split("\n"):
                if "策略" in line:
                    for qt in QueryType:
                        if qt.value.upper() in line.upper():
                            strategy = qt.value.upper()
                            break
                elif "原因" in line:
                    reason = line.split("：", 1)[-1].strip()
            
            query_type = QueryType[strategy]
            alpha_map = {
                QueryType.VECTOR: 0.9,
                QueryType.BM25: 0.3,
                QueryType.HYBRID: 0.7,
                QueryType.DIRECT: 0.5,
                QueryType.WEB: 0.5,
            }
            
            return RouteDecision(
                query_type=query_type,
                confidence=0.85,
                reason=reason,
                suggested_alpha=alpha_map.get(query_type, 0.7),
            )
            
        except Exception as e:
            logger.error(f"LLM 路由失败: {e}")
            return self.route(query, context)
