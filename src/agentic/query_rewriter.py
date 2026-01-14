"""
查询改写器 - 优化查询以提高检索效果

功能：
- 医学术语标准化
- 同义词扩展
- 消歧义
- 追问补全（基于对话上下文）
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from loguru import logger


@dataclass
class RewriteResult:
    """改写结果"""
    original_query: str           # 原始查询
    rewritten_query: str          # 改写后查询
    expanded_terms: List[str]     # 扩展的术语
    is_modified: bool             # 是否有修改
    reason: str                   # 改写原因


class QueryRewriter:
    """
    查询改写器
    
    通过术语标准化、同义词扩展等方式优化查询
    """
    
    # 医学术语同义词/缩写映射
    TERM_MAPPINGS: Dict[str, List[str]] = {
        # 疾病
        "糖尿病": ["diabetes", "DM", "血糖升高"],
        "高血压": ["hypertension", "HTN", "血压升高", "血压高"],
        "冠心病": ["冠状动脉粥样硬化性心脏病", "CAD", "coronary artery disease", "CHD"],
        "心梗": ["心肌梗死", "MI", "myocardial infarction", "急性心肌梗死", "AMI"],
        "脑梗": ["脑梗死", "缺血性脑卒中", "cerebral infarction"],
        "中风": ["脑卒中", "stroke", "脑血管意外", "CVA"],
        "肺炎": ["pneumonia", "肺部感染"],
        "肝炎": ["hepatitis", "肝脏炎症"],
        "肾衰": ["肾衰竭", "renal failure", "肾功能衰竭", "CKD"],
        
        # 症状
        "头疼": ["头痛", "headache", "头部疼痛"],
        "胸闷": ["胸闷气短", "胸部不适", "chest tightness"],
        "心慌": ["心悸", "palpitation", "心跳加快"],
        "气喘": ["呼吸困难", "dyspnea", "气促", "喘息"],
        
        # 药物
        "阿司匹林": ["aspirin", "ASA", "乙酰水杨酸"],
        "二甲双胍": ["metformin", "格华止", "美迪康"],
        "降压药": ["antihypertensive", "降血压药物"],
        "他汀": ["statin", "他汀类药物", "降脂药"],
        
        # 检查
        "心电图": ["ECG", "EKG", "electrocardiogram"],
        "CT": ["计算机断层扫描", "computed tomography"],
        "核磁": ["MRI", "磁共振", "magnetic resonance imaging"],
        "B超": ["超声", "ultrasound", "彩超"],
        "验血": ["血液检查", "blood test", "血常规"],
    }
    
    # 需要消歧义的术语
    AMBIGUOUS_TERMS: Dict[str, Dict[str, str]] = {
        "糖尿病": {
            "1型": "1型糖尿病 (T1DM)",
            "2型": "2型糖尿病 (T2DM)",
            "妊娠": "妊娠期糖尿病 (GDM)",
        },
        "高血压": {
            "原发": "原发性高血压",
            "继发": "继发性高血压",
            "妊娠": "妊娠期高血压",
        },
    }
    
    # 追问关键词
    FOLLOW_UP_PATTERNS = [
        r"^那.{0,5}呢[？?]?$",
        r"^.{0,3}怎么.{0,3}[？?]?$",
        r"^还有.{0,5}[？?]?$",
        r"^其他.{0,5}[？?]?$",
        r"^第[一二三四五]个",
    ]
    
    def __init__(
        self,
        llm=None,
        use_llm_rewrite: bool = False,
        enable_expansion: bool = True,
    ):
        """
        Args:
            llm: LLM 实例，用于复杂改写（可选）
            use_llm_rewrite: 是否使用 LLM 进行改写
            enable_expansion: 是否启用术语扩展
        """
        self._llm = llm
        self._use_llm_rewrite = use_llm_rewrite
        self._enable_expansion = enable_expansion
        
        # 构建反向映射
        self._build_reverse_mapping()
        
        # 编译追问模式
        self._follow_up_patterns = [re.compile(p) for p in self.FOLLOW_UP_PATTERNS]
    
    def _build_reverse_mapping(self):
        """构建反向映射（同义词 -> 标准术语）"""
        self._reverse_mapping: Dict[str, str] = {}
        for standard, synonyms in self.TERM_MAPPINGS.items():
            for syn in synonyms:
                self._reverse_mapping[syn.lower()] = standard
    
    def rewrite(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
        previous_query: Optional[str] = None,
    ) -> RewriteResult:
        """
        改写查询
        
        Args:
            query: 原始查询
            conversation_history: 对话历史（可选）
            previous_query: 上一个查询（用于追问补全）
            
        Returns:
            RewriteResult
        """
        original = query.strip()
        rewritten = original
        expanded_terms: List[str] = []
        reasons: List[str] = []
        
        # 1. 追问补全
        if self._is_follow_up(query) and previous_query:
            rewritten = self._complete_follow_up(query, previous_query, conversation_history)
            if rewritten != original:
                reasons.append("追问补全")
        
        # 2. 术语标准化
        standardized, std_terms = self._standardize_terms(rewritten)
        if standardized != rewritten:
            rewritten = standardized
            reasons.append(f"术语标准化: {', '.join(std_terms)}")
        
        # 3. 同义词扩展（可选）
        if self._enable_expansion:
            expanded = self._expand_synonyms(rewritten)
            expanded_terms.extend(expanded)
            if expanded:
                reasons.append(f"同义词扩展: {', '.join(expanded[:3])}")
        
        # 4. 消歧义提示（不修改查询，但记录）
        disambig = self._check_ambiguity(rewritten)
        if disambig:
            reasons.append(f"建议明确: {disambig}")
        
        is_modified = rewritten != original or len(expanded_terms) > 0
        
        return RewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            expanded_terms=expanded_terms,
            is_modified=is_modified,
            reason="; ".join(reasons) if reasons else "无需改写",
        )
    
    def _is_follow_up(self, query: str) -> bool:
        """检查是否是追问"""
        for pattern in self._follow_up_patterns:
            if pattern.search(query):
                return True
        
        # 短查询更可能是追问
        if len(query) < 10 and any(w in query for w in ["呢", "吗", "怎么", "那"]):
            return True
        
        return False
    
    def _complete_follow_up(
        self,
        query: str,
        previous_query: str,
        history: Optional[List[str]] = None,
    ) -> str:
        """补全追问查询"""
        # 提取上一个查询的主题
        topic = self._extract_topic(previous_query)
        
        if not topic:
            return query
        
        # 常见追问模式处理
        completions = {
            "怎么预防": f"{topic}的预防方法",
            "怎么治疗": f"{topic}的治疗方法",
            "怎么诊断": f"{topic}的诊断方法",
            "什么症状": f"{topic}的症状",
            "什么原因": f"{topic}的病因",
            "那呢": previous_query,  # 完全重复上一个问题
        }
        
        for pattern, completion in completions.items():
            if pattern in query:
                return completion
        
        # 通用补全：把主题加到追问前
        if "呢" in query:
            return f"{topic}{query.replace('呢', '')}"
        
        return f"{topic} {query}"
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """从查询中提取主题"""
        # 匹配"XXX的..."模式
        match = re.search(r'^(.{2,10})的', query)
        if match:
            return match.group(1)
        
        # 匹配"...XXX..."中的疾病名
        for term in self.TERM_MAPPINGS.keys():
            if term in query:
                return term
        
        return None
    
    def _standardize_terms(self, query: str) -> tuple:
        """标准化医学术语"""
        result = query
        standardized = []
        
        # 检查并替换同义词为标准术语
        query_lower = query.lower()
        for syn, standard in self._reverse_mapping.items():
            if syn in query_lower and standard not in query:
                # 不区分大小写替换
                pattern = re.compile(re.escape(syn), re.IGNORECASE)
                result = pattern.sub(standard, result)
                standardized.append(f"{syn} → {standard}")
        
        return result, standardized
    
    def _expand_synonyms(self, query: str) -> List[str]:
        """获取同义词扩展（不修改查询，仅返回扩展词）"""
        expanded = set()
        
        for term, synonyms in self.TERM_MAPPINGS.items():
            if term in query:
                # 添加一些常用同义词（不要太多）
                for syn in synonyms[:2]:
                    if syn.lower() not in query.lower():
                        expanded.add(syn)
        
        return list(expanded)
    
    def _check_ambiguity(self, query: str) -> Optional[str]:
        """检查是否需要消歧义"""
        for term, disambig in self.AMBIGUOUS_TERMS.items():
            if term in query:
                # 检查是否已经明确了类型
                has_specific = any(spec in query for spec in disambig.keys())
                if not has_specific:
                    options = list(disambig.values())
                    return f"'{term}'可能指: {', '.join(options[:3])}"
        
        return None
    
    def rewrite_with_llm(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
    ) -> RewriteResult:
        """
        使用 LLM 进行智能改写
        
        Args:
            query: 原始查询
            conversation_history: 对话历史
            
        Returns:
            RewriteResult
        """
        if self._llm is None:
            logger.warning("LLM 未配置，回退到规则改写")
            return self.rewrite(query, conversation_history)
        
        history_text = ""
        if conversation_history:
            history_text = "\n".join([f"- {h}" for h in conversation_history[-3:]])
        
        prompt = f"""你是一个医疗查询优化专家。请优化以下查询以提高检索效果。

原始查询：{query}
{"对话历史：" + history_text if history_text else ""}

优化要求：
1. 如果是追问（如"那怎么预防？"），补全主题
2. 标准化医学术语（如"高血压"保持，但可添加英文"hypertension"）
3. 扩展同义词但保持简洁
4. 不要改变原意

请返回优化后的查询和原因，格式：
优化查询：XXX
原因：XXX
"""
        
        try:
            response = self._llm.complete(prompt).text
            
            rewritten = query
            reason = "LLM 改写"
            
            for line in response.strip().split("\n"):
                if "优化查询" in line:
                    rewritten = line.split("：", 1)[-1].strip()
                elif "原因" in line:
                    reason = line.split("：", 1)[-1].strip()
            
            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                expanded_terms=[],
                is_modified=rewritten != query,
                reason=reason,
            )
            
        except Exception as e:
            logger.error(f"LLM 改写失败: {e}")
            return self.rewrite(query, conversation_history)
