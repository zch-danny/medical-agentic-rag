"""
DSPy Signatures 定义

Signature 是 DSPy 的核心概念，定义了任务的输入输出规范。
通过 docstring 和字段描述来引导 LLM 生成更好的输出。
"""

import dspy


class MedicalQA(dspy.Signature):
    """
    基于医疗文献回答用户的医疗健康问题。
    
    要求：
    1. 答案必须基于提供的文献内容
    2. 使用专业但易懂的语言
    3. 如果文献不足以回答问题，请明确说明
    4. 重要的医学建议需要标注"建议咨询医生"
    """
    
    context = dspy.InputField(
        desc="检索到的医疗文献内容，包含疾病、症状、治疗等信息"
    )
    question = dspy.InputField(
        desc="用户提出的医疗健康问题"
    )
    answer = dspy.OutputField(
        desc="基于文献的专业回答，包含关键信息和必要的医学说明"
    )


class MedicalQAWithCitation(dspy.Signature):
    """
    基于医疗文献回答问题，并提供引用来源。
    
    要求：
    1. 答案必须基于提供的文献
    2. 用 [1], [2] 等标记引用来源
    3. 在答案末尾列出引用的文献
    """
    
    context = dspy.InputField(
        desc="检索到的医疗文献，每条文献有唯一编号"
    )
    question = dspy.InputField(
        desc="用户的医疗问题"
    )
    answer = dspy.OutputField(
        desc="带引用标记的专业回答，格式：回答内容[引用编号]"
    )
    citations = dspy.OutputField(
        desc="引用列表，格式：[编号] 文献标题或来源"
    )


class QueryRewrite(dspy.Signature):
    """
    优化医疗查询以提高检索效果。
    
    改写策略：
    1. 医学术语标准化（如：糖尿病 → 2型糖尿病 / 1型糖尿病）
    2. 同义词扩展（如：高血压 → 高血压 高血压症 hypertension）
    3. 补充隐含信息（如：对于追问"怎么预防？"，需要补充疾病名称）
    4. 消除歧义
    """
    
    original_query = dspy.InputField(
        desc="用户原始查询"
    )
    conversation_history = dspy.InputField(
        desc="对话历史，用于理解追问的上下文"
    )
    rewritten_query = dspy.OutputField(
        desc="优化后的查询，包含医学术语扩展和上下文补充"
    )


class QueryRewriteWithReasoning(dspy.Signature):
    """
    带推理过程的查询改写。
    
    先分析查询的问题类型和可能的检索困难，
    然后给出优化后的查询。
    """
    
    original_query = dspy.InputField(
        desc="用户原始查询"
    )
    conversation_history = dspy.InputField(
        desc="对话历史"
    )
    reasoning = dspy.OutputField(
        desc="分析：1) 查询类型 2) 可能的检索困难 3) 改写策略"
    )
    rewritten_query = dspy.OutputField(
        desc="优化后的查询"
    )


class RelevanceEval(dspy.Signature):
    """
    评估检索文档与查询的相关性。
    
    相关性等级：
    - highly_relevant: 文档直接回答了问题
    - partially_relevant: 文档包含部分相关信息
    - not_relevant: 文档与问题无关
    """
    
    query = dspy.InputField(
        desc="用户查询"
    )
    document = dspy.InputField(
        desc="待评估的文档内容"
    )
    relevance = dspy.OutputField(
        desc="相关性评级：highly_relevant / partially_relevant / not_relevant"
    )
    reason = dspy.OutputField(
        desc="判断理由"
    )


class AnswerEval(dspy.Signature):
    """
    评估生成答案的质量。
    
    评估维度：
    1. 准确性：答案是否基于提供的文献
    2. 完整性：是否回答了问题的所有方面
    3. 清晰度：答案是否易于理解
    4. 安全性：是否包含必要的医学警示
    """
    
    question = dspy.InputField(
        desc="用户问题"
    )
    context = dspy.InputField(
        desc="参考文献"
    )
    answer = dspy.InputField(
        desc="待评估的答案"
    )
    accuracy_score = dspy.OutputField(
        desc="准确性评分 (1-5)"
    )
    completeness_score = dspy.OutputField(
        desc="完整性评分 (1-5)"
    )
    clarity_score = dspy.OutputField(
        desc="清晰度评分 (1-5)"
    )
    feedback = dspy.OutputField(
        desc="改进建议"
    )


class SufficiencyCheck(dspy.Signature):
    """
    判断检索结果是否足够回答问题。
    
    用于决定是否需要：
    1. 继续检索更多文档
    2. 改写查询重新检索
    3. 直接生成答案
    """
    
    query = dspy.InputField(
        desc="用户查询"
    )
    documents = dspy.InputField(
        desc="已检索到的文档列表"
    )
    is_sufficient = dspy.OutputField(
        desc="是否足够回答：yes / no / partial"
    )
    missing_info = dspy.OutputField(
        desc="如果不足，说明缺少什么信息"
    )
    suggestion = dspy.OutputField(
        desc="建议的下一步操作"
    )


class MedicalTermStandardization(dspy.Signature):
    """
    医学术语标准化。
    
    将用户使用的通俗词汇转换为标准医学术语，
    同时保留原始词汇以提高检索召回率。
    """
    
    term = dspy.InputField(
        desc="用户输入的医学相关词汇"
    )
    standard_term = dspy.OutputField(
        desc="标准医学术语"
    )
    synonyms = dspy.OutputField(
        desc="同义词和相关术语，逗号分隔"
    )
    icd_code = dspy.OutputField(
        desc="ICD-10 编码（如果适用）"
    )
