"""
DSPy Modules 实现

Module 是 DSPy 中可优化的组件，封装了 Signature 和执行逻辑。
支持通过 BootstrapFewShot 或 MIPRO 等优化器进行自动优化。
"""

from typing import Any, Dict, List, Optional, Union
import dspy

from .signatures import (
    MedicalQA,
    MedicalQAWithCitation,
    QueryRewrite,
    QueryRewriteWithReasoning,
    RelevanceEval,
    AnswerEval,
    SufficiencyCheck,
)


class OptimizedRAG(dspy.Module):
    """
    优化的 RAG 模块
    
    使用 DSPy 的 Predict 来生成答案，支持自动优化。
    """
    
    def __init__(self, with_citation: bool = False):
        """
        Args:
            with_citation: 是否生成带引用的答案
        """
        super().__init__()
        
        self.with_citation = with_citation
        
        if with_citation:
            self.generate = dspy.Predict(MedicalQAWithCitation)
        else:
            self.generate = dspy.Predict(MedicalQA)
    
    def forward(
        self,
        question: str,
        context: Union[str, List[str]],
    ) -> dspy.Prediction:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 检索到的文档（字符串或列表）
            
        Returns:
            DSPy Prediction 对象
        """
        # 处理 context
        if isinstance(context, list):
            # 格式化文档列表
            formatted_context = "\n\n".join([
                f"[{i+1}] {doc}" for i, doc in enumerate(context)
            ])
        else:
            formatted_context = context
        
        # 生成答案
        if self.with_citation:
            prediction = self.generate(
                context=formatted_context,
                question=question,
            )
        else:
            prediction = self.generate(
                context=formatted_context,
                question=question,
            )
        
        return prediction


class ChainOfThoughtRAG(dspy.Module):
    """
    使用 Chain of Thought 增强的 RAG 模块
    
    先进行推理分析，再生成答案，提高答案质量。
    """
    
    def __init__(self):
        super().__init__()
        
        # 使用 ChainOfThought 而不是简单的 Predict
        self.generate = dspy.ChainOfThought(MedicalQA)
    
    def forward(
        self,
        question: str,
        context: Union[str, List[str]],
    ) -> dspy.Prediction:
        """
        带推理的答案生成
        
        Args:
            question: 用户问题
            context: 检索到的文档
            
        Returns:
            DSPy Prediction 对象，包含 rationale 和 answer
        """
        if isinstance(context, list):
            formatted_context = "\n\n".join([
                f"[{i+1}] {doc}" for i, doc in enumerate(context)
            ])
        else:
            formatted_context = context
        
        prediction = self.generate(
            context=formatted_context,
            question=question,
        )
        
        return prediction


class OptimizedRewriter(dspy.Module):
    """
    优化的查询改写模块
    
    支持两种模式：
    1. 简单改写：直接输出改写结果
    2. 带推理改写：先分析再改写
    """
    
    def __init__(self, with_reasoning: bool = False):
        """
        Args:
            with_reasoning: 是否使用带推理的改写
        """
        super().__init__()
        
        self.with_reasoning = with_reasoning
        
        if with_reasoning:
            self.rewrite = dspy.ChainOfThought(QueryRewriteWithReasoning)
        else:
            self.rewrite = dspy.Predict(QueryRewrite)
    
    def forward(
        self,
        original_query: str,
        conversation_history: str = "",
    ) -> dspy.Prediction:
        """
        改写查询
        
        Args:
            original_query: 原始查询
            conversation_history: 对话历史
            
        Returns:
            DSPy Prediction 对象
        """
        prediction = self.rewrite(
            original_query=original_query,
            conversation_history=conversation_history,
        )
        
        return prediction


class OptimizedEvaluator(dspy.Module):
    """
    优化的结果评估模块
    
    评估检索结果的相关性和充分性。
    """
    
    def __init__(self):
        super().__init__()
        
        self.relevance_eval = dspy.Predict(RelevanceEval)
        self.sufficiency_check = dspy.Predict(SufficiencyCheck)
    
    def evaluate_relevance(
        self,
        query: str,
        document: str,
    ) -> dspy.Prediction:
        """
        评估单个文档的相关性
        
        Args:
            query: 用户查询
            document: 待评估文档
            
        Returns:
            包含 relevance 和 reason 的 Prediction
        """
        return self.relevance_eval(
            query=query,
            document=document,
        )
    
    def check_sufficiency(
        self,
        query: str,
        documents: List[str],
    ) -> dspy.Prediction:
        """
        检查文档集合是否足够回答问题
        
        Args:
            query: 用户查询
            documents: 文档列表
            
        Returns:
            包含 is_sufficient, missing_info, suggestion 的 Prediction
        """
        formatted_docs = "\n\n".join([
            f"[{i+1}] {doc}" for i, doc in enumerate(documents)
        ])
        
        return self.sufficiency_check(
            query=query,
            documents=formatted_docs,
        )
    
    def forward(
        self,
        query: str,
        documents: List[str],
    ) -> Dict[str, Any]:
        """
        完整评估流程
        
        Args:
            query: 用户查询
            documents: 文档列表
            
        Returns:
            评估结果字典
        """
        # 评估每个文档的相关性
        relevance_results = []
        for doc in documents:
            result = self.evaluate_relevance(query, doc)
            relevance_results.append({
                "document": doc[:100] + "...",
                "relevance": result.relevance,
                "reason": result.reason,
            })
        
        # 检查整体充分性
        sufficiency = self.check_sufficiency(query, documents)
        
        return {
            "relevance_results": relevance_results,
            "is_sufficient": sufficiency.is_sufficient,
            "missing_info": sufficiency.missing_info,
            "suggestion": sufficiency.suggestion,
        }


class AnswerQualityEvaluator(dspy.Module):
    """
    答案质量评估模块
    
    评估生成答案的多个维度。
    """
    
    def __init__(self):
        super().__init__()
        self.evaluator = dspy.Predict(AnswerEval)
    
    def forward(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        评估答案质量
        
        Args:
            question: 用户问题
            context: 参考文献
            answer: 待评估答案
            
        Returns:
            评估结果字典
        """
        result = self.evaluator(
            question=question,
            context=context,
            answer=answer,
        )
        
        # 解析分数
        try:
            accuracy = int(result.accuracy_score)
        except (ValueError, AttributeError):
            accuracy = 3
        
        try:
            completeness = int(result.completeness_score)
        except (ValueError, AttributeError):
            completeness = 3
        
        try:
            clarity = int(result.clarity_score)
        except (ValueError, AttributeError):
            clarity = 3
        
        return {
            "accuracy_score": accuracy,
            "completeness_score": completeness,
            "clarity_score": clarity,
            "overall_score": (accuracy + completeness + clarity) / 3,
            "feedback": result.feedback,
        }


class MedicalRAGPipeline(dspy.Module):
    """
    完整的医疗 RAG 流水线
    
    集成查询改写、答案生成和质量评估。
    """
    
    def __init__(
        self,
        retriever=None,
        with_rewrite: bool = True,
        with_citation: bool = True,
    ):
        """
        Args:
            retriever: 检索器实例（可选）
            with_rewrite: 是否启用查询改写
            with_citation: 是否生成带引用的答案
        """
        super().__init__()
        
        self.retriever = retriever
        self.with_rewrite = with_rewrite
        
        if with_rewrite:
            self.rewriter = OptimizedRewriter(with_reasoning=True)
        
        self.generator = OptimizedRAG(with_citation=with_citation)
    
    def forward(
        self,
        question: str,
        context: Optional[Union[str, List[str]]] = None,
        conversation_history: str = "",
    ) -> Dict[str, Any]:
        """
        执行完整的 RAG 流程
        
        Args:
            question: 用户问题
            context: 预先检索的文档（可选）
            conversation_history: 对话历史
            
        Returns:
            包含答案和中间结果的字典
        """
        result = {
            "original_question": question,
        }
        
        # 查询改写
        if self.with_rewrite:
            rewrite_result = self.rewriter(
                original_query=question,
                conversation_history=conversation_history,
            )
            rewritten_query = rewrite_result.rewritten_query
            result["rewritten_query"] = rewritten_query
            result["rewrite_reasoning"] = getattr(rewrite_result, "reasoning", None)
        else:
            rewritten_query = question
        
        # 检索（如果提供了检索器且没有预先提供 context）
        if context is None and self.retriever is not None:
            try:
                retrieved_docs = self.retriever.search(rewritten_query)
                context = [doc.text if hasattr(doc, "text") else str(doc) for doc in retrieved_docs]
                result["retrieved_documents"] = context
            except Exception as e:
                context = []
                result["retrieval_error"] = str(e)
        
        # 生成答案
        if context:
            gen_result = self.generator(
                question=question,
                context=context,
            )
            result["answer"] = gen_result.answer
            if hasattr(gen_result, "citations"):
                result["citations"] = gen_result.citations
            if hasattr(gen_result, "rationale"):
                result["rationale"] = gen_result.rationale
        else:
            result["answer"] = "抱歉，未找到相关的医疗文献来回答您的问题。"
        
        return result
