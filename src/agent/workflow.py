"""
Agentic RAG 工作流

使用 LlamaIndex Workflow API 实现多步推理流程
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from loguru import logger

# 导入 agentic 组件（使用相对导入）
from ..agentic.query_router import QueryRouter, QueryType
from ..agentic.query_rewriter import QueryRewriter
from ..agentic.result_evaluator import ResultEvaluator, EvaluationDecision


# ============== 事件定义 ==============

@dataclass
class QueryReceivedEvent(Event):
    """查询接收事件"""
    query: str
    history: Optional[List[Dict]] = None


@dataclass
class QueryRewrittenEvent(Event):
    """查询改写完成事件"""
    original_query: str
    rewritten_query: str
    rewrite_info: Dict[str, Any] = None


@dataclass
class RouteDecidedEvent(Event):
    """路由决策完成事件"""
    query: str
    query_type: QueryType
    route_info: Dict[str, Any] = None


@dataclass
class RetrievalEvent(Event):
    """检索事件"""
    query: str
    query_type: QueryType


@dataclass
class RetrievalResultEvent(Event):
    """检索结果事件"""
    query: str
    documents: List[Any]
    retrieval_method: str


@dataclass
class EvaluationEvent(Event):
    """评估事件"""
    query: str
    documents: List[Any]
    attempt: int = 1


@dataclass
class RefinementNeededEvent(Event):
    """需要优化检索事件"""
    query: str
    original_documents: List[Any]
    feedback: str
    attempt: int


@dataclass
class GenerationEvent(Event):
    """生成事件"""
    query: str
    documents: List[Any]


@dataclass
class DirectAnswerEvent(Event):
    """直接回答事件（无需检索）"""
    query: str


# ============== 工作流定义 ==============

class AgenticRAGWorkflow(Workflow):
    """
    Agentic RAG 工作流
    
    流程：
    1. 接收查询
    2. 查询改写（标准化术语、处理追问）
    3. 路由决策（选择检索策略）
    4. 执行检索
    5. 评估结果
    6. 必要时优化检索
    7. 生成回答
    """
    
    def __init__(
        self,
        retriever=None,
        generator=None,
        query_router: Optional[QueryRouter] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        result_evaluator: Optional[ResultEvaluator] = None,
        max_refinement_attempts: int = 2,
        **kwargs,
    ):
        """
        Args:
            retriever: 检索器实例
            generator: 生成器实例
            query_router: 查询路由器
            query_rewriter: 查询改写器
            result_evaluator: 结果评估器
            max_refinement_attempts: 最大优化尝试次数
        """
        super().__init__(**kwargs)
        
        self.retriever = retriever
        self.generator = generator
        self.query_router = query_router or QueryRouter()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.result_evaluator = result_evaluator or ResultEvaluator()
        self.max_refinement_attempts = max_refinement_attempts
    
    @step
    async def receive_query(self, ctx: Context, ev: StartEvent) -> QueryReceivedEvent:
        """步骤1：接收查询"""
        query = ev.get("query", "")
        history = ev.get("history", [])
        
        logger.info(f"[Workflow] 接收查询: {query[:50]}...")
        
        # 保存到上下文
        await ctx.store.set("original_query", query)
        await ctx.store.set("history", history)
        
        return QueryReceivedEvent(query=query, history=history)
    
    @step
    async def rewrite_query(self, ctx: Context, ev: QueryReceivedEvent) -> QueryRewrittenEvent:
        """步骤2：查询改写"""
        query = ev.query
        history = ev.history
        
        # 获取历史查询（用于追问处理）
        previous_query = None
        if history and len(history) >= 2:
            for msg in reversed(history[:-1]):
                if msg.get("role") == "user":
                    previous_query = msg.get("content")
                    break
        
        # 执行改写
        result = self.query_rewriter.rewrite(query, previous_query)
        
        logger.info(f"[Workflow] 查询改写: {query} -> {result.rewritten_query}")
        
        # 保存改写结果
        await ctx.store.set("rewritten_query", result.rewritten_query)
        
        return QueryRewrittenEvent(
            original_query=query,
            rewritten_query=result.rewritten_query,
            rewrite_info={
                "was_rewritten": result.was_rewritten,
                "standardized_terms": result.standardized_terms,
            }
        )
    
    @step
    async def route_query(self, ctx: Context, ev: QueryRewrittenEvent) -> RouteDecidedEvent | DirectAnswerEvent:
        """步骤3：路由决策"""
        query = ev.rewritten_query
        
        # 执行路由
        decision = self.query_router.route(query)
        
        logger.info(f"[Workflow] 路由决策: {decision.query_type.value}, 置信度: {decision.confidence:.2f}")
        
        # 保存路由结果
        await ctx.store.set("route_decision", decision)
        
        # 如果是直接回答类型
        if decision.query_type == QueryType.DIRECT:
            return DirectAnswerEvent(query=query)
        
        return RouteDecidedEvent(
            query=query,
            query_type=decision.query_type,
            route_info={
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
        )
    
    @step
    async def retrieve(self, ctx: Context, ev: RouteDecidedEvent | RefinementNeededEvent) -> RetrievalResultEvent:
        """步骤4：执行检索"""
        query = ev.query
        
        if isinstance(ev, RouteDecidedEvent):
            query_type = ev.query_type
            attempt = 1
        else:
            # 优化检索时使用混合模式
            query_type = QueryType.HYBRID
            attempt = ev.attempt
        
        logger.info(f"[Workflow] 执行检索: {query_type.value}, 尝试 #{attempt}")
        
        documents = []
        retrieval_method = query_type.value
        
        if self.retriever:
            try:
                # 根据路由结果选择检索方法
                if query_type == QueryType.VECTOR:
                    documents = self.retriever.search(
                        query=query,
                        search_type="semantic",
                        top_k=10,
                    )
                elif query_type == QueryType.BM25:
                    documents = self.retriever.search(
                        query=query,
                        search_type="keyword",
                        top_k=10,
                    )
                else:  # HYBRID
                    documents = self.retriever.search(
                        query=query,
                        search_type="hybrid",
                        top_k=10,
                    )
            except Exception as e:
                logger.error(f"检索失败: {e}")
                documents = []
        
        logger.info(f"[Workflow] 检索到 {len(documents)} 个文档")
        
        return RetrievalResultEvent(
            query=query,
            documents=documents,
            retrieval_method=retrieval_method,
        )
    
    @step
    async def evaluate_results(self, ctx: Context, ev: RetrievalResultEvent) -> EvaluationEvent:
        """步骤5：评估检索结果"""
        # 获取当前尝试次数
        current_attempt = await ctx.store.get("retrieval_attempt", 1) or 1
        await ctx.store.set("retrieval_attempt", current_attempt)
        
        return EvaluationEvent(
            query=ev.query,
            documents=ev.documents,
            attempt=current_attempt,
        )
    
    @step
    async def check_evaluation(self, ctx: Context, ev: EvaluationEvent) -> GenerationEvent | RefinementNeededEvent:
        """步骤6：检查评估结果，决定是否优化"""
        query = ev.query
        documents = ev.documents
        attempt = ev.attempt
        
        # 评估结果质量
        evaluation = self.result_evaluator.evaluate(query, documents)
        
        logger.info(f"[Workflow] 评估结果: {evaluation.decision.value}, 分数: {evaluation.relevance_score:.2f}")
        
        # 如果结果足够好或已达到最大尝试次数
        if evaluation.decision == EvaluationDecision.SUFFICIENT or attempt >= self.max_refinement_attempts:
            return GenerationEvent(query=query, documents=documents)
        
        # 需要优化
        if evaluation.decision in [EvaluationDecision.PARTIAL, EvaluationDecision.INSUFFICIENT]:
            # 更新尝试次数
            await ctx.store.set("retrieval_attempt", attempt + 1)
            
            logger.info(f"[Workflow] 需要优化检索，尝试 #{attempt + 1}")
            
            return RefinementNeededEvent(
                query=query,
                original_documents=documents,
                feedback=evaluation.suggestions[0] if evaluation.suggestions else "扩展搜索范围",
                attempt=attempt + 1,
            )
        
        # 默认继续生成
        return GenerationEvent(query=query, documents=documents)
    
    @step
    async def generate_answer(self, ctx: Context, ev: GenerationEvent | DirectAnswerEvent) -> StopEvent:
        """步骤7：生成回答"""
        if isinstance(ev, DirectAnswerEvent):
            query = ev.query
            documents = []
        else:
            query = ev.query
            documents = ev.documents
        
        logger.info(f"[Workflow] 生成回答，基于 {len(documents)} 个文档")
        
        answer = ""
        
        if self.generator:
            try:
                if documents:
                    # 提取文档内容
                    context_texts = []
                    for doc in documents:
                        if hasattr(doc, "text"):
                            context_texts.append(doc.text)
                        elif hasattr(doc, "content"):
                            context_texts.append(doc.content)
                        elif isinstance(doc, dict):
                            context_texts.append(doc.get("text", doc.get("content", str(doc))))
                        else:
                            context_texts.append(str(doc))
                    
                    answer = self.generator.generate(
                        query=query,
                        contexts=context_texts,
                    )
                else:
                    # 直接回答
                    answer = self.generator.generate(
                        query=query,
                        contexts=[],
                    )
            except Exception as e:
                logger.error(f"生成失败: {e}")
                answer = f"抱歉，生成回答时出现错误：{str(e)}"
        else:
            # 无生成器时返回检索结果摘要
            if documents:
                answer = f"找到 {len(documents)} 个相关文档，但未配置生成器。"
            else:
                answer = "未找到相关文档，且未配置生成器。"
        
        # 获取工作流执行信息
        original_query = await ctx.store.get("original_query", query) or query
        rewritten_query = await ctx.store.get("rewritten_query", query) or query
        route_decision = await ctx.store.get("route_decision", None)
        
        result = {
            "answer": answer,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "documents": documents,
            "route_info": {
                "query_type": route_decision.query_type.value if route_decision else "unknown",
                "confidence": route_decision.confidence if route_decision else 0,
            } if route_decision else None,
        }
        
        return StopEvent(result=result)


# ============== 便捷函数 ==============

async def run_agentic_rag(
    query: str,
    retriever=None,
    generator=None,
    history: Optional[List[Dict]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    运行 Agentic RAG 工作流
    
    Args:
        query: 用户查询
        retriever: 检索器
        generator: 生成器
        history: 对话历史
        
    Returns:
        包含回答和元信息的字典
    """
    workflow = AgenticRAGWorkflow(
        retriever=retriever,
        generator=generator,
        **kwargs,
    )
    
    result = await workflow.run(
        query=query,
        history=history or [],
    )
    
    return result
