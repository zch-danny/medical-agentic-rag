"""
医疗文献 Agentic RAG Agent

整合所有组件的主类，提供统一的接口
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger

# Agent 组件
from .memory import ConversationMemory, MemoryConfig
from .workflow import AgenticRAGWorkflow, run_agentic_rag

# Agentic 组件（使用相对导入）
from ..agentic.query_router import QueryRouter
from ..agentic.query_rewriter import QueryRewriter
from ..agentic.result_evaluator import ResultEvaluator


@dataclass
class AgentConfig:
    """Agent 配置"""
    # 记忆配置
    enable_memory: bool = True
    max_history_length: int = 20
    enable_persistence: bool = False
    session_storage_path: Optional[str] = None
    
    # 工作流配置
    max_refinement_attempts: int = 2
    
    # 检索配置
    default_top_k: int = 10
    
    # 日志配置
    verbose: bool = True


@dataclass
class AgentResponse:
    """Agent 响应"""
    answer: str                             # 生成的回答
    query: str                              # 原始查询
    rewritten_query: Optional[str] = None   # 改写后的查询
    documents: List[Any] = field(default_factory=list)  # 检索到的文档
    route_info: Optional[Dict] = None       # 路由信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元信息
    
    @property
    def success(self) -> bool:
        """是否成功生成回答"""
        return bool(self.answer)
    
    @property
    def has_sources(self) -> bool:
        """是否有来源文档"""
        return len(self.documents) > 0


class MedicalAgent:
    """
    医疗文献 Agentic RAG Agent
    
    整合查询改写、智能路由、检索、评估和生成的完整流程
    
    示例:
        ```python
        # 创建 Agent
        agent = MedicalAgent(
            retriever=my_retriever,
            generator=my_generator,
        )
        
        # 单次查询
        response = agent.query("糖尿病的症状有哪些？")
        print(response.answer)
        
        # 多轮对话
        agent.query("糖尿病的症状有哪些？")
        agent.query("如何预防？")  # 自动关联上下文
        
        # 异步调用
        response = await agent.aquery("高血压的治疗方法")
        ```
    """
    
    def __init__(
        self,
        retriever=None,
        generator=None,
        config: Optional[AgentConfig] = None,
        query_router: Optional[QueryRouter] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        result_evaluator: Optional[ResultEvaluator] = None,
        session_id: Optional[str] = None,
    ):
        """
        Args:
            retriever: 检索器实例（MedicalRetriever 或兼容接口）
            generator: 生成器实例（AnswerGenerator 或兼容接口）
            config: Agent 配置
            query_router: 查询路由器（可选，默认创建新实例）
            query_rewriter: 查询改写器（可选，默认创建新实例）
            result_evaluator: 结果评估器（可选，默认创建新实例）
            session_id: 会话ID（用于持久化）
        """
        self.config = config or AgentConfig()
        self._session_id = session_id  # 保存 session_id
        
        # 核心组件
        self.retriever = retriever
        self.generator = generator
        
        # Agentic 组件
        self.query_router = query_router or QueryRouter()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.result_evaluator = result_evaluator or ResultEvaluator()
        
        # 对话记忆
        self._memory: Optional[ConversationMemory] = None
        if self.config.enable_memory:
            memory_config = MemoryConfig(
                max_history_length=self.config.max_history_length,
                enable_persistence=self.config.enable_persistence,
                storage_path=Path(self.config.session_storage_path) if self.config.session_storage_path else None,
            )
            self._memory = ConversationMemory(
                config=memory_config,
                session_id=session_id,
            )
            # 从 memory 获取实际的 session_id（如果未提供，memory 会生成一个）
            self._session_id = self._memory.session_id
        
        # 工作流
        self._workflow = AgenticRAGWorkflow(
            retriever=retriever,
            generator=generator,
            query_router=self.query_router,
            query_rewriter=self.query_rewriter,
            result_evaluator=self.result_evaluator,
            max_refinement_attempts=self.config.max_refinement_attempts,
        )
        
        if self.config.verbose:
            logger.info("MedicalAgent 初始化完成")
    
    def query(self, question: str, **kwargs) -> AgentResponse:
        """
        同步查询接口
        
        Args:
            question: 用户问题
            **kwargs: 额外参数
            
        Returns:
            AgentResponse 响应对象
        """
        return asyncio.run(self.aquery(question, **kwargs))
    
    async def aquery(self, question: str, **kwargs) -> AgentResponse:
        """
        异步查询接口
        
        Args:
            question: 用户问题
            **kwargs: 额外参数
            
        Returns:
            AgentResponse 响应对象
        """
        if self.config.verbose:
            logger.info(f"[Agent] 收到查询: {question[:50]}...")
        
        # 获取对话历史
        history = []
        if self._memory:
            history = self._memory.get_history()
            self._memory.add_user_message(question)
        
        try:
            # 运行工作流
            result = await self._workflow.run(
                query=question,
                history=history,
            )
            
            # 构建响应
            response = AgentResponse(
                answer=result.get("answer", ""),
                query=result.get("original_query", question),
                rewritten_query=result.get("rewritten_query"),
                documents=result.get("documents", []),
                route_info=result.get("route_info"),
                metadata={
                    "history_length": len(history),
                },
            )
            
            # 保存助手回复到记忆
            if self._memory and response.answer:
                self._memory.add_assistant_message(
                    response.answer,
                    metadata={"route_info": response.route_info},
                )
            
            if self.config.verbose:
                logger.info(f"[Agent] 生成回答: {response.answer[:50]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"[Agent] 查询失败: {e}")
            return AgentResponse(
                answer=f"抱歉，处理您的问题时出现错误：{str(e)}",
                query=question,
                metadata={"error": str(e)},
            )
    
    def chat(self, message: str) -> str:
        """
        简化的对话接口，直接返回回答文本
        
        Args:
            message: 用户消息
            
        Returns:
            回答文本
        """
        response = self.query(message)
        return response.answer
    
    async def achat(self, message: str) -> str:
        """
        异步简化对话接口
        
        Args:
            message: 用户消息
            
        Returns:
            回答文本
        """
        response = await self.aquery(message)
        return response.answer
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            max_messages: 最大返回消息数
            
        Returns:
            消息列表
        """
        if self._memory:
            return self._memory.get_history(max_messages)
        return []
    
    def clear_history(self) -> None:
        """清除对话历史"""
        if self._memory:
            self._memory.clear_history()
            logger.info("[Agent] 对话历史已清除")
    
    def set_context(self, key: str, value: Any) -> None:
        """设置上下文变量"""
        if self._memory:
            self._memory.set_context(key, value)
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文变量"""
        if self._memory:
            return self._memory.get_context(key, default)
        return default
    
    @property
    def session_id(self) -> Optional[str]:
        """获取当前会话ID"""
        return self._session_id
    
    @property
    def history_length(self) -> int:
        """获取历史消息数量"""
        if self._memory:
            return len(self._memory)
        return 0
    
    def __repr__(self) -> str:
        return f"MedicalAgent(session={self.session_id}, history={self.history_length})"


# ============== 便捷函数 ==============

def create_agent(
    retriever=None,
    generator=None,
    enable_memory: bool = True,
    verbose: bool = True,
    **kwargs,
) -> MedicalAgent:
    """
    创建 MedicalAgent 实例的便捷函数
    
    Args:
        retriever: 检索器
        generator: 生成器
        enable_memory: 是否启用记忆
        verbose: 是否启用详细日志
        **kwargs: 其他配置参数
        
    Returns:
        MedicalAgent 实例
    """
    config = AgentConfig(
        enable_memory=enable_memory,
        verbose=verbose,
        **kwargs,
    )
    
    return MedicalAgent(
        retriever=retriever,
        generator=generator,
        config=config,
    )


async def quick_query(
    question: str,
    retriever=None,
    generator=None,
) -> str:
    """
    快速查询函数，无状态
    
    Args:
        question: 问题
        retriever: 检索器
        generator: 生成器
        
    Returns:
        回答文本
    """
    result = await run_agentic_rag(
        query=question,
        retriever=retriever,
        generator=generator,
    )
    return result.get("answer", "")
