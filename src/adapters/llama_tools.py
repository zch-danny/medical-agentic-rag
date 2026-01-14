"""
LlamaIndex Tools 封装

将 MedicalRetriever 和 AnswerGenerator 封装为 LlamaIndex 可用的 Tools
供 Agent 调用
"""

from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.tools import FunctionTool, ToolMetadata
from loguru import logger


class MedicalRetrieverTool:
    """
    医疗文献检索工具

    封装现有的 MedicalRetriever，供 LlamaIndex Agent 调用
    """

    def __init__(
        self,
        retriever=None,
        top_k: int = 10,
        alpha: float = 0.7,
        use_rerank: bool = True,
        lazy_load: bool = True,
    ):
        """
        Args:
            retriever: 已有的 MedicalRetriever 实例
            top_k: 默认返回数量
            alpha: 混合检索权重
            use_rerank: 是否使用重排序
            lazy_load: 是否延迟加载
        """
        self._retriever = retriever
        self._top_k = top_k
        self._alpha = alpha
        self._use_rerank = use_rerank
        self._lazy_load = lazy_load

    def _init_retriever(self):
        """初始化 Retriever"""
        if self._retriever is not None:
            return

        from config import settings as cfg
        from src.retriever import MedicalRetriever

        logger.info("初始化 MedicalRetriever...")
        self._retriever = MedicalRetriever(
            config=cfg,
            lazy_load=self._lazy_load,
        )

    @property
    def retriever(self):
        if self._retriever is None:
            self._init_retriever()
        return self._retriever

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> str:
        """
        搜索医疗文献

        Args:
            query: 搜索查询，例如"糖尿病的治疗方法"
            top_k: 返回结果数量，默认10
            alpha: 向量检索权重(0-1)，越大越依赖语义，默认0.7

        Returns:
            格式化的检索结果字符串
        """
        results = self.retriever.search(
            query=query,
            top_k=top_k or self._top_k,
            alpha=alpha or self._alpha,
            use_rerank=self._use_rerank,
        )

        if not results:
            return "未找到相关医疗文献。"

        # 格式化结果
        formatted = []
        for i, result in enumerate(results, 1):
            entity = result.get("entity", result)
            text = entity.get("original_text") or entity.get("text", "")
            source = entity.get("source", "未知来源")
            title = entity.get("title", "")
            score = self._get_score(result)

            header = f"[文献{i}]"
            if title:
                header += f" {title}"
            header += f" (来源: {source}, 相关度: {score:.4f})"

            formatted.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(formatted)

    def search_raw(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索医疗文献，返回原始结果

        Args:
            query: 搜索查询
            top_k: 返回数量
            alpha: 向量检索权重

        Returns:
            原始检索结果列表
        """
        return self.retriever.search(
            query=query,
            top_k=top_k or self._top_k,
            alpha=alpha or self._alpha,
            use_rerank=self._use_rerank,
        )

    def _get_score(self, result: Dict) -> float:
        """提取分数"""
        score = result.get("rerank_score") or result.get("score")
        if score is not None:
            return float(score)
        distance = result.get("distance")
        if distance is not None:
            return -float(distance)
        return 0.0

    def as_tool(self) -> FunctionTool:
        """
        转换为 LlamaIndex FunctionTool

        Returns:
            可被 Agent 调用的 FunctionTool
        """
        return FunctionTool.from_defaults(
            fn=self.search,
            name="medical_literature_search",
            description=(
                "搜索医疗文献数据库。输入医疗相关问题或关键词，"
                "返回相关的医学文献摘要。适用于查询疾病、症状、治疗方法、"
                "药物信息、临床指南等医疗知识。"
            ),
        )


class MedicalGeneratorTool:
    """
    医疗答案生成工具

    封装现有的 AnswerGenerator，供 LlamaIndex Agent 调用
    """

    def __init__(self, generator=None):
        """
        Args:
            generator: 已有的 AnswerGenerator 实例
        """
        self._generator = generator

    def _init_generator(self):
        """初始化 Generator"""
        if self._generator is not None:
            return

        from src.generator import AnswerGenerator

        logger.info("初始化 AnswerGenerator...")
        self._generator = AnswerGenerator()

    @property
    def generator(self):
        if self._generator is None:
            self._init_generator()
        return self._generator

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        基于检索到的文献生成答案

        Args:
            question: 用户问题
            context: 检索到的医疗文献内容

        Returns:
            生成的医疗答案
        """
        # 将 context 字符串转换为 documents 格式
        documents = [{"entity": {"original_text": context, "source": "检索结果"}}]

        try:
            answer = self.generator.generate_sync(question, documents)
            return answer
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def generate_from_docs(
        self,
        question: str,
        documents: List[Dict[str, Any]],
    ) -> str:
        """
        基于文档列表生成答案

        Args:
            question: 用户问题
            documents: 检索结果文档列表

        Returns:
            生成的医疗答案
        """
        try:
            answer = self.generator.generate_sync(question, documents)
            return answer
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def as_tool(self) -> FunctionTool:
        """
        转换为 LlamaIndex FunctionTool

        Returns:
            可被 Agent 调用的 FunctionTool
        """
        return FunctionTool.from_defaults(
            fn=self.generate,
            name="medical_answer_generator",
            description=(
                "基于医疗文献生成专业答案。输入用户问题和检索到的文献内容，"
                "生成专业、准确的医疗回答。必须在使用 medical_literature_search "
                "获取相关文献后才能使用此工具。"
            ),
        )


def create_medical_tools(
    retriever=None,
    generator=None,
    retriever_config: Optional[Dict[str, Any]] = None,
) -> Tuple[FunctionTool, FunctionTool]:
    """
    创建医疗 RAG 所需的工具集

    Args:
        retriever: 已有的 MedicalRetriever 实例
        generator: 已有的 AnswerGenerator 实例
        retriever_config: 检索器配置（top_k, alpha, use_rerank）

    Returns:
        (retriever_tool, generator_tool) 元组
    """
    retriever_config = retriever_config or {}

    retriever_tool = MedicalRetrieverTool(
        retriever=retriever,
        top_k=retriever_config.get("top_k", 10),
        alpha=retriever_config.get("alpha", 0.7),
        use_rerank=retriever_config.get("use_rerank", True),
        lazy_load=retriever_config.get("lazy_load", True),
    )

    generator_tool = MedicalGeneratorTool(generator=generator)

    return retriever_tool.as_tool(), generator_tool.as_tool()


def create_rag_query_tool(
    retriever=None,
    generator=None,
    retriever_config: Optional[Dict[str, Any]] = None,
) -> FunctionTool:
    """
    创建一个端到端的 RAG 查询工具

    将检索和生成合并为单一工具，简化 Agent 调用

    Args:
        retriever: MedicalRetriever 实例
        generator: AnswerGenerator 实例
        retriever_config: 检索器配置

    Returns:
        端到端的 RAG 工具
    """
    retriever_config = retriever_config or {}

    retriever_tool = MedicalRetrieverTool(
        retriever=retriever,
        top_k=retriever_config.get("top_k", 10),
        alpha=retriever_config.get("alpha", 0.7),
        use_rerank=retriever_config.get("use_rerank", True),
        lazy_load=retriever_config.get("lazy_load", True),
    )

    generator_tool = MedicalGeneratorTool(generator=generator)

    def rag_query(question: str) -> str:
        """
        查询医疗文献并生成答案

        Args:
            question: 医疗相关问题

        Returns:
            基于文献的专业答案
        """
        # 1. 检索
        documents = retriever_tool.search_raw(question)

        if not documents:
            return "未找到相关医疗文献，无法回答该问题。请尝试更换关键词或咨询专业医生。"

        # 2. 生成答案
        answer = generator_tool.generate_from_docs(question, documents)

        return answer

    return FunctionTool.from_defaults(
        fn=rag_query,
        name="medical_rag_query",
        description=(
            "医疗知识问答工具。输入医疗相关问题，自动检索相关文献并生成专业答案。"
            "适用于疾病诊断、治疗方案、药物信息、临床指南等医疗问题。"
            "注意：生成的答案仅供参考，具体诊疗请咨询专业医生。"
        ),
    )
