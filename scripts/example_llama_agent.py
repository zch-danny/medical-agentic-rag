"""
LlamaIndex Agent 使用示例

展示如何使用适配器构建 Agentic RAG
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_retrieval():
    """
    示例 1: 基础检索

    使用 MedicalLlamaRetriever 进行检索
    """
    from src.adapters import MedicalLlamaRetriever
    from llama_index.core.schema import QueryBundle

    print("=" * 50)
    print("示例 1: 基础检索")
    print("=" * 50)

    # 创建检索器（延迟加载模型）
    retriever = MedicalLlamaRetriever(
        top_k=5,
        alpha=0.7,
        use_rerank=True,
        lazy_load=True,
    )

    # 检索
    query = QueryBundle(query_str="糖尿病的治疗方法有哪些？")
    results = retriever._retrieve(query)

    print(f"\n查询: {query.query_str}")
    print(f"检索到 {len(results)} 个结果:\n")

    for i, result in enumerate(results, 1):
        print(f"[{i}] 分数: {result.score:.4f}")
        print(f"    来源: {result.node.metadata.get('source', '未知')}")
        print(f"    内容: {result.node.text[:100]}...")
        print()


def example_tools_usage():
    """
    示例 2: 使用 Tools

    展示如何创建和使用 LlamaIndex Tools
    """
    from src.adapters import MedicalRetrieverTool, MedicalGeneratorTool

    print("=" * 50)
    print("示例 2: 使用 Tools")
    print("=" * 50)

    # 创建检索工具
    retriever_tool = MedicalRetrieverTool(
        top_k=5,
        alpha=0.7,
        lazy_load=True,
    )

    # 获取 LlamaIndex FunctionTool
    search_tool = retriever_tool.as_tool()
    print(f"\n工具名称: {search_tool.metadata.name}")
    print(f"工具描述: {search_tool.metadata.description}")

    # 调用工具
    result = retriever_tool.search("高血压的诊断标准")
    print(f"\n检索结果:\n{result[:500]}...")


def example_rag_query():
    """
    示例 3: 端到端 RAG 查询

    使用 create_rag_query_tool 进行完整的检索+生成
    """
    from src.adapters.llama_tools import create_rag_query_tool

    print("=" * 50)
    print("示例 3: 端到端 RAG 查询")
    print("=" * 50)

    # 创建端到端 RAG 工具
    rag_tool = create_rag_query_tool(
        retriever_config={
            "top_k": 5,
            "alpha": 0.7,
            "use_rerank": True,
        }
    )

    print(f"\n工具名称: {rag_tool.metadata.name}")
    print(f"工具描述: {rag_tool.metadata.description}")

    # 调用工具进行问答
    question = "2型糖尿病的一线治疗药物是什么？"
    print(f"\n问题: {question}")

    answer = rag_tool.call(question)
    print(f"\n答案:\n{answer}")


def example_with_llama_agent():
    """
    示例 4: 与 LlamaIndex Agent 集成

    展示如何将工具与 LlamaIndex ReActAgent 集成
    """
    try:
        from llama_index.core.agent import ReActAgent
        from llama_index.llms.openai import OpenAI
    except ImportError:
        print("需要安装 llama-index-llms-openai: pip install llama-index-llms-openai")
        return

    from src.adapters import create_medical_tools

    print("=" * 50)
    print("示例 4: LlamaIndex Agent 集成")
    print("=" * 50)

    # 创建工具
    retriever_tool, generator_tool = create_medical_tools(
        retriever_config={"top_k": 5, "lazy_load": True}
    )

    # 创建 LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    # 创建 Agent
    agent = ReActAgent.from_tools(
        tools=[retriever_tool, generator_tool],
        llm=llm,
        verbose=True,
    )

    # 与 Agent 对话
    response = agent.chat("请帮我查询糖尿病的常见并发症有哪些？")
    print(f"\nAgent 响应:\n{response}")


def example_custom_workflow():
    """
    示例 5: 自定义工作流

    展示如何组合多个工具实现自定义逻辑
    """
    from src.adapters import MedicalRetrieverTool, MedicalGeneratorTool

    print("=" * 50)
    print("示例 5: 自定义工作流")
    print("=" * 50)

    retriever = MedicalRetrieverTool(top_k=10, lazy_load=True)
    generator = MedicalGeneratorTool()

    def custom_rag_workflow(question: str, min_score: float = 0.5) -> str:
        """
        自定义 RAG 工作流：
        1. 检索文档
        2. 过滤低分结果
        3. 如果结果不足，降低阈值重试
        4. 生成答案
        """
        # 第一次检索
        results = retriever.search_raw(question)

        # 过滤低分结果
        filtered = [r for r in results if retriever._get_score(r) >= min_score]

        # 如果结果不足，降低阈值
        if len(filtered) < 3 and min_score > 0.3:
            print(f"结果不足，降低阈值重试...")
            filtered = [r for r in results if retriever._get_score(r) >= min_score - 0.2]

        if not filtered:
            return "未找到足够相关的文献来回答您的问题。"

        # 生成答案
        answer = generator.generate_from_docs(question, filtered)
        return answer

    question = "高血压患者的运动建议有哪些？"
    print(f"\n问题: {question}")

    answer = custom_rag_workflow(question)
    print(f"\n答案:\n{answer}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LlamaIndex 适配器使用示例")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="选择要运行的示例 (1-5)",
    )

    args = parser.parse_args()

    examples = {
        1: example_basic_retrieval,
        2: example_tools_usage,
        3: example_rag_query,
        4: example_with_llama_agent,
        5: example_custom_workflow,
    }

    examples[args.example]()
