"""
MedicalAgent 使用示例

展示如何使用 Agentic RAG Agent 进行医疗问答
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import MedicalAgent, AgentConfig, create_agent


def example_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("示例 1: 基础使用")
    print("=" * 50)
    
    # 创建 Agent（无实际检索器和生成器，仅演示流程）
    agent = create_agent(
        enable_memory=True,
        verbose=True,
    )
    
    # 单次查询
    response = agent.query("糖尿病的常见症状有哪些？")
    
    print(f"\n查询: {response.query}")
    print(f"改写后查询: {response.rewritten_query}")
    print(f"路由信息: {response.route_info}")
    print(f"回答: {response.answer}")
    print(f"是否成功: {response.success}")
    print(f"是否有来源: {response.has_sources}")


def example_multi_turn():
    """多轮对话示例"""
    print("\n" + "=" * 50)
    print("示例 2: 多轮对话")
    print("=" * 50)
    
    agent = MedicalAgent(
        config=AgentConfig(
            enable_memory=True,
            verbose=False,
        )
    )
    
    # 第一轮
    print("\n[用户] 高血压是什么？")
    response1 = agent.query("高血压是什么？")
    print(f"[助手] {response1.answer}")
    
    # 第二轮（追问，系统会自动关联上下文）
    print("\n[用户] 有什么危害？")
    response2 = agent.query("有什么危害？")
    print(f"[助手] {response2.answer}")
    print(f"  (改写后: {response2.rewritten_query})")
    
    # 第三轮
    print("\n[用户] 如何预防？")
    response3 = agent.query("如何预防？")
    print(f"[助手] {response3.answer}")
    
    # 查看对话历史
    print(f"\n对话历史长度: {agent.history_length}")
    print("历史消息:")
    for msg in agent.get_history():
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
        print(f"  - [{role}] {content}")


def example_with_context():
    """使用上下文示例"""
    print("\n" + "=" * 50)
    print("示例 3: 上下文管理")
    print("=" * 50)
    
    agent = create_agent(verbose=False)
    
    # 设置上下文（例如用户偏好）
    agent.set_context("user_age", 65)
    agent.set_context("medical_history", ["高血压", "糖尿病"])
    
    # 获取上下文
    print(f"用户年龄: {agent.get_context('user_age')}")
    print(f"病史: {agent.get_context('medical_history')}")
    
    # 这些上下文可以在实际实现中用于个性化回答
    response = agent.query("我应该注意什么？")
    print(f"\n查询: {response.query}")


async def example_async_usage():
    """异步使用示例"""
    print("\n" + "=" * 50)
    print("示例 4: 异步使用")
    print("=" * 50)
    
    agent = create_agent(verbose=False)
    
    # 异步查询
    response = await agent.aquery("什么是糖尿病？")
    print(f"异步查询结果: {response.answer[:50]}...")
    
    # 简化的异步对话接口
    answer = await agent.achat("有哪些症状？")
    print(f"异步对话结果: {answer[:50]}...")


def example_session_management():
    """会话管理示例"""
    print("\n" + "=" * 50)
    print("示例 5: 会话管理")
    print("=" * 50)
    
    # 创建带会话ID的 Agent
    agent = MedicalAgent(
        session_id="user_123_session_001",
        config=AgentConfig(
            enable_memory=True,
            enable_persistence=False,  # 设为 True 可持久化到文件
            verbose=False,
        )
    )
    
    print(f"会话ID: {agent.session_id}")
    
    # 进行对话
    agent.query("心脏病的预防方法")
    agent.query("需要做哪些检查？")
    
    print(f"当前历史长度: {agent.history_length}")
    
    # 清除历史
    agent.clear_history()
    print(f"清除后历史长度: {agent.history_length}")


def example_chat_interface():
    """简化对话接口示例"""
    print("\n" + "=" * 50)
    print("示例 6: 简化对话接口")
    print("=" * 50)
    
    agent = create_agent(verbose=False)
    
    # 使用 chat 方法直接获取回答文本
    answer1 = agent.chat("感冒了怎么办？")
    print(f"回答: {answer1}")
    
    answer2 = agent.chat("需要吃药吗？")
    print(f"追问回答: {answer2}")


def main():
    """运行所有示例"""
    print("Medical Agent 使用示例\n")
    
    # 基础示例
    example_basic_usage()
    
    # 多轮对话
    example_multi_turn()
    
    # 上下文管理
    example_with_context()
    
    # 异步使用
    asyncio.run(example_async_usage())
    
    # 会话管理
    example_session_management()
    
    # 简化对话接口
    example_chat_interface()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
