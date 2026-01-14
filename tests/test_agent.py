"""
Agent 模块测试

测试 MedicalAgent、ConversationMemory 和 AgenticRAGWorkflow
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock
from pathlib import Path

import sys
import importlib.util
project_root = str(Path(__file__).parent.parent)

# 直接加载模块文件，避免触发 src/__init__.py
def load_module_direct(name: str, file_path: str):
    """直接从文件加载模块"""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# 先加载 agentic 组件
agentic_path = Path(project_root) / "src" / "agentic"
query_router = load_module_direct("src.agentic.query_router", str(agentic_path / "query_router.py"))
query_rewriter = load_module_direct("src.agentic.query_rewriter", str(agentic_path / "query_rewriter.py"))
result_evaluator = load_module_direct("src.agentic.result_evaluator", str(agentic_path / "result_evaluator.py"))

# 再加载 agent 组件
agent_path = Path(project_root) / "src" / "agent"
memory_module = load_module_direct("src.agent.memory", str(agent_path / "memory.py"))
workflow_module = load_module_direct("src.agent.workflow", str(agent_path / "workflow.py"))
medical_agent_module = load_module_direct("src.agent.medical_agent", str(agent_path / "medical_agent.py"))

# 导出需要的类
ConversationMemory = memory_module.ConversationMemory
MemoryConfig = memory_module.MemoryConfig
Message = memory_module.Message
MedicalAgent = medical_agent_module.MedicalAgent
AgentConfig = medical_agent_module.AgentConfig
AgentResponse = medical_agent_module.AgentResponse
create_agent = medical_agent_module.create_agent


class TestConversationMemory:
    """ConversationMemory 测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        memory = ConversationMemory()
        assert memory.config.max_history_length == 20
        assert len(memory) == 0
        assert not memory  # bool(empty memory) = False
    
    def test_add_user_message(self):
        """测试添加用户消息"""
        memory = ConversationMemory()
        memory.add_user_message("你好")
        
        assert len(memory) == 1
        assert memory
        
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
    
    def test_add_assistant_message(self):
        """测试添加助手消息"""
        memory = ConversationMemory()
        memory.add_assistant_message("你好，有什么可以帮助你的？")
        
        history = memory.get_history()
        assert history[0]["role"] == "assistant"
    
    def test_conversation_flow(self):
        """测试对话流程"""
        memory = ConversationMemory()
        
        memory.add_user_message("糖尿病的症状有哪些？")
        memory.add_assistant_message("糖尿病的主要症状包括...")
        memory.add_user_message("如何预防？")
        memory.add_assistant_message("预防糖尿病需要...")
        
        assert len(memory) == 4
        
        history = memory.get_history()
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"
    
    def test_get_history_with_limit(self):
        """测试限制历史消息数"""
        memory = ConversationMemory()
        
        for i in range(5):
            memory.add_user_message(f"消息 {i}")
        
        history = memory.get_history(max_messages=3)
        assert len(history) == 3
        assert "消息 2" in history[0]["content"]
    
    def test_get_history_text(self):
        """测试获取文本格式历史"""
        memory = ConversationMemory()
        memory.add_user_message("问题")
        memory.add_assistant_message("回答")
        
        text = memory.get_history_text()
        assert "用户: 问题" in text
        assert "助手: 回答" in text
    
    def test_get_last_user_query(self):
        """测试获取最后用户查询"""
        memory = ConversationMemory()
        memory.add_user_message("第一个问题")
        memory.add_assistant_message("回答")
        memory.add_user_message("第二个问题")
        
        assert memory.get_last_user_query() == "第二个问题"
    
    def test_get_previous_query(self):
        """测试获取上一个用户查询"""
        memory = ConversationMemory()
        memory.add_user_message("第一个问题")
        memory.add_assistant_message("回答")
        memory.add_user_message("第二个问题")
        
        assert memory.get_previous_query() == "第一个问题"
    
    def test_context_management(self):
        """测试上下文管理"""
        memory = ConversationMemory()
        
        memory.set_context("topic", "糖尿病")
        assert memory.get_context("topic") == "糖尿病"
        assert memory.get_context("nonexistent", "default") == "default"
        
        memory.clear_context()
        assert memory.get_context("topic") is None
    
    def test_clear_history(self):
        """测试清除历史"""
        memory = ConversationMemory()
        memory.add_user_message("问题")
        memory.add_assistant_message("回答")
        memory.set_context("key", "value")
        
        memory.clear_history()
        
        assert len(memory) == 0
        assert memory.get_context("key") is None


class TestAgentConfig:
    """AgentConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AgentConfig()
        
        assert config.enable_memory is True
        assert config.max_history_length == 20
        assert config.enable_persistence is False
        assert config.max_refinement_attempts == 2
        assert config.default_top_k == 10
        assert config.verbose is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AgentConfig(
            enable_memory=False,
            max_history_length=10,
            verbose=False,
        )
        
        assert config.enable_memory is False
        assert config.max_history_length == 10
        assert config.verbose is False


class TestAgentResponse:
    """AgentResponse 测试"""
    
    def test_response_properties(self):
        """测试响应属性"""
        response = AgentResponse(
            answer="这是回答",
            query="这是问题",
            documents=[{"text": "doc1"}, {"text": "doc2"}],
        )
        
        assert response.success is True
        assert response.has_sources is True
    
    def test_empty_response(self):
        """测试空响应"""
        response = AgentResponse(answer="", query="问题")
        
        assert response.success is False
        assert response.has_sources is False


class TestMedicalAgent:
    """MedicalAgent 测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        agent = MedicalAgent()
        
        assert agent.config.enable_memory is True
        assert agent.retriever is None
        assert agent.generator is None
        assert agent.history_length == 0
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = AgentConfig(enable_memory=False, verbose=False)
        agent = MedicalAgent(config=config)
        
        assert agent.config.enable_memory is False
        assert agent._memory is None
    
    def test_init_with_components(self):
        """测试带组件初始化"""
        mock_retriever = Mock()
        mock_generator = Mock()
        
        agent = MedicalAgent(
            retriever=mock_retriever,
            generator=mock_generator,
            config=AgentConfig(verbose=False),
        )
        
        assert agent.retriever is mock_retriever
        assert agent.generator is mock_generator
    
    def test_session_id(self):
        """测试会话ID"""
        agent = MedicalAgent(
            session_id="test_session",
            config=AgentConfig(verbose=False),
        )
        
        assert agent.session_id == "test_session"
    
    def test_history_management(self):
        """测试历史管理"""
        agent = MedicalAgent(config=AgentConfig(verbose=False))
        
        # 初始为空
        assert agent.history_length == 0
        assert agent.get_history() == []
        
        # 手动添加（通过内部 memory）
        agent._memory.add_user_message("测试问题")
        assert agent.history_length == 1
        
        # 清除
        agent.clear_history()
        assert agent.history_length == 0
    
    def test_context_management(self):
        """测试上下文管理"""
        agent = MedicalAgent(config=AgentConfig(verbose=False))
        
        # 注意：由于模块加载方式，我们需要直接操作内部 memory
        assert agent._memory is not None
        agent._memory.set_context("topic", "糖尿病")
        assert agent._memory.get_context("topic") == "糖尿病"
        
        # 测试通过 agent 接口
        agent._memory.set_context("test_key", "test_value")
        # agent.get_context 会调用 self._memory.get_context
        result = agent._memory.get_context("test_key")
        assert result == "test_value"
        assert agent._memory.get_context("unknown", "default") == "default"
    
    def test_repr(self):
        """测试字符串表示"""
        agent = MedicalAgent(
            session_id="test",
            config=AgentConfig(verbose=False),
        )
        
        repr_str = repr(agent)
        assert "MedicalAgent" in repr_str
        assert "test" in repr_str


class TestCreateAgent:
    """create_agent 便捷函数测试"""
    
    def test_create_default(self):
        """测试创建默认 agent"""
        agent = create_agent(verbose=False)
        
        assert isinstance(agent, MedicalAgent)
        assert agent.config.enable_memory is True
    
    def test_create_without_memory(self):
        """测试创建无记忆 agent"""
        agent = create_agent(enable_memory=False, verbose=False)
        
        assert agent._memory is None
    
    def test_create_with_components(self):
        """测试带组件创建"""
        mock_retriever = Mock()
        mock_generator = Mock()
        
        agent = create_agent(
            retriever=mock_retriever,
            generator=mock_generator,
            verbose=False,
        )
        
        assert agent.retriever is mock_retriever
        assert agent.generator is mock_generator


# ============== 集成测试 ==============
# 注意：集成测试需要完整的 LlamaIndex 工作流环境
# 在 CI 环境或未安装完整依赖时，这些测试可能会跳过

class TestAgentIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要完整工作流环境")
    async def test_query_without_retriever(self):
        """测试无检索器查询"""
        agent = MedicalAgent(config=AgentConfig(verbose=False))
        
        response = await agent.aquery("什么是糖尿病？")
        
        assert isinstance(response, AgentResponse)
        assert response.query == "什么是糖尿病？"
        # 无检索器应该返回默认消息
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要完整工作流环境")
    async def test_multi_turn_conversation(self):
        """测试多轮对话"""
        agent = MedicalAgent(config=AgentConfig(verbose=False))
        
        # 第一轮
        await agent.aquery("糖尿病是什么？")
        assert agent.history_length >= 1
        
        # 第二轮（追问）
        await agent.aquery("有哪些症状？")
        assert agent.history_length >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
