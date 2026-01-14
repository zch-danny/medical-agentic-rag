"""
LlamaIndex 适配器测试

测试 adapters 模块的功能
"""

import pytest
from unittest.mock import MagicMock, patch


class TestMedicalLlamaRetriever:
    """测试 MedicalLlamaRetriever"""

    @pytest.fixture
    def mock_retriever(self):
        """模拟 MedicalRetriever"""
        mock = MagicMock()
        mock.search.return_value = [
            {
                "entity": {
                    "original_text": "糖尿病是一种代谢疾病...",
                    "text": "糖尿 病 是 一种 代谢 疾病",
                    "source": "医学教科书.pdf",
                    "path": "/data/医学教科书.pdf",
                    "title": "内科学",
                    "year": "2023",
                    "doi": "10.1234/example",
                    "keywords": '["糖尿病", "代谢"]',
                    "chunk_index": 0,
                },
                "rerank_score": 0.95,
            },
            {
                "entity": {
                    "original_text": "2型糖尿病的治疗方法包括...",
                    "text": "2 型 糖尿 病 的 治疗 方法",
                    "source": "临床指南.pdf",
                    "path": "/data/临床指南.pdf",
                    "title": "糖尿病诊疗指南",
                    "year": "2024",
                    "doi": "",
                    "keywords": '["治疗", "指南"]',
                    "chunk_index": 1,
                },
                "rerank_score": 0.88,
            },
        ]
        return mock

    def test_retrieve_returns_nodes_with_scores(self, mock_retriever):
        """测试检索返回 NodeWithScore 列表"""
        from src.adapters.llama_retriever import MedicalLlamaRetriever
        from llama_index.core.schema import QueryBundle

        retriever = MedicalLlamaRetriever(retriever=mock_retriever)
        query = QueryBundle(query_str="糖尿病的治疗方法")

        results = retriever._retrieve(query)

        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.88
        assert "糖尿病是一种代谢疾病" in results[0].node.text

    def test_retrieve_with_details(self, mock_retriever):
        """测试 retrieve_with_details 返回原始格式"""
        from src.adapters.llama_retriever import MedicalLlamaRetriever

        retriever = MedicalLlamaRetriever(retriever=mock_retriever)
        results = retriever.retrieve_with_details("糖尿病")

        assert len(results) == 2
        assert "entity" in results[0]

    def test_metadata_extraction(self, mock_retriever):
        """测试元数据正确提取"""
        from src.adapters.llama_retriever import MedicalLlamaRetriever
        from llama_index.core.schema import QueryBundle

        retriever = MedicalLlamaRetriever(retriever=mock_retriever)
        query = QueryBundle(query_str="test")

        results = retriever._retrieve(query)

        metadata = results[0].node.metadata
        assert metadata["source"] == "医学教科书.pdf"
        assert metadata["title"] == "内科学"
        assert metadata["year"] == "2023"


class TestMedicalRetrieverTool:
    """测试 MedicalRetrieverTool"""

    @pytest.fixture
    def mock_retriever(self):
        """模拟 MedicalRetriever"""
        mock = MagicMock()
        mock.search.return_value = [
            {
                "entity": {
                    "original_text": "高血压的诊断标准...",
                    "source": "诊断标准.pdf",
                    "title": "高血压诊疗指南",
                },
                "rerank_score": 0.92,
            },
        ]
        return mock

    def test_search_returns_formatted_string(self, mock_retriever):
        """测试 search 返回格式化字符串"""
        from src.adapters.llama_tools import MedicalRetrieverTool

        tool = MedicalRetrieverTool(retriever=mock_retriever)
        result = tool.search("高血压诊断")

        assert "[文献1]" in result
        assert "高血压诊疗指南" in result
        assert "0.9200" in result

    def test_search_raw_returns_list(self, mock_retriever):
        """测试 search_raw 返回原始列表"""
        from src.adapters.llama_tools import MedicalRetrieverTool

        tool = MedicalRetrieverTool(retriever=mock_retriever)
        result = tool.search_raw("高血压")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_as_tool_returns_function_tool(self, mock_retriever):
        """测试 as_tool 返回 FunctionTool"""
        from src.adapters.llama_tools import MedicalRetrieverTool
        from llama_index.core.tools import FunctionTool

        tool = MedicalRetrieverTool(retriever=mock_retriever)
        fn_tool = tool.as_tool()

        assert isinstance(fn_tool, FunctionTool)
        assert fn_tool.metadata.name == "medical_literature_search"

    def test_empty_results(self, mock_retriever):
        """测试空结果处理"""
        from src.adapters.llama_tools import MedicalRetrieverTool

        mock_retriever.search.return_value = []
        tool = MedicalRetrieverTool(retriever=mock_retriever)
        result = tool.search("不存在的查询")

        assert "未找到" in result


class TestMedicalGeneratorTool:
    """测试 MedicalGeneratorTool"""

    @pytest.fixture
    def mock_generator(self):
        """模拟 AnswerGenerator"""
        mock = MagicMock()
        mock.generate_sync.return_value = "糖尿病的治疗包括饮食控制、运动和药物治疗..."
        return mock

    def test_generate_returns_string(self, mock_generator):
        """测试 generate 返回字符串"""
        from src.adapters.llama_tools import MedicalGeneratorTool

        tool = MedicalGeneratorTool(generator=mock_generator)
        result = tool.generate("糖尿病怎么治疗？", "相关文献内容...")

        assert "糖尿病的治疗" in result

    def test_generate_from_docs(self, mock_generator):
        """测试从文档列表生成"""
        from src.adapters.llama_tools import MedicalGeneratorTool

        tool = MedicalGeneratorTool(generator=mock_generator)
        docs = [{"entity": {"original_text": "文献1内容"}}]
        result = tool.generate_from_docs("问题", docs)

        assert "糖尿病的治疗" in result
        mock_generator.generate_sync.assert_called_once()

    def test_as_tool_returns_function_tool(self, mock_generator):
        """测试 as_tool 返回 FunctionTool"""
        from src.adapters.llama_tools import MedicalGeneratorTool
        from llama_index.core.tools import FunctionTool

        tool = MedicalGeneratorTool(generator=mock_generator)
        fn_tool = tool.as_tool()

        assert isinstance(fn_tool, FunctionTool)
        assert fn_tool.metadata.name == "medical_answer_generator"


class TestCreateMedicalTools:
    """测试 create_medical_tools 工厂函数"""

    def test_returns_tuple_of_tools(self):
        """测试返回工具元组"""
        from src.adapters.llama_tools import create_medical_tools
        from llama_index.core.tools import FunctionTool

        mock_retriever = MagicMock()
        mock_generator = MagicMock()

        retriever_tool, generator_tool = create_medical_tools(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        assert isinstance(retriever_tool, FunctionTool)
        assert isinstance(generator_tool, FunctionTool)


class TestCreateRagQueryTool:
    """测试端到端 RAG 工具"""

    def test_rag_query_tool(self):
        """测试端到端 RAG 查询"""
        from src.adapters.llama_tools import create_rag_query_tool

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"entity": {"original_text": "相关内容"}, "rerank_score": 0.9}
        ]

        mock_generator = MagicMock()
        mock_generator.generate_sync.return_value = "这是生成的答案"

        tool = create_rag_query_tool(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        result = tool.call("测试问题")
        assert "这是生成的答案" in str(result)


# 集成测试（需要实际环境）
class TestIntegration:
    """集成测试 - 需要 Milvus 和模型"""

    @pytest.mark.skip(reason="需要实际环境")
    def test_full_retrieval_flow(self):
        """完整检索流程测试"""
        from src.adapters import MedicalLlamaRetriever
        from llama_index.core.schema import QueryBundle

        retriever = MedicalLlamaRetriever(lazy_load=False)
        query = QueryBundle(query_str="高血压的治疗方法")

        results = retriever._retrieve(query)

        assert len(results) > 0
        for result in results:
            assert result.node.text
            assert result.score >= 0
