"""
LlamaIndex 适配器模块

将现有 medical_embedding 组件封装为 LlamaIndex 兼容的接口
"""

from .llama_retriever import MedicalLlamaRetriever
from .llama_tools import (
    MedicalRetrieverTool,
    MedicalGeneratorTool,
    create_medical_tools,
)

__all__ = [
    "MedicalLlamaRetriever",
    "MedicalRetrieverTool",
    "MedicalGeneratorTool",
    "create_medical_tools",
]
