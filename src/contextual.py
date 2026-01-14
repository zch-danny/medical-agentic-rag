"""
Contextual Retrieval 模块

基于 Anthropic 的 Contextual Retrieval 方法：
为每个文本块添加上下文说明，提升检索精度。

核心思想：
1. 对每个 chunk，将完整文档作为背景
2. 使用 LLM 生成简短的上下文描述
3. 将上下文与原始 chunk 拼接后再嵌入

参考：https://www.anthropic.com/news/contextual-retrieval
"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


@dataclass
class ContextualConfig:
    """Contextual Retrieval 配置"""
    
    # LLM 配置
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "qwen3-8b"  # 默认使用 Qwen3-8B
    
    # 生成参数
    max_context_tokens: int = 150  # 上下文最大 token 数
    temperature: float = 0.0       # 使用确定性输出
    
    # 处理参数
    max_workers: int = 4           # 并行处理数
    max_doc_tokens: int = 6000     # 文档最大 token 数（超过则截断）
    batch_size: int = 10           # 批处理大小
    
    # 是否启用
    enabled: bool = True


# 默认 Prompt 模板（参考 Anthropic 官方）
CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# 中文版本的 Prompt
CONTEXT_PROMPT_TEMPLATE_ZH = """<document>
{document}
</document>

以下是需要添加上下文的文本块：
<chunk>
{chunk}
</chunk>

请为这个文本块生成简短的上下文说明，用于提升搜索检索效果。说明应该包含：
1. 这个文本块来自什么文档/主题
2. 文本块讨论的核心内容
3. 与文档整体的关系

只输出上下文说明，不要其他内容。请用中文回答。"""

# 医学文献专用 Prompt
MEDICAL_CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>

以下是来自医学文献的文本块：
<chunk>
{chunk}
</chunk>

请为这个医学文本块生成简短的上下文说明（50-100字），用于提升医学信息检索效果。说明应包含：
- 文献类型（研究论文/临床指南/综述等）
- 涉及的疾病/症状/药物/治疗方法
- 文本块在文献中的位置（摘要/方法/结果/讨论等）

只输出上下文说明，不要其他内容。"""

# Qwen3-8B 优化版 Prompt（禁用思考模式，更简洁）
QWEN3_MEDICAL_CONTEXT_PROMPT = """你是医学文献分析专家。为文本块生成简短的检索上下文（50-80字）。

文档：
{document}

文本块：
{chunk}

要求：
1. 指明文献类型（论文/指南/综述）
2. 列出关键医学实体（疾病、药物、治疗）
3. 说明文本块在文档中的位置或作用

直接输出上下文，无需解释。/no_think"""

# Qwen3 系统提示词
QWEN3_SYSTEM_PROMPT = """你是一个医学文献分析助手。你的任务是为文本块生成简洁的上下文描述，帮助提升检索效果。
要求：
- 输出简洁，50-80字
- 只输出上下文描述本身
- 不要输出思考过程
- 使用中文"""


class ContextualEnricher:
    """
    上下文增强器
    
    为文本块添加上下文说明，提升检索精度。
    
    使用示例：
    ```python
    enricher = ContextualEnricher(
        api_key="your-api-key",
        base_url="http://localhost:8000/v1",  # 本地 Qwen3-8B
        model="qwen3-8b",
    )
    
    # 单个文档处理
    enriched_chunks = enricher.enrich_chunks(
        document="完整文档内容...",
        chunks=["chunk1", "chunk2", ...],
    )
    
    # 批量处理
    results = enricher.enrich_batch(documents_with_chunks)
    ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[ContextualConfig] = None,
        prompt_template: Optional[str] = None,
        use_qwen3_optimization: bool = True,
    ):
        self.config = config or ContextualConfig()
        
        # 覆盖配置
        self.api_key = api_key or self.config.api_key or os.getenv("CONTEXT_LLM_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = base_url or self.config.base_url or os.getenv("CONTEXT_LLM_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.model = model or self.config.model or os.getenv("CONTEXT_LLM_MODEL", "qwen3-8b")
        
        # 检测是否为 Qwen3 模型
        self._is_qwen3 = use_qwen3_optimization and "qwen3" in self.model.lower()
        
        # Prompt 模板：优先使用传入的，否则根据模型自动选择
        if prompt_template:
            self.prompt_template = prompt_template
        elif self._is_qwen3:
            self.prompt_template = QWEN3_MEDICAL_CONTEXT_PROMPT
            logger.info("检测到 Qwen3 模型，使用优化版提示词")
        else:
            self.prompt_template = MEDICAL_CONTEXT_PROMPT_TEMPLATE
        
        # 系统提示词（仅 Qwen3 使用）
        self.system_prompt = QWEN3_SYSTEM_PROMPT if self._is_qwen3 else None
        
        # 初始化客户端
        self._client: Optional[OpenAI] = None
        
        if self.api_key and self.base_url:
            self._init_client()
        
        logger.info(f"ContextualEnricher 初始化: model={self.model}, base_url={self.base_url}, qwen3_opt={self._is_qwen3}")
    
    def _init_client(self):
        """初始化 OpenAI 客户端"""
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.debug("LLM 客户端初始化成功")
        except Exception as e:
            logger.error(f"LLM 客户端初始化失败: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self._client is not None and self.config.enabled
    
    def _truncate_document(self, document: str) -> str:
        """截断过长的文档"""
        # 简单按字符截断（约 4 字符 = 1 token）
        max_chars = self.config.max_doc_tokens * 4
        if len(document) > max_chars:
            # 保留开头和结尾
            half = max_chars // 2
            document = document[:half] + "\n\n[...文档中间部分已省略...]\n\n" + document[-half:]
        return document
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def _generate_context(self, document: str, chunk: str) -> str:
        """调用 LLM 生成上下文"""
        if not self._client:
            return ""
        
        # 截断文档
        truncated_doc = self._truncate_document(document)
        
        # 构建 prompt
        prompt = self.prompt_template.format(
            document=truncated_doc,
            chunk=chunk,
        )
        
        try:
            # 构建消息列表
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_context_tokens,
            )
            
            context = response.choices[0].message.content.strip()
            
            # 清理 Qwen3 可能残留的思考标签
            if self._is_qwen3 and "<think>" in context:
                # 移除 <think>...</think> 内容
                import re
                context = re.sub(r"<think>.*?</think>", "", context, flags=re.DOTALL).strip()
            
            return context
            
        except Exception as e:
            logger.warning(f"生成上下文失败: {e}")
            return ""
    
    def enrich_chunk(self, document: str, chunk: str) -> Tuple[str, str]:
        """
        为单个 chunk 添加上下文
        
        Args:
            document: 完整文档
            chunk: 文本块
        
        Returns:
            (上下文, 增强后的文本)
        """
        if not self.is_available:
            return "", chunk
        
        context = self._generate_context(document, chunk)
        
        if context:
            # 将上下文拼接到 chunk 前面
            enriched = f"{context}\n\n{chunk}"
        else:
            enriched = chunk
        
        return context, enriched
    
    def enrich_chunks(
        self,
        document: str,
        chunks: List[str],
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        为文档的所有 chunk 添加上下文
        
        Args:
            document: 完整文档
            chunks: 文本块列表
            show_progress: 是否显示进度条
        
        Returns:
            [{"original": str, "context": str, "enriched": str}, ...]
        """
        if not self.is_available:
            logger.warning("Contextual Enricher 不可用，返回原始 chunks")
            return [
                {"original": c, "context": "", "enriched": c}
                for c in chunks
            ]
        
        results = []
        iterator = tqdm(chunks, desc="生成上下文") if show_progress else chunks
        
        for chunk in iterator:
            context, enriched = self.enrich_chunk(document, chunk)
            results.append({
                "original": chunk,
                "context": context,
                "enriched": enriched,
            })
        
        logger.info(f"上下文生成完成: {len(chunks)} 个块")
        return results
    
    def enrich_chunks_parallel(
        self,
        document: str,
        chunks: List[str],
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        并行为文档的所有 chunk 添加上下文
        
        Args:
            document: 完整文档
            chunks: 文本块列表
            show_progress: 是否显示进度条
        
        Returns:
            [{"original": str, "context": str, "enriched": str}, ...]
        """
        if not self.is_available:
            logger.warning("Contextual Enricher 不可用，返回原始 chunks")
            return [
                {"original": c, "context": "", "enriched": c}
                for c in chunks
            ]
        
        results = [None] * len(chunks)
        
        def process_chunk(idx_chunk: Tuple[int, str]) -> Tuple[int, Dict]:
            idx, chunk = idx_chunk
            context, enriched = self.enrich_chunk(document, chunk)
            return idx, {
                "original": chunk,
                "context": context,
                "enriched": enriched,
            }
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = list(executor.map(process_chunk, enumerate(chunks)))
            
            if show_progress:
                futures = tqdm(futures, total=len(chunks), desc="生成上下文(并行)")
            
            for idx, result in futures:
                results[idx] = result
        
        logger.info(f"并行上下文生成完成: {len(chunks)} 个块")
        return results
    
    def enrich_documents(
        self,
        documents: List[Dict],
        text_key: str = "text",
        chunks_key: str = "chunks",
        parallel: bool = True,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        批量处理多个文档
        
        Args:
            documents: [{"text": "文档内容", "chunks": ["chunk1", ...]}, ...]
            text_key: 文档文本字段名
            chunks_key: chunks 字段名
            parallel: 是否并行处理
            show_progress: 是否显示进度
        
        Returns:
            [{"text": ..., "chunks": ..., "enriched_chunks": [...]}, ...]
        """
        results = []
        
        doc_iterator = tqdm(documents, desc="处理文档") if show_progress else documents
        
        for doc in doc_iterator:
            document_text = doc.get(text_key, "")
            chunks = doc.get(chunks_key, [])
            
            if parallel:
                enriched = self.enrich_chunks_parallel(
                    document_text, chunks, show_progress=False
                )
            else:
                enriched = self.enrich_chunks(
                    document_text, chunks, show_progress=False
                )
            
            result = dict(doc)
            result["enriched_chunks"] = enriched
            results.append(result)
        
        return results


def create_contextual_enricher(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    use_medical_prompt: bool = True,
    use_qwen3_optimization: bool = True,
) -> ContextualEnricher:
    """
    创建上下文增强器的工厂函数
    
    Args:
        api_key: LLM API Key
        base_url: LLM API Base URL
        model: 模型名称
        use_medical_prompt: 是否使用医学专用 prompt
        use_qwen3_optimization: 是否启用 Qwen3 优化（自动检测模型）
    
    Returns:
        ContextualEnricher 实例
    """
    # 如果启用 Qwen3 优化，不传入 prompt_template，让类自动选择
    prompt = None
    if not use_qwen3_optimization:
        prompt = MEDICAL_CONTEXT_PROMPT_TEMPLATE if use_medical_prompt else CONTEXT_PROMPT_TEMPLATE_ZH
    
    return ContextualEnricher(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt_template=prompt,
        use_qwen3_optimization=use_qwen3_optimization,
    )
