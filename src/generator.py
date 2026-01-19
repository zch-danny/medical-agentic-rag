"""
答案生成模块 - 支持 DeepSeek / OpenAI 兼容 API
"""
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class GenerationConfig:
    """生成配置"""

    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9


class AnswerGenerator:
    """
    医疗问答生成器

    支持:
    - DeepSeek API (默认)
    - OpenAI API
    - 任意 OpenAI 兼容 API (通过 base_url 配置)
    """

    SYSTEM_PROMPT = """你是一个专业的医疗知识助手。请基于提供的参考文献，准确回答用户的医疗问题。

要求：
1. 回答必须基于参考文献内容，不要编造信息
2. 如果文献信息不足以回答问题，请明确说明
3. 对于涉及诊断、治疗的问题，提醒用户咨询专业医生
4. 引用文献时标注来源
5. 使用专业但易懂的语言"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        max_context_chars: Optional[int] = None,
        max_context_chars_per_doc: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.config = config or GenerationConfig(model=self.model)
        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else _get_int_env("LLM_CONTEXT_MAX_CHARS", 12000)
        )
        self.max_context_chars_per_doc = (
            max_context_chars_per_doc
            if max_context_chars_per_doc is not None
            else _get_int_env("LLM_CONTEXT_MAX_CHARS_PER_DOC", 2000)
        )
        if self.max_context_chars is not None and self.max_context_chars <= 0:
            self.max_context_chars = None
        if self.max_context_chars_per_doc is not None and self.max_context_chars_per_doc <= 0:
            self.max_context_chars_per_doc = None

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未设置")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"AnswerGenerator 初始化完成: {self.base_url}, model={self.config.model}")

    def _get_score(self, doc: Dict) -> float:
        """用于展示的分数提取"""
        v = doc.get("rerank_score")
        if v is None:
            v = doc.get("score")
        if v is None and doc.get("distance") is not None:
            try:
                v = -float(doc["distance"])
            except Exception:
                v = 0.0
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0
    def _truncate_text(self, text: str, max_chars: Optional[int]) -> str:
        """按字符数截断文本，确保长度不超过 max_chars"""
        if max_chars is None or max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return text[: max_chars - 1].rstrip() + "…"

    def _format_context(self, documents: List[Dict]) -> str:
        """格式化检索结果为上下文"""
        context_parts: List[str] = []
        sep = "\n\n---\n\n"
        current_len = 0
        for i, doc in enumerate(documents, 1):
            entity = doc.get("entity", doc)
            text = entity.get("original_text") or entity.get("text", "")
            source = entity.get("source", "未知来源")
            score = self._get_score(doc)
            if self.max_context_chars_per_doc:
                text = self._truncate_text(text, self.max_context_chars_per_doc)

            prefix = f"[文献{i}] 来源: {source} (相关度: {score:.4f})\n"
            doc_str = prefix + text

            if self.max_context_chars:
                if context_parts:
                    if current_len + len(sep) > self.max_context_chars:
                        break
                    context_parts.append(sep)
                    current_len += len(sep)

                remaining = self.max_context_chars - current_len
                if remaining <= 0:
                    break
                if len(doc_str) > remaining:
                    allowed_text = max(0, remaining - len(prefix))
                    if allowed_text <= 0:
                        break
                    text = self._truncate_text(text, allowed_text)
                    doc_str = prefix + text

                context_parts.append(doc_str)
                current_len += len(doc_str)
            else:
                if context_parts:
                    context_parts.append(sep)
                context_parts.append(doc_str)
            context_parts.append(f"[文献{i}] 来源: {source} (相关度: {score:.4f})\n{text}")

        return "".join(context_parts)

    def _build_messages(self, query: str, context: str) -> List[Dict]:
        """构建消息列表"""
        user_content = f"""参考文献：
{context}

用户问题：{query}

请基于以上参考文献回答用户问题。"""

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _create_completion(self, messages: List[Dict], stream: bool):
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=stream,
        )

    def generate_stream(self, query: str, documents: List[Dict]) -> Iterator[str]:
        """流式生成回答"""
        if not documents:
            yield "未找到相关文献，无法回答该问题。"
            return

        context = self._format_context(documents)
        messages = self._build_messages(query, context)

        response = self._create_completion(messages, stream=True)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def generate_sync(self, query: str, documents: List[Dict]) -> str:
        """同步生成回答（非流式）"""
        if not documents:
            return "未找到相关文献，无法回答该问题。"

        context = self._format_context(documents)
        messages = self._build_messages(query, context)

        response = self._create_completion(messages, stream=False)
        return response.choices[0].message.content

    def generate(
        self, query: str, documents: List[Dict], stream: bool = True
    ) -> Union[Iterator[str], str]:
        """统一入口：stream=True 返回迭代器，stream=False 返回字符串"""
        if stream:
            return self.generate_stream(query, documents)
        return self.generate_sync(query, documents)
