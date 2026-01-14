"""
高级文本分块器

支持多种分块策略：
1. Markdown感知分块 - 利用文档结构（标题、段落）
2. 语义分块 - 基于句子嵌入相似度
3. 递归分块 - 层级分隔符递归切分
4. 固定大小分块 - 简单滑窗（fallback）
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    MARKDOWN = "markdown"       # Markdown结构感知
    SEMANTIC = "semantic"       # 语义相似度分块
    RECURSIVE = "recursive"     # 递归分块
    FIXED = "fixed"            # 固定大小滑窗


@dataclass
class ChunkConfig:
    """分块配置"""
    strategy: ChunkingStrategy = ChunkingStrategy.MARKDOWN
    chunk_size: int = 512           # 目标块大小（字符数）
    chunk_overlap: int = 64         # 块重叠
    min_chunk_size: int = 100       # 最小块大小
    max_chunk_size: int = 1500      # 最大块大小
    
    # 语义分块参数
    semantic_threshold: float = 0.5  # 语义相似度阈值（低于此值则切分）
    
    # Markdown分块参数
    heading_weight: float = 1.5      # 标题边界权重


@dataclass
class Chunk:
    """文本块"""
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "chunk_index": self.index,
            "chunk_start": self.start_char,
            "chunk_end": self.end_char,
            **self.metadata,
        }


class BaseChunker(ABC):
    """分块器基类"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """将文本分块"""
        pass
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()


class MarkdownChunker(BaseChunker):
    """
    Markdown感知分块器
    
    利用Markdown结构（标题、段落、列表）进行智能分块：
    1. 首先按标题（## / ###）切分章节
    2. 章节内按段落切分
    3. 超长段落按句子切分
    """
    
    # Markdown标题正则
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    # 段落分隔
    PARAGRAPH_PATTERN = re.compile(r'\n\n+')
    # 句子分隔（中英文）
    SENTENCE_PATTERN = re.compile(r'([。！？.!?]+["\'）\)」』]?)\s*')
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """按Markdown结构分块"""
        text = self._clean_text(text)
        metadata = metadata or {}
        chunks: List[Chunk] = []
        
        # Step 1: 按标题切分章节
        sections = self._split_by_headings(text)
        
        chunk_idx = 0
        for section_title, section_text, section_level in sections:
            # Step 2: 章节内按段落切分
            paragraphs = self._split_by_paragraphs(section_text)
            
            current_chunk_parts: List[str] = []
            current_length = 0
            chunk_start = text.find(section_text) if section_text else 0
            
            # 添加标题作为上下文
            title_prefix = f"## {section_title}\n\n" if section_title else ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_len = len(para)
                
                # 如果段落本身超过最大块大小，按句子切分
                if para_len > self.config.max_chunk_size:
                    # 先保存当前累积的内容
                    if current_chunk_parts:
                        chunk_text = title_prefix + "\n\n".join(current_chunk_parts)
                        chunks.append(Chunk(
                            text=chunk_text,
                            index=chunk_idx,
                            start_char=chunk_start,
                            end_char=chunk_start + len(chunk_text),
                            metadata={**metadata, "section": section_title},
                        ))
                        chunk_idx += 1
                        current_chunk_parts = []
                        current_length = 0
                    
                    # 按句子切分超长段落
                    sentence_chunks = self._split_long_paragraph(para, title_prefix, section_title)
                    for sent_chunk in sentence_chunks:
                        chunks.append(Chunk(
                            text=sent_chunk,
                            index=chunk_idx,
                            start_char=chunk_start,
                            end_char=chunk_start + len(sent_chunk),
                            metadata={**metadata, "section": section_title},
                        ))
                        chunk_idx += 1
                    continue
                
                # 检查是否超过目标大小
                if current_length + para_len + len(title_prefix) > self.config.chunk_size:
                    # 保存当前块
                    if current_chunk_parts:
                        chunk_text = title_prefix + "\n\n".join(current_chunk_parts)
                        chunks.append(Chunk(
                            text=chunk_text,
                            index=chunk_idx,
                            start_char=chunk_start,
                            end_char=chunk_start + len(chunk_text),
                            metadata={**metadata, "section": section_title},
                        ))
                        chunk_idx += 1
                        
                        # 保留重叠部分
                        overlap_parts = self._get_overlap_parts(current_chunk_parts)
                        current_chunk_parts = overlap_parts
                        current_length = sum(len(p) for p in overlap_parts)
                
                current_chunk_parts.append(para)
                current_length += para_len
            
            # 保存最后一个块
            if current_chunk_parts:
                chunk_text = title_prefix + "\n\n".join(current_chunk_parts)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=chunk_idx,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        metadata={**metadata, "section": section_title},
                    ))
                    chunk_idx += 1
        
        logger.debug(f"Markdown分块: 输入{len(text)}字符 → {len(chunks)}个块")
        return chunks
    
    def _split_by_headings(self, text: str) -> List[Tuple[str, str, int]]:
        """按标题切分章节，返回 [(标题, 内容, 级别), ...]"""
        sections: List[Tuple[str, str, int]] = []
        
        # 找到所有标题位置
        headings = list(self.HEADING_PATTERN.finditer(text))
        
        if not headings:
            # 没有标题，整个文本作为一个章节
            return [("", text, 0)]
        
        # 第一个标题之前的内容
        if headings[0].start() > 0:
            pre_content = text[:headings[0].start()].strip()
            if pre_content:
                sections.append(("", pre_content, 0))
        
        # 每个标题和其内容
        for i, match in enumerate(headings):
            level = len(match.group(1))  # 标题级别
            title = match.group(2).strip()
            
            # 内容到下一个标题或文档结尾
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end].strip()
            
            if content or title:
                sections.append((title, content, level))
        
        return sections
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落切分"""
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_long_paragraph(self, para: str, title_prefix: str, section_title: str) -> List[str]:
        """按句子切分超长段落"""
        sentences = self.SENTENCE_PATTERN.split(para)
        # 重新组合句子（split会分开标点）
        merged_sentences: List[str] = []
        current = ""
        for s in sentences:
            if not s:
                continue
            if self.SENTENCE_PATTERN.match(s):
                current += s
                if current.strip():
                    merged_sentences.append(current.strip())
                current = ""
            else:
                current += s
        if current.strip():
            merged_sentences.append(current.strip())
        
        # 组合成合适大小的块
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = len(title_prefix)
        
        for sent in merged_sentences:
            if current_len + len(sent) > self.config.chunk_size and current_chunk:
                chunks.append(title_prefix + " ".join(current_chunk))
                # 保留最后一句作为重叠
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_len = len(title_prefix) + (len(current_chunk[0]) if current_chunk else 0)
            
            current_chunk.append(sent)
            current_len += len(sent)
        
        if current_chunk:
            chunks.append(title_prefix + " ".join(current_chunk))
        
        return chunks
    
    def _get_overlap_parts(self, parts: List[str]) -> List[str]:
        """获取重叠部分"""
        if not parts or self.config.chunk_overlap <= 0:
            return []
        
        # 从后往前取，直到达到重叠大小
        overlap_parts: List[str] = []
        overlap_len = 0
        for part in reversed(parts):
            if overlap_len + len(part) <= self.config.chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_len += len(part)
            else:
                break
        
        return overlap_parts


class RecursiveChunker(BaseChunker):
    """
    递归分块器
    
    按层级分隔符递归切分：
    1. 先按章节标题切分
    2. 再按段落切分
    3. 再按句子切分
    4. 最后按字符切分
    """
    
    # 分隔符优先级（从高到低）
    SEPARATORS = [
        r'\n#{1,6}\s+',      # Markdown标题
        r'\n\n+',            # 段落
        r'(?<=[。！？.!?])\s*',  # 句子
        r'\n',               # 换行
        r'\s+',              # 空白
    ]
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """递归分块"""
        text = self._clean_text(text)
        metadata = metadata or {}
        
        raw_chunks = self._recursive_split(text, 0)
        
        chunks: List[Chunk] = []
        current_pos = 0
        
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < self.config.min_chunk_size:
                continue
            
            start = text.find(chunk_text, current_pos)
            if start == -1:
                start = current_pos
            
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=idx,
                start_char=start,
                end_char=start + len(chunk_text),
                metadata=metadata,
            ))
            current_pos = start + len(chunk_text)
        
        logger.debug(f"递归分块: 输入{len(text)}字符 → {len(chunks)}个块")
        return chunks
    
    def _recursive_split(self, text: str, sep_idx: int) -> List[str]:
        """递归切分"""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        
        if sep_idx >= len(self.SEPARATORS):
            # 所有分隔符都试过了，强制按大小切分
            return self._force_split(text)
        
        # 尝试当前分隔符
        pattern = re.compile(self.SEPARATORS[sep_idx])
        parts = pattern.split(text)
        parts = [p for p in parts if p and p.strip()]
        
        if len(parts) <= 1:
            # 这个分隔符没用，尝试下一个
            return self._recursive_split(text, sep_idx + 1)
        
        # 合并小块，递归处理大块
        result: List[str] = []
        current = ""
        
        for part in parts:
            if len(current) + len(part) <= self.config.chunk_size:
                current += part
            else:
                if current:
                    result.append(current)
                
                if len(part) > self.config.chunk_size:
                    # 递归处理超大块
                    result.extend(self._recursive_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        
        if current:
            result.append(current)
        
        return result
    
    def _force_split(self, text: str) -> List[str]:
        """强制按大小切分"""
        chunks: List[str] = []
        step = self.config.chunk_size - self.config.chunk_overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.config.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    语义分块器
    
    基于句子嵌入相似度进行分块：
    1. 将文本切分为句子
    2. 计算相邻句子的嵌入相似度
    3. 在相似度低谷处切分
    """
    
    SENTENCE_PATTERN = re.compile(r'([。！？.!?]+["\'）\)」』]?)\s*')
    
    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        super().__init__(config)
        self._embedding_fn = embedding_fn
    
    def set_embedding_fn(self, fn: Callable[[List[str]], np.ndarray]):
        """设置嵌入函数"""
        self._embedding_fn = fn
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """语义分块"""
        if self._embedding_fn is None:
            logger.warning("未设置嵌入函数，回退到递归分块")
            return RecursiveChunker(self.config).chunk(text, metadata)
        
        text = self._clean_text(text)
        metadata = metadata or {}
        
        # 切分句子
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text, index=0, start_char=0, end_char=len(text), metadata=metadata)]
        
        # 计算句子嵌入
        try:
            embeddings = self._embedding_fn(sentences)
        except Exception as e:
            logger.error(f"计算嵌入失败: {e}，回退到递归分块")
            return RecursiveChunker(self.config).chunk(text, metadata)
        
        # 计算相邻句子相似度
        similarities = self._compute_similarities(embeddings)
        
        # 找到切分点（相似度低谷）
        split_points = self._find_split_points(similarities, sentences)
        
        # 构建块
        chunks: List[Chunk] = []
        current_pos = 0
        
        for idx, (start_sent, end_sent) in enumerate(split_points):
            chunk_sentences = sentences[start_sent:end_sent + 1]
            chunk_text = " ".join(chunk_sentences)
            
            start_char = text.find(chunk_sentences[0], current_pos)
            if start_char == -1:
                start_char = current_pos
            
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=metadata,
            ))
            current_pos = start_char + len(chunk_text)
        
        logger.debug(f"语义分块: 输入{len(text)}字符, {len(sentences)}句 → {len(chunks)}个块")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """切分句子"""
        parts = self.SENTENCE_PATTERN.split(text)
        sentences: List[str] = []
        current = ""
        
        for part in parts:
            if not part:
                continue
            if self.SENTENCE_PATTERN.match(part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """计算相邻句子余弦相似度"""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(float(sim))
        return similarities
    
    def _find_split_points(
        self,
        similarities: List[float],
        sentences: List[str],
    ) -> List[Tuple[int, int]]:
        """找到切分点"""
        if not similarities:
            return [(0, len(sentences) - 1)]
        
        # 计算阈值（使用百分位数）
        threshold = np.percentile(similarities, 20)  # 第20百分位作为阈值
        threshold = max(threshold, self.config.semantic_threshold)
        
        # 找到低于阈值的点作为切分点
        split_indices = [0]  # 开始
        current_len = len(sentences[0])
        
        for i, sim in enumerate(similarities):
            current_len += len(sentences[i + 1])
            
            # 切分条件：相似度低于阈值 且 当前块足够大
            if sim < threshold and current_len >= self.config.min_chunk_size:
                split_indices.append(i + 1)
                current_len = 0
            # 强制切分：当前块太大
            elif current_len >= self.config.max_chunk_size:
                split_indices.append(i + 1)
                current_len = 0
        
        split_indices.append(len(sentences))  # 结束
        
        # 转换为区间
        ranges = []
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1] - 1
            if start <= end:
                ranges.append((start, end))
        
        return ranges


class FixedSizeChunker(BaseChunker):
    """固定大小分块器（简单滑窗）"""
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """固定大小滑窗分块"""
        text = self._clean_text(text)
        metadata = metadata or {}
        
        if not text:
            return []
        
        chunks: List[Chunk] = []
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        
        idx = 0
        start = 0
        while start < len(text):
            end = min(len(text), start + self.config.chunk_size)
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    metadata=metadata,
                ))
                idx += 1
            
            if end >= len(text):
                break
            start += step
        
        logger.debug(f"固定分块: 输入{len(text)}字符 → {len(chunks)}个块")
        return chunks


def create_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.MARKDOWN,
    config: Optional[ChunkConfig] = None,
    embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> BaseChunker:
    """
    创建分块器工厂函数
    
    Args:
        strategy: 分块策略
        config: 分块配置
        embedding_fn: 嵌入函数（语义分块需要）
    
    Returns:
        对应策略的分块器实例
    """
    config = config or ChunkConfig(strategy=strategy)
    
    if strategy == ChunkingStrategy.MARKDOWN:
        return MarkdownChunker(config)
    elif strategy == ChunkingStrategy.SEMANTIC:
        chunker = SemanticChunker(config, embedding_fn)
        return chunker
    elif strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(config)
    elif strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker(config)
    else:
        logger.warning(f"未知策略 {strategy}，使用Markdown分块")
        return MarkdownChunker(config)
