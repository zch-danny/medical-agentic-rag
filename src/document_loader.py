"""
文档加载器 - 基于 MinerU 2.7 的 PDF 解析

支持:
- MinerU PDF解析（多后端）
- 高级分块策略（Markdown感知/语义/递归）
- Contextual Retrieval（上下文增强）
"""
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from src.chunker import (
    BaseChunker,
    Chunk,
    ChunkConfig,
    ChunkingStrategy,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
    create_chunker,
)
from src.contextual import ContextualEnricher, create_contextual_enricher


def extract_metadata_from_text(text: str, filename: str) -> Dict:
    """
    从文本中提取元数据
    
    尝试识别：标题、作者、发表日期、期刊、DOI、关键词等
    """
    metadata = {}
    
    # 取前 2000 字符用于元数据提取（通常在文档开头）
    header = text[:2000] if len(text) > 2000 else text
    
    # 提取 DOI
    doi_match = re.search(r'\b(10\.\d{4,}/[^\s]+)\b', header)
    if doi_match:
        metadata['doi'] = doi_match.group(1).rstrip('.,;')
    
    # 提取年份（常见格式：2023、(2023)、2023年）
    year_match = re.search(r'[（(]?((?:19|20)\d{2})[)）年]?', header)
    if year_match:
        metadata['year'] = year_match.group(1)
    
    # 提取关键词（中文）
    keywords_match = re.search(r'关键词[：:](.*?)(?:\n|摘要|Abstract)', header, re.DOTALL)
    if keywords_match:
        keywords = keywords_match.group(1).strip()
        # 按分号、逗号、空格分割
        keywords = re.split(r'[;；,，\s]+', keywords)
        keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) < 20]
        if keywords:
            metadata['keywords'] = keywords
    
    # 提取关键词（英文）
    if 'keywords' not in metadata:
        keywords_match = re.search(r'Keywords?[：:](.+?)(?:\n\n|Introduction|Background)', header, re.IGNORECASE | re.DOTALL)
        if keywords_match:
            keywords = keywords_match.group(1).strip()
            keywords = re.split(r'[;,]+', keywords)
            keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) < 50]
            if keywords:
                metadata['keywords'] = keywords
    
    # 尝试提取标题（通常是第一行非空文本，且较短）
    lines = [l.strip() for l in header.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # 标题通常 10-100 字符
        if 10 < len(first_line) < 150 and not first_line.startswith(('http', 'doi', 'DOI')):
            metadata['title'] = first_line
    
    # 如果没提取到标题，用文件名
    if 'title' not in metadata:
        # 去掉文件名前缀（如 file_id）和扩展名
        clean_name = re.sub(r'^[a-f0-9]{8}_', '', filename)  # 移除 UUID 前缀
        clean_name = re.sub(r'\.pdf$', '', clean_name, flags=re.IGNORECASE)
        if clean_name:
            metadata['title'] = clean_name
    
    return metadata


class MinerUDocumentLoader:
    """
    基于 MinerU 2.7 的文档加载器

    MinerU 支持三种后端：
    - pipeline: CPU 友好，配置要求低
    - vlm-auto-engine: GPU 加速，精度最高
    - hybrid-auto-engine: 结合两者优势（默认推荐）

    该 Loader 负责:
    1) 调用 MinerU 将 PDF 转为 Markdown
    2) 使用高级分块策略切分文本（Markdown感知/语义/递归）
    3) 可选：使用 Contextual Retrieval 增强每个块的上下文
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(
        self,
        backend: str = "hybrid-auto-engine",
        output_dir: str = "./temp_mineru",
        chunking_strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.MARKDOWN,
        chunk_config: Optional[ChunkConfig] = None,
        contextual_enricher: Optional[ContextualEnricher] = None,
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        """
        Args:
            backend: MinerU 后端，可选 "pipeline", "vlm-auto-engine", "hybrid-auto-engine"
            output_dir: MinerU 输出目录
            chunking_strategy: 分块策略，可选 "markdown", "semantic", "recursive", "fixed"
            chunk_config: 分块配置
            contextual_enricher: 上下文增强器（可选）
            embedding_fn: 嵌入函数（语义分块需要）
        """
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 分块策略
        if isinstance(chunking_strategy, str):
            chunking_strategy = ChunkingStrategy(chunking_strategy)
        self.chunking_strategy = chunking_strategy
        self.chunk_config = chunk_config or ChunkConfig(strategy=chunking_strategy)
        
        # 创建分块器
        self._chunker = create_chunker(
            strategy=chunking_strategy,
            config=self.chunk_config,
            embedding_fn=embedding_fn,
        )
        
        # 上下文增强器
        self._contextual_enricher = contextual_enricher
        
        logger.info(f"MinerUDocumentLoader 初始化: backend={backend}, chunking={chunking_strategy.value}")

    def load_file(self, file_path: str) -> Optional[Dict]:
        """
        加载单个 PDF 文件

        Returns:
            {"text": str, "metadata": {"source": str, "path": str}} 或 None
        """
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的文件类型: {path.suffix}")
            return None

        try:
            # 调用 MinerU CLI
            output_path = self.output_dir / path.stem
            cmd = [
                "mineru",
                "-p", str(path),
                "-o", str(output_path),
                "-b", self.backend,
            ]
            logger.debug(f"执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # 读取输出的 Markdown 文件
            md_file = output_path / "auto" / f"{path.stem}.md"
            if not md_file.exists():
                # 尝试其他可能的路径
                md_files = list(output_path.rglob("*.md"))
                if md_files:
                    md_file = md_files[0]
                else:
                    logger.error(f"未找到 MinerU 输出文件: {path}")
                    return None

            text = md_file.read_text(encoding="utf-8")

            # 提取元数据
            doc_metadata = extract_metadata_from_text(text, path.name)
            doc_metadata.update({
                "source": path.name,
                "path": str(path.absolute()),
                "file_size": path.stat().st_size,
                "indexed_at": datetime.now().isoformat(),
            })
            
            return {
                "text": text,
                "metadata": doc_metadata,
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"MinerU 解析失败: {path}, 错误: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"加载文件失败: {path}, 错误: {e}")
            return None

    def set_embedding_fn(self, fn: Callable[[List[str]], np.ndarray]):
        """
        设置嵌入函数（语义分块需要）
        
        Args:
            fn: 接受文本列表，返回嵌入向量数组的函数
        """
        if isinstance(self._chunker, SemanticChunker):
            self._chunker.set_embedding_fn(fn)
    
    def set_contextual_enricher(self, enricher: ContextualEnricher):
        """
        设置上下文增强器
        
        Args:
            enricher: ContextualEnricher 实例
        """
        self._contextual_enricher = enricher
    
    def _chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        使用配置的分块策略切分文本
        
        Args:
            text: 要切分的文本
            metadata: 元数据
        
        Returns:
            Chunk 对象列表
        """
        return self._chunker.chunk(text, metadata)
    
    def _enrich_chunks(
        self,
        full_text: str,
        chunks: List[Chunk],
        enable_contextual: bool = True,
    ) -> List[Dict]:
        """
        可选：使用 Contextual Retrieval 增强 chunks
        
        Args:
            full_text: 完整文档文本
            chunks: 分块后的 Chunk 列表
            enable_contextual: 是否启用上下文增强
        
        Returns:
            [{"text": str, "original_text": str, "context": str, "metadata": {...}}, ...]
        """
        results = []
        
        if enable_contextual and self._contextual_enricher and self._contextual_enricher.is_available:
            # 使用 Contextual Retrieval
            chunk_texts = [c.text for c in chunks]
            enriched = self._contextual_enricher.enrich_chunks(
                document=full_text,
                chunks=chunk_texts,
                show_progress=True,
            )
            
            for chunk, enrich_info in zip(chunks, enriched):
                results.append({
                    "text": enrich_info["enriched"],  # 增强后的文本（用于嵌入）
                    "original_text": enrich_info["original"],  # 原始文本（用于展示）
                    "context": enrich_info["context"],  # 生成的上下文
                    "metadata": chunk.to_dict(),
                })
        else:
            # 不使用 Contextual Retrieval
            for chunk in chunks:
                results.append({
                    "text": chunk.text,
                    "original_text": chunk.text,
                    "context": "",
                    "metadata": chunk.to_dict(),
                })
        
        return results

    def load(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_contextual: bool = False,
    ) -> List[Dict]:
        """
        加载单个 PDF，并返回分块后的 chunks
        
        Args:
            file_path: PDF 文件路径
            chunk_size: 块大小（覆盖配置）
            chunk_overlap: 块重叠（覆盖配置）
            enable_contextual: 是否启用 Contextual Retrieval

        Returns:
            [{"text": str, "original_text": str, "context": str, "metadata": {...}}]
        """
        # 覆盖配置
        if chunk_size is not None:
            self.chunk_config.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_config.chunk_overlap = chunk_overlap
        
        doc = self.load_file(file_path)
        if not doc:
            return []

        text = doc.get("text", "")
        base_meta = doc.get("metadata", {})

        # 使用高级分块策略
        chunks = self._chunk_text(text, base_meta)
        logger.info(f"分块完成: {len(chunks)} 个块 (策略: {self.chunking_strategy.value})")
        
        # 可选：Contextual Retrieval 增强
        results = self._enrich_chunks(text, chunks, enable_contextual)
        
        return results

    def load_directory(
        self,
        directory: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_contextual: bool = False,
    ) -> List[Dict]:
        """
        加载目录下所有 PDF 文件，返回聚合后的 chunks
        
        Args:
            directory: PDF 文件目录
            chunk_size: 块大小（覆盖配置）
            chunk_overlap: 块重叠（覆盖配置）
            enable_contextual: 是否启用 Contextual Retrieval
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"目录不存在: {directory}")
            return []

        all_chunks: List[Dict] = []
        pdf_files = list(dir_path.glob("*.pdf"))
        logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")

        for pdf_file in pdf_files:
            all_chunks.extend(
                self.load(
                    str(pdf_file),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    enable_contextual=enable_contextual,
                )
            )

        return all_chunks
    
    # ============== 兼容旧 API ==============
    
    def _iter_chunks(self, text: str, chunk_size: int, chunk_overlap: int):
        """[已废弃] 按字符数滑窗切分，保留用于兼容"""
        logger.warning("_iter_chunks 已废弃，请使用高级分块策略")
        text = (text or "").strip()
        if not text:
            return
        if chunk_size <= 0:
            yield text, 0, len(text)
            return

        step = max(1, chunk_size - max(0, chunk_overlap))
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                yield chunk, start, end
            if end >= n:
                break
            start += step


# 别名
DocumentLoader = MinerUDocumentLoader
