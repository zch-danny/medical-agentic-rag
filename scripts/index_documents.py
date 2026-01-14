#!/usr/bin/env python
"""
文档索引脚本 - 解析 PDF 并构建向量索引

支持:
- 多种分块策略: markdown, semantic, recursive, fixed
- Contextual Retrieval: 使用 LLM 为每个块添加上下文说明

用法示例:
    # 基本用法（Markdown分块）
    python scripts/index_documents.py --input-dir data/documents
    
    # 指定分块策略
    python scripts/index_documents.py --input-dir data/documents --chunking-strategy semantic
    
    # 启用 Contextual Retrieval
    python scripts/index_documents.py --input-dir data/documents --enable-contextual
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import settings
from src.chunker import ChunkConfig, ChunkingStrategy
from src.contextual import ContextualConfig, ContextualEnricher, create_contextual_enricher
from src.document_loader import MinerUDocumentLoader
from src.embedder import MedicalEmbedder
from src.embedding_cache import EmbeddingCache
from src.vector_store import VectorStore


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.add(
        settings.LOG_DIR / "index_{time}.log",
        rotation="100 MB",
        retention="7 days",
        level="DEBUG",
    )


def main():
    parser = argparse.ArgumentParser(
        description="索引医疗文献到向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/index_documents.py --input-dir data/documents
  
  # 使用语义分块
  python scripts/index_documents.py --input-dir data/documents --chunking-strategy semantic
  
  # 启用 Contextual Retrieval
  python scripts/index_documents.py --input-dir data/documents --enable-contextual
  
  # 完整示例
  python scripts/index_documents.py --input-dir data/documents --chunking-strategy markdown --enable-contextual --chunk-size 600
"""
    )
    
    # 基本参数
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(settings.DOCUMENTS_DIR),
        help="PDF 文件目录",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.MILVUS_COLLECTION,
        help="Milvus collection 名称",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="嵌入批处理大小",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除并重建 collection",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    
    # 分块策略参数
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=settings.CHUNKING_STRATEGY,
        choices=["markdown", "semantic", "recursive", "fixed"],
        help="分块策略: markdown(推荐), semantic, recursive, fixed",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE,
        help="块大小（字符数）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.CHUNK_OVERLAP,
        help="块重叠（字符数）",
    )
    
    # Contextual Retrieval 参数
    parser.add_argument(
        "--enable-contextual",
        action="store_true",
        default=settings.CONTEXTUAL_ENABLED,
        help="启用 Contextual Retrieval（使用 LLM 为每个块添加上下文）",
    )
    parser.add_argument(
        "--context-llm-url",
        type=str,
        default=settings.CONTEXT_LLM_BASE_URL,
        help="Contextual LLM API URL",
    )
    parser.add_argument(
        "--context-llm-model",
        type=str,
        default=settings.CONTEXT_LLM_MODEL,
        help="Contextual LLM 模型名称",
    )
    
    args = parser.parse_args()

    setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"目录中没有 PDF 文件: {input_dir}")
        sys.exit(0)

    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

    # 初始化组件
    logger.info("初始化组件...")
    
    # 分块配置
    chunk_config = ChunkConfig(
        strategy=ChunkingStrategy(args.chunking_strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=settings.CHUNK_MIN_SIZE,
        max_chunk_size=settings.CHUNK_MAX_SIZE,
        semantic_threshold=settings.SEMANTIC_THRESHOLD,
    )
    
    # Contextual Retrieval 配置
    contextual_enricher = None
    if args.enable_contextual:
        logger.info("启用 Contextual Retrieval...")
        contextual_enricher = create_contextual_enricher(
            api_key=settings.CONTEXT_LLM_API_KEY,
            base_url=args.context_llm_url,
            model=args.context_llm_model,
            use_medical_prompt=True,
        )
        if not contextual_enricher.is_available:
            logger.warning("Contextual Enricher 不可用，请检查 LLM 配置")
            contextual_enricher = None
    
    # 文档加载器
    loader = MinerUDocumentLoader(
        backend=settings.MINERU_BACKEND,
        output_dir=str(settings.PARSED_DIR),
        chunking_strategy=args.chunking_strategy,
        chunk_config=chunk_config,
        contextual_enricher=contextual_enricher,
    )
    
    # 嵌入模型
    embedder = MedicalEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        instruction=settings.MEDICAL_INSTRUCT,
        max_length=settings.MAX_SEQ_LENGTH,
    )
    
    # 如果使用语义分块，设置嵌入函数
    if args.chunking_strategy == "semantic":
        logger.info("使用语义分块，设置嵌入函数...")
        loader.set_embedding_fn(
            lambda texts: embedder.encode_documents(texts, batch_size=args.batch_size, show_progress=False)
        )
    
    cache = EmbeddingCache(cache_dir=settings.EMBEDDING_CACHE_DIR)
    vector_store = VectorStore(collection_name=args.collection)
    
    logger.info(f"分块策略: {args.chunking_strategy}")
    logger.info(f"块大小: {args.chunk_size}, 重叠: {args.chunk_overlap}")
    logger.info(f"Contextual Retrieval: {'enabled' if contextual_enricher else 'disabled'}")

    # 如果需要重建 collection
    if args.recreate:
        logger.warning(f"删除 collection: {args.collection}")
        vector_store.drop_collection()
        vector_store = VectorStore(collection_name=args.collection)

    total_chunks = 0
    total_indexed = 0

    for pdf_path in pdf_files:
        logger.info(f"处理文件: {pdf_path.name}")

        try:
            # 解析 PDF + 分块 + 可选 Contextual Retrieval
            chunks = loader.load(
                str(pdf_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                enable_contextual=args.enable_contextual,
            )
            logger.info(f"  解析得到 {len(chunks)} 个文本块")
            total_chunks += len(chunks)

            if not chunks:
                continue

            # 生成嵌入 (带缓存)
            # 注意: 嵌入使用 "text" 字段（包含上下文增强后的文本）
            texts = [c["text"] for c in chunks]
            # 保存原始文本用于展示
            original_texts = [c.get("original_text", c["text"]) for c in chunks]
            contexts = [c.get("context", "") for c in chunks]
            
            # 构建元数据（包含上下文信息）
            metadatas = []
            for i, c in enumerate(chunks):
                meta = c.get("metadata", {})
                meta["original_text"] = original_texts[i]
                meta["context"] = contexts[i]
                meta["has_context"] = bool(contexts[i])
                metadatas.append(meta)

            def _compute_fn(batch_texts):
                return embedder.encode_documents(
                    batch_texts,
                    batch_size=args.batch_size,
                    show_progress=False,
                )

            embeddings = cache.get_or_compute(texts, _compute_fn)

            # 插入向量数据库
            vector_store.insert(embeddings=embeddings, texts=original_texts, metadata=metadatas)
            total_indexed += len(chunks)
            
            # 统计上下文增强数量
            enriched_count = sum(1 for c in contexts if c)
            if enriched_count:
                logger.info(f"  已索引 {len(chunks)} 个文本块 ({enriched_count} 个已上下文增强)")
            else:
                logger.info(f"  已索引 {len(chunks)} 个文本块")

        except Exception as e:
            logger.error(f"处理文件失败 {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("=" * 50)
    logger.info(f"索引完成!")
    logger.info(f"  处理文件: {len(pdf_files)}")
    logger.info(f"  总文本块: {total_chunks}")
    logger.info(f"  已索引: {total_indexed}")
    logger.info(f"  分块策略: {args.chunking_strategy}")
    logger.info(f"  Contextual Retrieval: {'enabled' if args.enable_contextual else 'disabled'}")


if __name__ == "__main__":
    main()
