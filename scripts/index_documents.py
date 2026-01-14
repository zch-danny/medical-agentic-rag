#!/usr/bin/env python
"""
文档索引脚本 - 解析 PDF 并构建向量索引
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import settings
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
    parser = argparse.ArgumentParser(description="索引医疗文献到向量数据库")
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
    loader = MinerUDocumentLoader(backend=settings.MINERU_BACKEND, output_dir=str(settings.PARSED_DIR))
    embedder = MedicalEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        instruction=settings.MEDICAL_INSTRUCT,
        max_length=settings.MAX_SEQ_LENGTH,
    )
    cache = EmbeddingCache(cache_dir=settings.EMBEDDING_CACHE_DIR)
    vector_store = VectorStore(collection_name=args.collection)

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
            # 解析 PDF + 分块
            chunks = loader.load(
                str(pdf_path),
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            logger.info(f"  解析得到 {len(chunks)} 个文本块")
            total_chunks += len(chunks)

            if not chunks:
                continue

            # 生成嵌入 (带缓存)
            texts = [c["text"] for c in chunks]
            metadatas = [c.get("metadata", {}) for c in chunks]

            def _compute_fn(batch_texts):
                return embedder.encode_documents(
                    batch_texts,
                    batch_size=args.batch_size,
                    show_progress=False,
                )

            embeddings = cache.get_or_compute(texts, _compute_fn)

            # 插入向量数据库
            vector_store.insert(embeddings=embeddings, texts=texts, metadata=metadatas)
            total_indexed += len(chunks)
            logger.info(f"  已索引 {len(chunks)} 个文本块")

        except Exception as e:
            logger.error(f"处理文件失败 {pdf_path.name}: {e}")
            continue

    logger.info("=" * 50)
    logger.info(f"索引完成!")
    logger.info(f"  处理文件: {len(pdf_files)}")
    logger.info(f"  总文本块: {total_chunks}")
    logger.info(f"  已索引: {total_indexed}")


if __name__ == "__main__":
    main()
