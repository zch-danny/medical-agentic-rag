#!/usr/bin/env python
"""
交互式搜索脚本
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.pipeline import MedicalRAGPipeline, RAGConfig


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "WARNING"
    logger.add(sys.stderr, level=level)


def main():
    parser = argparse.ArgumentParser(description="医疗文献搜索")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="搜索查询（不提供则进入交互模式）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="混合检索权重 (0=纯BM25, 1=纯向量)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回结果数量",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="禁用答案生成",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # 初始化 Pipeline
    config = RAGConfig(
        alpha=args.alpha,
        final_top_k=args.top_k,
        enable_generation=not args.no_generate,
        stream_output=True,
    )
    pipeline = MedicalRAGPipeline(config=config)

    def do_search(query: str):
        """执行搜索并显示结果"""
        print(f"\n🔍 搜索: {query}\n")
        print("-" * 60)

        result = pipeline.query(query)

        # 显示检索结果
        print(f"\n📚 检索到 {len(result.documents)} 篇相关文献:\n")
        for i, doc in enumerate(result.documents, 1):
            entity = doc.get("entity", doc)
            text = entity.get("original_text") or entity.get("text", "")
            source = entity.get("source", "未知")
            score = doc.get("rerank_score", doc.get("score", 0))

            # 截断长文本
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"[{i}] 来源: {source} | 相关度: {score:.4f}")
            print(f"    {preview}\n")

        # 显示生成的答案
        if result.answer_stream:
            print("-" * 60)
            print("\n💡 AI 回答:\n")
            for chunk in result.answer_stream:
                print(chunk, end="", flush=True)
            print("\n")
        elif result.answer:
            print("-" * 60)
            print(f"\n💡 AI 回答:\n{result.answer}\n")

    # 单次查询模式
    if args.query:
        do_search(args.query)
        return

    # 交互模式
    print("=" * 60)
    print("医疗文献搜索系统 (输入 'exit' 或 'quit' 退出)")
    print("=" * 60)

    while True:
        try:
            query = input("\n请输入问题: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("再见!")
                break
            do_search(query)
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            logger.error(f"搜索出错: {e}")
            print(f"❌ 搜索出错: {e}")


if __name__ == "__main__":
    main()
