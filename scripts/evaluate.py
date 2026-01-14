#!/usr/bin/env python
"""
检索评估脚本 - 计算 MRR, NDCG, Recall 等指标
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.pipeline import MedicalRAGPipeline, RAGConfig


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def load_test_data(filepath: str) -> List[Dict]:
    """
    加载测试数据

    格式:
    [
        {
            "query": "问题文本",
            "relevant_docs": ["doc_id_1", "doc_id_2", ...]
        },
        ...
    ]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Mean Reciprocal Rank @ K"""
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall @ K"""
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 因为 log2(1) = 0

    # 理想 DCG
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate(
    pipeline: MedicalRAGPipeline,
    test_data: List[Dict],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    执行评估

    Args:
        pipeline: RAG 管道
        test_data: 测试数据
        k_values: 要评估的 K 值列表

    Returns:
        评估指标字典
    """
    metrics = {f"mrr@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    for i, item in enumerate(test_data):
        query = item["query"]
        relevant = item["relevant_docs"]

        logger.debug(f"评估查询 {i+1}/{len(test_data)}: {query[:50]}...")

        # 执行检索
        result = pipeline.query(query, enable_generation=False)

        # 提取检索到的文档 ID
        retrieved = []
        for doc in result.documents:
            entity = doc.get("entity", doc)
            doc_id = entity.get("doc_id") or entity.get("source", "")
            retrieved.append(doc_id)

        # 计算指标
        for k in k_values:
            metrics[f"mrr@{k}"].append(mrr_at_k(retrieved, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    # 计算平均值
    return {name: np.mean(values) for name, values in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="检索系统评估")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="测试数据文件 (JSON 格式)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="混合检索权重",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="候选检索数量",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出结果文件",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # 加载测试数据
    test_file = Path(args.test_file)
    if not test_file.exists():
        logger.error(f"测试文件不存在: {test_file}")
        sys.exit(1)

    test_data = load_test_data(str(test_file))
    logger.info(f"加载 {len(test_data)} 条测试数据")

    # 初始化 Pipeline
    config = RAGConfig(
        alpha=args.alpha,
        candidate_top_k=args.top_k,
        final_top_k=args.top_k,
        enable_generation=False,
    )
    pipeline = MedicalRAGPipeline(config=config)

    # 执行评估
    logger.info("开始评估...")
    results = evaluate(pipeline, test_data)

    # 输出结果
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    for metric, value in sorted(results.items()):
        print(f"  {metric}: {value:.4f}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
