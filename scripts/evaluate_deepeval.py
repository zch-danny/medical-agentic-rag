#!/usr/bin/env python
"""
DeepEval RAG 评估脚本 - 端到端评估检索增强生成质量

支持指标:
- Faithfulness: 答案是否基于检索文档，无幻觉（医疗场景最重要）
- Answer Relevancy: 答案与问题的相关性
- Contextual Precision: 检索文档中相关内容的比例
- Contextual Recall: 相关信息是否被检索到

使用方法:
    python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries.json
    python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries.json --metrics faithfulness relevancy
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# DeepEval imports
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )
    from deepeval.test_case import LLMTestCase
except ImportError:
    logger.error("请先安装 deepeval: pip install deepeval")
    sys.exit(1)

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
            "question": "问题文本",
            "ground_truth": "参考答案（可选，用于 Contextual Recall）",
            "language": "zh" 或 "en"（可选）
        },
        ...
    ]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def create_metrics(metric_names: List[str], threshold: float = 0.5) -> List:
    """创建评估指标"""
    metric_map = {
        "faithfulness": FaithfulnessMetric(threshold=threshold),
        "relevancy": AnswerRelevancyMetric(threshold=threshold),
        "precision": ContextualPrecisionMetric(threshold=threshold),
        "recall": ContextualRecallMetric(threshold=threshold),
    }
    
    metrics = []
    for name in metric_names:
        if name in metric_map:
            metrics.append(metric_map[name])
        else:
            logger.warning(f"未知指标: {name}，可选: {list(metric_map.keys())}")
    
    return metrics


def run_rag_and_create_test_case(
    pipeline: MedicalRAGPipeline,
    question: str,
    ground_truth: Optional[str] = None,
) -> LLMTestCase:
    """
    运行 RAG 管道并创建测试用例
    """
    # 执行 RAG
    result = pipeline.query(
        query=question,
        enable_generation=True,
        stream=False,
    )
    
    # 提取检索上下文
    contexts = []
    for doc in result.documents:
        entity = doc.get("entity", doc)
        text = entity.get("original_text") or entity.get("text", "")
        if text:
            contexts.append(text)
    
    # 创建测试用例
    test_case = LLMTestCase(
        input=question,
        actual_output=result.answer or "",
        retrieval_context=contexts,
        expected_output=ground_truth,  # 可选，用于 recall
    )
    
    return test_case


def evaluate_rag(
    pipeline: MedicalRAGPipeline,
    test_data: List[Dict],
    metrics: List,
    verbose: bool = False,
) -> Dict:
    """
    执行 RAG 评估

    Args:
        pipeline: RAG 管道
        test_data: 测试数据
        metrics: 评估指标
        verbose: 是否显示详细信息

    Returns:
        评估结果字典
    """
    test_cases = []
    
    for i, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item.get("ground_truth")
        language = item.get("language", "unknown")
        
        logger.info(f"[{i+1}/{len(test_data)}] 评估: {question[:50]}... ({language})")
        
        try:
            test_case = run_rag_and_create_test_case(
                pipeline=pipeline,
                question=question,
                ground_truth=ground_truth,
            )
            test_cases.append(test_case)
            
            if verbose:
                logger.debug(f"  检索文档数: {len(test_case.retrieval_context)}")
                logger.debug(f"  答案长度: {len(test_case.actual_output)} 字符")
                
        except Exception as e:
            logger.error(f"  评估失败: {e}")
            continue
    
    if not test_cases:
        logger.error("没有成功的测试用例")
        return {}
    
    # 运行 DeepEval 评估
    logger.info(f"\n开始 DeepEval 评估，共 {len(test_cases)} 个测试用例...")
    
    try:
        results = evaluate(test_cases, metrics)
        return results
    except Exception as e:
        logger.error(f"DeepEval 评估失败: {e}")
        return {}


def evaluate_single_metric(
    test_cases: List[LLMTestCase],
    metric,
) -> Dict:
    """单独评估一个指标并返回详细结果"""
    scores = []
    reasons = []
    
    for i, test_case in enumerate(test_cases):
        try:
            metric.measure(test_case)
            scores.append(metric.score)
            reasons.append(getattr(metric, "reason", ""))
        except Exception as e:
            logger.warning(f"测试用例 {i+1} 评估失败: {e}")
            scores.append(0.0)
            reasons.append(str(e))
    
    return {
        "metric_name": metric.__class__.__name__,
        "scores": scores,
        "reasons": reasons,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "pass_rate": sum(1 for s in scores if s >= metric.threshold) / len(scores) if scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="DeepEval RAG 评估")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="测试数据文件 (JSON 格式)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "relevancy", "precision"],
        choices=["faithfulness", "relevancy", "precision", "recall"],
        help="要评估的指标",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="评估通过阈值",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="混合检索权重",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="检索返回数量",
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

    # 统计语言分布
    lang_counts = {}
    for item in test_data:
        lang = item.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    logger.info(f"语言分布: {lang_counts}")

    # 创建评估指标
    metrics = create_metrics(args.metrics, args.threshold)
    logger.info(f"评估指标: {[m.__class__.__name__ for m in metrics]}")

    # 初始化 Pipeline
    config = RAGConfig(
        alpha=args.alpha,
        final_top_k=args.top_k,
        enable_generation=True,
        stream_output=False,
    )
    pipeline = MedicalRAGPipeline(config=config)

    # 执行评估
    logger.info("开始评估...")
    results = evaluate_rag(pipeline, test_data, metrics, args.verbose)

    # 输出结果
    print("\n" + "=" * 60)
    print("DeepEval RAG 评估结果")
    print("=" * 60)
    
    if results:
        # DeepEval 2.0+ 返回的结果格式可能不同
        # 这里提供一个通用的输出方式
        print(f"测试用例数: {len(test_data)}")
        print(f"评估指标: {args.metrics}")
        print(f"阈值: {args.threshold}")
        print("\n详细结果请查看 DeepEval 输出或 Confident AI 平台")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "test_count": len(test_data),
            "metrics": args.metrics,
            "threshold": args.threshold,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "language_distribution": lang_counts,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
