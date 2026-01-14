#!/usr/bin/env python
"""
MIRAGE 医学基准评估脚本 - 使用标准医学 QA 数据集评估检索质量

MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation) 包含
7,663 个来自 5 个医学 QA 数据集的问题:
- MMLU-Med: 医学考试题 (1,089 题)
- MedQA: 美国医师执照考试题 (1,273 题)
- MedMCQA: 印度医学入学考试题 (4,183 题)
- PubMedQA: 生物医学研究问答 (500 题)
- BioASQ: 生物医学语义问答 (618 题)

使用方法:
    # 下载 MIRAGE 基准数据
    python scripts/evaluate_mirage.py --download

    # 评估单个数据集
    python scripts/evaluate_mirage.py --dataset mmlu --limit 100

    # 评估所有数据集
    python scripts/evaluate_mirage.py --dataset all --limit 50
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.pipeline import MedicalRAGPipeline, RAGConfig


# MIRAGE 数据集 URL
MIRAGE_BENCHMARK_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"

# 数据集名称映射
DATASET_NAMES = {
    "mmlu": "MMLU-Med",
    "medqa": "MedQA",
    "medmcqa": "MedMCQA",
    "pubmedqa": "PubMedQA",
    "bioasq": "BioASQ",
}


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def download_mirage_benchmark(output_dir: Path) -> Path:
    """下载 MIRAGE 基准数据"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mirage_benchmark.json"
    
    if output_file.exists():
        logger.info(f"MIRAGE 基准数据已存在: {output_file}")
        return output_file
    
    logger.info(f"下载 MIRAGE 基准数据...")
    try:
        urlretrieve(MIRAGE_BENCHMARK_URL, output_file)
        logger.info(f"下载完成: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("请手动下载: https://github.com/Teddy-XiongGZ/MIRAGE")
        sys.exit(1)


def load_mirage_dataset(
    benchmark_file: Path,
    dataset_name: str,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    加载 MIRAGE 数据集

    Returns:
        问题列表，每个问题包含:
        - question: 问题文本
        - options: 选项字典 {"A": "...", "B": "...", ...}
        - answer: 正确答案 (A/B/C/D)
    """
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    if dataset_name == "all":
        # 合并所有数据集
        questions = []
        for name in DATASET_NAMES.keys():
            if name in benchmark:
                ds_questions = benchmark[name]
                for q in ds_questions:
                    q["dataset"] = name
                questions.extend(ds_questions)
    else:
        if dataset_name not in benchmark:
            logger.error(f"数据集 {dataset_name} 不存在")
            logger.info(f"可用数据集: {list(benchmark.keys())}")
            sys.exit(1)
        questions = benchmark[dataset_name]
        for q in questions:
            q["dataset"] = dataset_name
    
    if limit and limit < len(questions):
        # 随机采样
        import random
        random.seed(42)
        questions = random.sample(questions, limit)
    
    return questions


def format_question_with_options(question: Dict) -> str:
    """格式化问题和选项为检索查询"""
    q_text = question["question"]
    # 只用问题文本进行检索（符合 MIRAGE 的 Question-Only Retrieval 设置）
    return q_text


def evaluate_retrieval_accuracy(
    pipeline: MedicalRAGPipeline,
    questions: List[Dict],
    use_generation: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    评估检索准确率

    评估方式:
    1. 使用问题检索相关文档
    2. 如果启用生成，使用 LLM 基于检索结果回答
    3. 比较答案与正确答案

    Returns:
        (metrics_dict, detailed_results)
    """
    correct = 0
    total = 0
    results = []
    
    for i, q in enumerate(questions):
        question_text = format_question_with_options(q)
        options = q.get("options", {})
        correct_answer = q.get("answer", "")
        dataset = q.get("dataset", "unknown")
        
        logger.info(f"[{i+1}/{len(questions)}] ({dataset}) {question_text[:60]}...")
        
        try:
            # 执行检索
            rag_result = pipeline.query(
                query=question_text,
                enable_generation=use_generation,
                stream=False,
            )
            
            # 提取预测答案
            predicted_answer = ""
            if use_generation and rag_result.answer:
                # 从生成的答案中提取选项
                answer_text = rag_result.answer.upper()
                for opt in options.keys():
                    if opt in answer_text:
                        predicted_answer = opt
                        break
            
            # 计算准确率
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
            total += 1
            
            # 记录详细结果
            results.append({
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "dataset": dataset,
                "retrieved_docs": len(rag_result.documents),
            })
            
        except Exception as e:
            logger.error(f"  评估失败: {e}")
            results.append({
                "question": question_text,
                "error": str(e),
                "is_correct": False,
                "dataset": dataset,
            })
            total += 1
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0.0
    
    # 按数据集统计
    dataset_stats = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        if ds not in dataset_stats:
            dataset_stats[ds] = {"correct": 0, "total": 0}
        dataset_stats[ds]["total"] += 1
        if r.get("is_correct", False):
            dataset_stats[ds]["correct"] += 1
    
    for ds in dataset_stats:
        stats = dataset_stats[ds]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    metrics = {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "dataset_stats": dataset_stats,
    }
    
    return metrics, results


def evaluate_retrieval_only(
    pipeline: MedicalRAGPipeline,
    questions: List[Dict],
) -> Dict:
    """
    仅评估检索质量（不使用生成）

    检查检索到的文档是否包含与正确答案相关的信息
    """
    results = {
        "has_relevant_context": 0,
        "total": 0,
        "avg_retrieved_docs": 0,
    }
    
    total_docs = 0
    
    for i, q in enumerate(questions):
        question_text = format_question_with_options(q)
        correct_answer = q.get("answer", "")
        options = q.get("options", {})
        correct_option_text = options.get(correct_answer, "")
        
        logger.debug(f"[{i+1}/{len(questions)}] {question_text[:50]}...")
        
        try:
            # 仅检索，不生成
            rag_result = pipeline.query(
                query=question_text,
                enable_generation=False,
            )
            
            # 检查检索结果是否包含相关信息
            has_relevant = False
            for doc in rag_result.documents:
                entity = doc.get("entity", doc)
                text = entity.get("original_text") or entity.get("text", "")
                # 简单检查：正确选项的关键词是否出现在检索结果中
                if correct_option_text and len(correct_option_text) > 5:
                    keywords = correct_option_text.split()[:3]
                    if any(kw.lower() in text.lower() for kw in keywords if len(kw) > 3):
                        has_relevant = True
                        break
            
            if has_relevant:
                results["has_relevant_context"] += 1
            
            total_docs += len(rag_result.documents)
            results["total"] += 1
            
        except Exception as e:
            logger.warning(f"  检索失败: {e}")
            results["total"] += 1
    
    results["avg_retrieved_docs"] = total_docs / results["total"] if results["total"] > 0 else 0
    results["context_hit_rate"] = results["has_relevant_context"] / results["total"] if results["total"] > 0 else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MIRAGE 医学基准评估")
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载 MIRAGE 基准数据",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq", "all"],
        help="要评估的数据集",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评估数量（用于快速测试）",
    )
    parser.add_argument(
        "--use-generation",
        action="store_true",
        help="使用 LLM 生成答案（否则仅评估检索）",
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

    # 数据目录
    data_dir = Path(__file__).parent.parent / "data" / "evaluation"
    benchmark_file = data_dir / "mirage_benchmark.json"

    # 下载数据
    if args.download or not benchmark_file.exists():
        benchmark_file = download_mirage_benchmark(data_dir)
        if args.download:
            return

    # 加载数据集
    questions = load_mirage_dataset(benchmark_file, args.dataset, args.limit)
    logger.info(f"加载 {len(questions)} 个问题 (数据集: {args.dataset})")

    # 初始化 Pipeline
    config = RAGConfig(
        alpha=args.alpha,
        final_top_k=args.top_k,
        enable_generation=args.use_generation,
        stream_output=False,
    )
    pipeline = MedicalRAGPipeline(config=config)

    # 执行评估
    if args.use_generation:
        logger.info("开始 RAG + 生成评估...")
        metrics, results = evaluate_retrieval_accuracy(
            pipeline, questions, use_generation=True
        )
    else:
        logger.info("开始检索质量评估...")
        metrics = evaluate_retrieval_only(pipeline, questions)
        results = []

    # 输出结果
    print("\n" + "=" * 60)
    print(f"MIRAGE 评估结果 (数据集: {args.dataset})")
    print("=" * 60)

    if args.use_generation:
        print(f"\n整体准确率: {metrics['overall_accuracy']:.2%}")
        print(f"正确: {metrics['correct']}/{metrics['total']}")
        
        if "dataset_stats" in metrics:
            print("\n各数据集统计:")
            for ds, stats in metrics["dataset_stats"].items():
                print(f"  {DATASET_NAMES.get(ds, ds)}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    else:
        print(f"\n上下文命中率: {metrics.get('context_hit_rate', 0):.2%}")
        print(f"平均检索文档数: {metrics.get('avg_retrieved_docs', 0):.1f}")
        print(f"评估数量: {metrics.get('total', 0)}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "dataset": args.dataset,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "use_generation": args.use_generation,
            "metrics": metrics,
        }
        if results:
            output_data["detailed_results"] = results
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
