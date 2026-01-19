#!/usr/bin/env python
"""
Embedding 模型评估脚本

对比原始模型和微调模型在医学检索任务上的效果。

使用方式:
    # 评估原始模型
    python scripts/evaluate_embedding.py --model Qwen/Qwen3-Embedding-8B

    # 评估微调模型
    python scripts/evaluate_embedding.py --model models/medical-embedding/final

    # 对比两个模型
    python scripts/evaluate_embedding.py --compare Qwen/Qwen3-Embedding-8B models/medical-embedding/final
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def setup_logging(verbose: bool = False):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def load_eval_data(filepath: str) -> List[Dict]:
    """
    加载评估数据
    
    格式:
    {"query": "...", "pos": ["正确文档1", ...], "neg": ["干扰文档1", ...]}
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("query") and item.get("pos"):
                    data.append(item)
    return data


def compute_embeddings(
    model,
    texts: List[str],
    batch_size: int = 8,
    show_progress: bool = True,
) -> np.ndarray:
    """计算文本嵌入"""
    embeddings = []
    
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="计算嵌入")
    
    for i in iterator:
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    
    return np.vstack(embeddings)


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    relevant_indices: List[List[int]],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    评估检索效果
    
    Args:
        query_embeddings: 查询嵌入 [num_queries, dim]
        doc_embeddings: 文档嵌入 [num_docs, dim]
        relevant_indices: 每个查询的相关文档索引 [[idx1, idx2], ...]
        k_values: 评估的 K 值
    
    Returns:
        评估指标字典
    """
    # 计算相似度
    similarities = np.dot(query_embeddings, doc_embeddings.T)  # [num_queries, num_docs]
    
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"mrr@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    
    for i, relevant in enumerate(relevant_indices):
        # 获取排名
        scores = similarities[i]
        ranked_indices = np.argsort(scores)[::-1]  # 降序排列
        
        relevant_set = set(relevant)
        
        for k in k_values:
            top_k = ranked_indices[:k]
            
            # Recall@K
            hits = len(set(top_k) & relevant_set)
            recall = hits / len(relevant_set) if relevant_set else 0
            metrics[f"recall@{k}"].append(recall)
            
            # MRR@K
            mrr = 0
            for rank, idx in enumerate(top_k):
                if idx in relevant_set:
                    mrr = 1.0 / (rank + 1)
                    break
            metrics[f"mrr@{k}"].append(mrr)
            
            # NDCG@K
            dcg = 0
            for rank, idx in enumerate(top_k):
                if idx in relevant_set:
                    dcg += 1.0 / np.log2(rank + 2)
            ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant_set), k)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            metrics[f"ndcg@{k}"].append(ndcg)
    
    # 计算平均值
    return {name: np.mean(values) for name, values in metrics.items()}


def evaluate_model(
    model_name: str,
    eval_data: List[Dict],
    batch_size: int = 8,
    max_samples: int = None,
) -> Dict[str, float]:
    """
    评估单个模型
    """
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"加载模型: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # 限制样本数
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    # 收集所有文本
    queries = []
    all_docs = []
    relevant_indices = []
    doc_offset = 0
    
    for item in eval_data:
        queries.append(item["query"])
        
        # 正样本
        pos_docs = item.get("pos", [])
        pos_indices = list(range(doc_offset, doc_offset + len(pos_docs)))
        all_docs.extend(pos_docs)
        doc_offset += len(pos_docs)
        
        # 负样本 (如果有)
        neg_docs = item.get("neg", [])
        all_docs.extend(neg_docs)
        doc_offset += len(neg_docs)
        
        relevant_indices.append(pos_indices)
    
    logger.info(f"查询数: {len(queries)}, 文档数: {len(all_docs)}")
    
    # 计算嵌入
    logger.info("计算查询嵌入...")
    query_emb = compute_embeddings(model, queries, batch_size)
    
    logger.info("计算文档嵌入...")
    doc_emb = compute_embeddings(model, all_docs, batch_size)
    
    # 评估
    logger.info("计算指标...")
    metrics = evaluate_retrieval(query_emb, doc_emb, relevant_indices)
    
    return metrics


def compare_models(
    model1_name: str,
    model2_name: str,
    eval_data: List[Dict],
    batch_size: int = 8,
    max_samples: int = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    对比两个模型
    
    Returns:
        (model1_metrics, model2_metrics, diff)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"评估模型 1: {model1_name}")
    logger.info(f"{'='*50}")
    metrics1 = evaluate_model(model1_name, eval_data, batch_size, max_samples)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"评估模型 2: {model2_name}")
    logger.info(f"{'='*50}")
    metrics2 = evaluate_model(model2_name, eval_data, batch_size, max_samples)
    
    # 计算差异
    diff = {}
    for key in metrics1:
        diff[key] = metrics2[key] - metrics1[key]
    
    return metrics1, metrics2, diff


def print_metrics(metrics: Dict[str, float], title: str = "评估结果"):
    """打印指标"""
    print(f"\n{'='*50}")
    print(title)
    print("="*50)
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")


def print_comparison(
    metrics1: Dict, 
    metrics2: Dict, 
    diff: Dict,
    name1: str = "模型1",
    name2: str = "模型2",
):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print("模型对比")
    print("="*60)
    print(f"{'指标':<15} {name1:<15} {name2:<15} {'差异':<15}")
    print("-"*60)
    
    for key in sorted(metrics1.keys()):
        v1 = metrics1[key]
        v2 = metrics2[key]
        d = diff[key]
        sign = "+" if d > 0 else ""
        color = "\033[92m" if d > 0 else "\033[91m" if d < 0 else ""
        reset = "\033[0m"
        print(f"{key:<15} {v1:<15.4f} {v2:<15.4f} {color}{sign}{d:.4f}{reset}")


def main():
    parser = argparse.ArgumentParser(description="Embedding 模型评估")
    
    parser.add_argument(
        "--model",
        type=str,
        help="要评估的模型名称或路径",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("MODEL1", "MODEL2"),
        help="对比两个模型",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/training/val.jsonl",
        help="评估数据文件",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批大小",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="最大评估样本数",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="结果输出文件 (JSON)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细日志",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # 检查评估数据
    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        logger.error(f"评估数据不存在: {eval_path}")
        logger.info("请先构建训练数据:")
        logger.info('  python -c "from src.training.data_builder import build_medical_training_data; build_medical_training_data()"')
        sys.exit(1)
    
    # 加载评估数据
    eval_data = load_eval_data(str(eval_path))
    logger.info(f"加载评估数据: {len(eval_data)} 样本")
    
    results = {}
    
    if args.compare:
        # 对比模式
        model1, model2 = args.compare
        metrics1, metrics2, diff = compare_models(
            model1, model2, eval_data,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        
        print_comparison(metrics1, metrics2, diff, model1, model2)
        
        results = {
            "model1": {"name": model1, "metrics": metrics1},
            "model2": {"name": model2, "metrics": metrics2},
            "diff": diff,
        }
        
    elif args.model:
        # 单模型评估
        metrics = evaluate_model(
            args.model, eval_data,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        
        print_metrics(metrics, f"模型: {args.model}")
        results = {"model": args.model, "metrics": metrics}
        
    else:
        parser.print_help()
        sys.exit(1)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
