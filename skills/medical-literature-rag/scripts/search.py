#!/usr/bin/env python3
"""
医疗文献检索脚本 - 供 Skill 调用

用法:
    python search.py --query "糖尿病治疗" --top-k 10
    python search.py -q "高血压用药" -k 5 --no-rerank
"""

import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def search(query: str, top_k: int = 10, use_rerank: bool = True) -> list:
    """执行检索"""
    from config import settings
    from src.retriever import MedicalRetriever
    
    retriever = MedicalRetriever(settings, lazy_load=True)
    results = retriever.search(query, top_k=top_k, use_rerank=use_rerank)
    
    return results


def format_results(results: list) -> str:
    """格式化检索结果"""
    if not results:
        return "未找到相关文献。"
    
    output = []
    for i, r in enumerate(results, 1):
        entity = r.get("entity", r)
        text = entity.get("text", "")
        source = entity.get("source", "未知来源")
        score = r.get("rerank_score") or r.get("score") or r.get("distance", 0)
        
        output.append(f"[{i}] 来源: {source}")
        output.append(f"    相关度: {score:.4f}")
        output.append(f"    内容: {text[:500]}{'...' if len(text) > 500 else ''}")
        output.append("")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="医疗文献检索")
    parser.add_argument("-q", "--query", required=True, help="检索查询")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="返回结果数量")
    parser.add_argument("--no-rerank", action="store_true", help="禁用重排序")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()
    
    results = search(
        query=args.query,
        top_k=args.top_k,
        use_rerank=not args.no_rerank
    )
    
    if args.json:
        # JSON 输出（供程序解析）
        output = []
        for r in results:
            entity = r.get("entity", r)
            output.append({
                "text": entity.get("text", ""),
                "source": entity.get("source", ""),
                "score": r.get("rerank_score") or r.get("score") or r.get("distance", 0)
            })
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 人类可读输出
        print(f"查询: {args.query}")
        print(f"参数: top_k={args.top_k}, rerank={'否' if args.no_rerank else '是'}")
        print("=" * 60)
        print(format_results(results))


if __name__ == "__main__":
    main()
