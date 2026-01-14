"""
嵌入向量缓存 - 避免重复计算
"""
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger


class EmbeddingCache:
    """
    嵌入向量缓存，避免重复计算

    使用文本哈希作为键，将嵌入向量存储为 JSON 文件
    """

    def __init__(self, cache_dir: Path):
        """
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                return json.loads(self.index_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
        return {}

    def _save_index(self):
        """保存缓存索引"""
        self.index_file.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def _hash_text(self, text: str) -> str:
        """计算文本哈希"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional[List[float]]:
        """
        获取缓存的嵌入向量

        Args:
            text: 文本

        Returns:
            嵌入向量或 None
        """
        key = self._hash_text(text)
        if key in self._index:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text())
                except Exception:
                    pass
        return None

    def set(self, text: str, embedding: List[float]):
        """
        缓存嵌入向量

        Args:
            text: 文本
            embedding: 嵌入向量
        """
        key = self._hash_text(text)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(embedding))
        self._index[key] = len(text)
        self._save_index()

    def get_or_compute(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], List[List[float]]]
    ) -> List[List[float]]:
        """
        批量获取嵌入，缓存未命中时调用 compute_fn 计算

        Args:
            texts: 文本列表
            compute_fn: 计算函数，接收文本列表，返回嵌入列表

        Returns:
            嵌入向量列表
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        to_compute: List[str] = []
        to_compute_idx: List[int] = []

        # 检查缓存
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append(text)
                to_compute_idx.append(i)

        # 计算未缓存的
        if to_compute:
            cache_hit = len(texts) - len(to_compute)
            logger.info(f"缓存命中 {cache_hit}/{len(texts)}，计算 {len(to_compute)} 条")
            computed = compute_fn(to_compute)
            for idx, emb, text in zip(to_compute_idx, computed, to_compute):
                results[idx] = emb
                self.set(text, emb)

        return results  # type: ignore

    def clear(self):
        """清空缓存"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self._index = {}
        self._save_index()
        logger.info("缓存已清空")

    def stats(self) -> Dict:
        """缓存统计"""
        return {
            "total_entries": len(self._index),
            "cache_dir": str(self.cache_dir),
        }
