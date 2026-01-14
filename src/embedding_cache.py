"""
嵌入向量缓存 - 避免重复计算
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger


class EmbeddingCache:
    """
    嵌入向量缓存，避免重复计算

    支持:
    - 模型版本追踪（模型变化时自动失效）
    - TTL 过期机制
    - 批量写入优化
    """

    def __init__(
        self,
        cache_dir: Path,
        model_id: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ):
        """
        Args:
            cache_dir: 缓存目录
            model_id: 模型标识（如 "Qwen/Qwen3-Embedding-8B:4096"），用于版本追踪
            ttl_seconds: 缓存过期时间（秒），None 表示永不过期
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.model_id = model_id
        self.ttl_seconds = ttl_seconds
        self._index = self._load_index()
        self._dirty = False  # 追踪是否有未保存的变更

        # 模型版本检查
        self._check_model_version()

    def _load_index(self) -> Dict:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                return json.loads(self.index_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
        return {"model_id": None, "entries": {}}

    def _save_index(self):
        """保存缓存索引"""
        self._index["model_id"] = self.model_id
        self.index_file.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self._dirty = False

    def _check_model_version(self):
        """检查模型版本，不匹配则清空缓存"""
        cached_model_id = self._index.get("model_id")
        if self.model_id and cached_model_id and cached_model_id != self.model_id:
            logger.warning(
                f"模型版本变化: {cached_model_id} -> {self.model_id}，清空旧缓存"
            )
            self.clear()
        elif self.model_id and not cached_model_id:
            # 首次设置模型 ID
            self._index["model_id"] = self.model_id
            self._dirty = True

    def _hash_text(self, text: str) -> str:
        """计算文本哈希（使用完整 SHA256）"""
        return hashlib.sha256(text.encode()).hexdigest()

    def _is_expired(self, entry: Dict) -> bool:
        """检查缓存条目是否过期"""
        if self.ttl_seconds is None:
            return False
        created_at = entry.get("created_at", 0)
        return (time.time() - created_at) > self.ttl_seconds

    def get(self, text: str) -> Optional[List[float]]:
        """
        获取缓存的嵌入向量

        Args:
            text: 文本

        Returns:
            嵌入向量或 None
        """
        key = self._hash_text(text)
        entries = self._index.get("entries", {})

        if key in entries:
            entry = entries[key]
            # 检查过期
            if self._is_expired(entry):
                self._remove_entry(key)
                return None

            cache_file = self.cache_dir / f"{key[:32]}.json"  # 文件名用截断哈希
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text())
                except Exception:
                    pass
        return None

    def set(self, text: str, embedding: List[float], save_index: bool = True):
        """
        缓存嵌入向量

        Args:
            text: 文本
            embedding: 嵌入向量
            save_index: 是否立即保存索引（批量操作时可设为 False）
        """
        key = self._hash_text(text)
        cache_file = self.cache_dir / f"{key[:32]}.json"
        cache_file.write_text(json.dumps(embedding))

        if "entries" not in self._index:
            self._index["entries"] = {}

        self._index["entries"][key] = {
            "text_len": len(text),
            "created_at": time.time(),
        }
        self._dirty = True

        if save_index:
            self._save_index()

    def _remove_entry(self, key: str):
        """删除单个缓存条目"""
        entries = self._index.get("entries", {})
        if key in entries:
            del entries[key]
            self._dirty = True

        cache_file = self.cache_dir / f"{key[:32]}.json"
        if cache_file.exists():
            cache_file.unlink()

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

            # 批量写入（不立即保存索引）
            for idx, emb, text in zip(to_compute_idx, computed, to_compute):
                results[idx] = emb
                self.set(text, emb, save_index=False)

            # 一次性保存索引
            self._save_index()

        return results  # type: ignore

    def flush(self):
        """强制保存索引（如果有未保存的变更）"""
        if self._dirty:
            self._save_index()

    def clear(self):
        """清空缓存"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self._index = {"model_id": self.model_id, "entries": {}}
        self._save_index()
        logger.info("缓存已清空")

    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理数量"""
        if self.ttl_seconds is None:
            return 0

        entries = self._index.get("entries", {})
        expired_keys = [
            key for key, entry in entries.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            self._save_index()
            logger.info(f"清理过期缓存: {len(expired_keys)} 条")

        return len(expired_keys)

    def stats(self) -> Dict:
        """缓存统计"""
        entries = self._index.get("entries", {})
        return {
            "total_entries": len(entries),
            "cache_dir": str(self.cache_dir),
            "model_id": self._index.get("model_id"),
            "ttl_seconds": self.ttl_seconds,
        }
