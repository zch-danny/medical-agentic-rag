"""
缓存管理模块

提供多种缓存策略:
- LRU 缓存（最近最少使用）
- TTL 缓存（带过期时间）
- 持久化缓存（磁盘存储）
"""

import hashlib
import json
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

from loguru import logger


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """缓存条目"""
    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """更新访问计数"""
        self.hits += 1


class BaseCache(ABC, Generic[T]):
    """缓存基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """缓存大小"""
        pass
    
    def __contains__(self, key: str) -> bool:
        """检查 key 是否存在"""
        return self.get(key) is not None


class LRUCache(BaseCache[T]):
    """
    LRU 缓存（最近最少使用）
    
    当缓存满时，自动淘汰最久未使用的条目
    
    示例:
        ```python
        cache = LRUCache(max_size=100)
        cache.set("key1", "value1")
        value = cache.get("key1")
        ```
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """获取缓存值（并移到最近使用）"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            # 移到最后（最近使用）
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            expires_at = time.time() + ttl if ttl else None
            
            if key in self._cache:
                # 更新现有条目
                self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
                self._cache.move_to_end(key)
            else:
                # 添加新条目
                if len(self._cache) >= self.max_size:
                    # 移除最旧的条目
                    self._cache.popitem(last=False)
                self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def __len__(self) -> int:
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return {
            "size": len(self),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class TTLCache(BaseCache[T]):
    """
    TTL 缓存（带过期时间）
    
    所有条目都有默认过期时间，支持自动清理过期条目
    
    示例:
        ```python
        cache = TTLCache(default_ttl=300)  # 5 分钟过期
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=60)  # 1 分钟过期
        ```
    """
    
    def __init__(
        self,
        default_ttl: float = 300.0,
        max_size: int = 10000,
        cleanup_interval: float = 60.0,
    ):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
    
    def _cleanup_expired(self) -> int:
        """清理过期条目"""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return 0
        
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        self._last_cleanup = now
        return len(expired_keys)
    
    def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        with self._lock:
            self._cleanup_expired()
            
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            self._cleanup_expired()
            
            # 检查大小限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                # 移除最旧的条目
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
                del self._cache[oldest_key]
            
            ttl = ttl if ttl is not None else self.default_ttl
            expires_at = time.time() + ttl
            
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def __len__(self) -> int:
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)


class PersistentCache(BaseCache[T]):
    """
    持久化缓存
    
    将缓存存储到磁盘，支持应用重启后恢复
    
    示例:
        ```python
        cache = PersistentCache(cache_dir="./cache")
        cache.set("key1", {"data": "value"})
        # 重启后数据仍在
        value = cache.get("key1")
        ```
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: Optional[float] = None,
        use_json: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.use_json = use_json
        
        self._meta_file = self.cache_dir / "_meta.json"
        self._meta: Dict[str, Dict] = self._load_meta()
        self._lock = threading.RLock()
    
    def _load_meta(self) -> Dict[str, Dict]:
        """加载元数据"""
        if self._meta_file.exists():
            try:
                with open(self._meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存元数据失败: {e}")
        return {}
    
    def _save_meta(self) -> None:
        """保存元数据"""
        try:
            with open(self._meta_file, "w", encoding="utf-8") as f:
                json.dump(self._meta, f)
        except Exception as e:
            logger.warning(f"保存缓存元数据失败: {e}")
    
    def _key_to_filename(self, key: str) -> str:
        """将 key 转换为安全的文件名"""
        hash_val = hashlib.md5(key.encode()).hexdigest()
        return hash_val
    
    def _get_filepath(self, key: str) -> Path:
        """获取缓存文件路径"""
        filename = self._key_to_filename(key)
        ext = ".json" if self.use_json else ".pkl"
        return self.cache_dir / f"{filename}{ext}"
    
    def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        with self._lock:
            if key not in self._meta:
                return None
            
            meta = self._meta[key]
            
            # 检查过期
            if meta.get("expires_at"):
                if time.time() > meta["expires_at"]:
                    self.delete(key)
                    return None
            
            filepath = self._get_filepath(key)
            if not filepath.exists():
                del self._meta[key]
                self._save_meta()
                return None
            
            try:
                if self.use_json:
                    with open(filepath, "r", encoding="utf-8") as f:
                        return json.load(f)
                else:
                    with open(filepath, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                logger.warning(f"读取缓存失败 ({key}): {e}")
                return None
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            filepath = self._get_filepath(key)
            
            try:
                if self.use_json:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(value, f, ensure_ascii=False)
                else:
                    with open(filepath, "wb") as f:
                        pickle.dump(value, f)
                
                ttl = ttl if ttl is not None else self.default_ttl
                self._meta[key] = {
                    "created_at": time.time(),
                    "expires_at": time.time() + ttl if ttl else None,
                }
                self._save_meta()
                
            except Exception as e:
                logger.error(f"写入缓存失败 ({key}): {e}")
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key not in self._meta:
                return False
            
            filepath = self._get_filepath(key)
            try:
                if filepath.exists():
                    filepath.unlink()
            except Exception as e:
                logger.warning(f"删除缓存文件失败: {e}")
            
            del self._meta[key]
            self._save_meta()
            return True
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            for key in list(self._meta.keys()):
                self.delete(key)
            self._meta.clear()
            self._save_meta()
    
    def __len__(self) -> int:
        return len(self._meta)


class CacheManager:
    """
    统一缓存管理器
    
    支持多层缓存、命名空间和便捷的装饰器
    
    示例:
        ```python
        cache = CacheManager()
        
        # 手动使用
        cache.set("query:123", results, namespace="retrieval")
        results = cache.get("query:123", namespace="retrieval")
        
        # 装饰器
        @cache.cached(namespace="embedding", ttl=3600)
        def compute_embedding(text):
            return model.encode(text)
        ```
    """
    
    def __init__(
        self,
        default_backend: str = "lru",
        lru_size: int = 1000,
        ttl_default: float = 300.0,
        persistent_dir: Optional[Union[str, Path]] = None,
    ):
        self._backends: Dict[str, BaseCache] = {}
        self._namespaces: Dict[str, str] = {}  # namespace -> backend
        
        # 初始化默认后端
        self._backends["lru"] = LRUCache(max_size=lru_size)
        self._backends["ttl"] = TTLCache(default_ttl=ttl_default)
        
        if persistent_dir:
            self._backends["persistent"] = PersistentCache(cache_dir=persistent_dir)
        
        self._default_backend = default_backend
    
    def _get_backend(self, namespace: Optional[str] = None) -> BaseCache:
        """获取后端"""
        if namespace and namespace in self._namespaces:
            backend_name = self._namespaces[namespace]
        else:
            backend_name = self._default_backend
        return self._backends[backend_name]
    
    def _make_key(self, key: str, namespace: Optional[str] = None) -> str:
        """构建完整的 key"""
        if namespace:
            return f"{namespace}:{key}"
        return key
    
    def get(self, key: str, namespace: Optional[str] = None, default: T = None) -> Optional[T]:
        """获取缓存值"""
        backend = self._get_backend(namespace)
        full_key = self._make_key(key, namespace)
        value = backend.get(full_key)
        return value if value is not None else default
    
    def set(
        self,
        key: str,
        value: T,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """设置缓存值"""
        backend = self._get_backend(namespace)
        full_key = self._make_key(key, namespace)
        backend.set(full_key, value, ttl=ttl)
    
    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """删除缓存"""
        backend = self._get_backend(namespace)
        full_key = self._make_key(key, namespace)
        return backend.delete(full_key)
    
    def clear(self, namespace: Optional[str] = None) -> None:
        """清空缓存"""
        if namespace:
            backend = self._get_backend(namespace)
            # 只清空指定 namespace 的 key（简化实现：清空整个后端）
            backend.clear()
        else:
            for backend in self._backends.values():
                backend.clear()
    
    def register_namespace(self, namespace: str, backend: str) -> None:
        """注册命名空间到指定后端"""
        if backend not in self._backends:
            raise ValueError(f"Unknown backend: {backend}")
        self._namespaces[namespace] = backend
    
    def add_backend(self, name: str, backend: BaseCache) -> None:
        """添加自定义后端"""
        self._backends[name] = backend
    
    def cached(
        self,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        key_func: Optional[Callable[..., str]] = None,
    ):
        """
        缓存装饰器
        
        Args:
            namespace: 命名空间
            ttl: 过期时间
            key_func: 自定义 key 生成函数
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # 生成 cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # 默认：使用函数名和参数的哈希
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # 尝试获取缓存
                cached_value = self.get(cache_key, namespace=namespace)
                if cached_value is not None:
                    return cached_value
                
                # 计算并缓存
                result = func(*args, **kwargs)
                self.set(cache_key, result, namespace=namespace, ttl=ttl)
                return result
            
            return wrapper
        return decorator
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取所有后端的统计信息"""
        stats = {}
        for name, backend in self._backends.items():
            if hasattr(backend, "stats"):
                stats[name] = backend.stats
            else:
                stats[name] = {"size": len(backend)}
        return stats


# ============== 便捷函数 ==============

# 全局缓存实例
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def cache_get(key: str, namespace: str = "default") -> Optional[Any]:
    """获取缓存"""
    return get_cache().get(key, namespace=namespace)


def cache_set(key: str, value: Any, namespace: str = "default", ttl: float = None) -> None:
    """设置缓存"""
    get_cache().set(key, value, namespace=namespace, ttl=ttl)


def cached(namespace: str = "default", ttl: float = None):
    """缓存装饰器的便捷版本"""
    return get_cache().cached(namespace=namespace, ttl=ttl)
