"""
异步执行和性能优化工具

提供:
- 并发任务执行器
- 批处理优化
- 超时控制
- 重试机制
- 限流器
"""

import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any, Awaitable, Callable, Dict, Generic, Iterable, List, 
    Optional, Sequence, Tuple, TypeVar, Union
)

from loguru import logger


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult(Generic[T]):
    """任务执行结果"""
    value: Optional[T] = None
    error: Optional[Exception] = None
    duration: float = 0.0
    retries: int = 0
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    def get_or_raise(self) -> T:
        """获取值或抛出异常"""
        if self.error:
            raise self.error
        return self.value


@dataclass
class BatchResult(Generic[T]):
    """批处理结果"""
    results: List[TaskResult[T]] = field(default_factory=list)
    total_duration: float = 0.0
    
    @property
    def successful_results(self) -> List[T]:
        """成功的结果"""
        return [r.value for r in self.results if r.success and r.value is not None]
    
    @property
    def failed_results(self) -> List[Exception]:
        """失败的错误"""
        return [r.error for r in self.results if r.error is not None]
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if not self.results:
            return 0.0
        return len(self.successful_results) / len(self.results)


class RateLimiter:
    """
    令牌桶限流器
    
    控制请求速率，防止过载
    
    示例:
        ```python
        limiter = RateLimiter(rate=10, burst=20)  # 10 QPS, 最大突发 20
        
        async def make_request():
            await limiter.acquire()
            # ... 执行请求
        ```
    """
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: 每秒允许的请求数
            burst: 突发请求上限
        """
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """获取令牌（阻塞直到获取成功）"""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                
                # 计算需要等待的时间
                wait_time = (tokens - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
    
    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now
    
    @property
    def available_tokens(self) -> float:
        """当前可用令牌数"""
        self._refill()
        return self._tokens


class Semaphore:
    """
    异步信号量
    
    限制并发数量
    """
    
    def __init__(self, limit: int):
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
    
    async def __aenter__(self):
        await self._semaphore.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


class AsyncExecutor:
    """
    异步任务执行器
    
    提供并发执行、超时控制、重试机制等功能
    
    示例:
        ```python
        executor = AsyncExecutor(max_concurrency=10)
        
        # 并发执行多个任务
        results = await executor.run_concurrent([
            fetch_data(url1),
            fetch_data(url2),
            fetch_data(url3),
        ])
        
        # 带重试的执行
        result = await executor.run_with_retry(
            risky_operation,
            max_retries=3,
            retry_delay=1.0,
        )
        ```
    """
    
    def __init__(
        self,
        max_concurrency: int = 10,
        default_timeout: float = 30.0,
        rate_limit: Optional[float] = None,
    ):
        """
        Args:
            max_concurrency: 最大并发数
            default_timeout: 默认超时时间（秒）
            rate_limit: 速率限制（QPS），None 表示不限制
        """
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_limiter = RateLimiter(rate_limit, burst=max_concurrency) if rate_limit else None
        
        # 线程池（用于同步函数）
        self._thread_pool = ThreadPoolExecutor(max_workers=max_concurrency)
    
    async def run_concurrent(
        self,
        coroutines: Sequence[Awaitable[T]],
        timeout: Optional[float] = None,
        return_exceptions: bool = True,
    ) -> BatchResult[T]:
        """
        并发执行多个协程
        
        Args:
            coroutines: 协程列表
            timeout: 超时时间
            return_exceptions: 是否在结果中包含异常
            
        Returns:
            BatchResult 包含所有结果
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        async def wrapped_task(coro: Awaitable[T], index: int) -> TaskResult[T]:
            task_start = time.time()
            try:
                async with self._semaphore:
                    if self._rate_limiter:
                        await self._rate_limiter.acquire()
                    
                    result = await asyncio.wait_for(coro, timeout=timeout)
                    return TaskResult(
                        value=result,
                        duration=time.time() - task_start,
                    )
            except asyncio.TimeoutError:
                return TaskResult(
                    error=TimeoutError(f"Task {index} timed out after {timeout}s"),
                    duration=time.time() - task_start,
                )
            except Exception as e:
                return TaskResult(
                    error=e,
                    duration=time.time() - task_start,
                )
        
        tasks = [wrapped_task(coro, i) for i, coro in enumerate(coroutines)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return BatchResult(
            results=list(results),
            total_duration=time.time() - start_time,
        )
    
    async def run_with_retry(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
        retry_on: Optional[Tuple[type, ...]] = None,
    ) -> TaskResult[T]:
        """
        带重试的执行
        
        Args:
            coro_factory: 协程工厂函数（每次重试创建新协程）
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟
            exponential_backoff: 是否使用指数退避
            retry_on: 只在这些异常时重试
            
        Returns:
            TaskResult
        """
        last_error = None
        start_time = time.time()
        
        for attempt in range(max_retries + 1):
            try:
                async with self._semaphore:
                    if self._rate_limiter:
                        await self._rate_limiter.acquire()
                    
                    result = await asyncio.wait_for(
                        coro_factory(),
                        timeout=self.default_timeout
                    )
                    return TaskResult(
                        value=result,
                        duration=time.time() - start_time,
                        retries=attempt,
                    )
                    
            except Exception as e:
                last_error = e
                
                # 检查是否应该重试
                if retry_on and not isinstance(e, retry_on):
                    break
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                    logger.debug(f"重试 {attempt + 1}/{max_retries}，等待 {delay:.1f}s")
                    await asyncio.sleep(delay)
        
        return TaskResult(
            error=last_error,
            duration=time.time() - start_time,
            retries=max_retries,
        )
    
    async def run_in_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        在线程池中运行同步函数
        
        Args:
            func: 同步函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数返回值
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_pool,
            functools.partial(func, *args, **kwargs)
        )
    
    async def map_concurrent(
        self,
        func: Callable[[T], Awaitable[R]],
        items: Iterable[T],
        timeout: Optional[float] = None,
    ) -> BatchResult[R]:
        """
        并发映射函数到多个输入
        
        Args:
            func: 异步函数
            items: 输入列表
            timeout: 超时时间
            
        Returns:
            BatchResult
        """
        coroutines = [func(item) for item in items]
        return await self.run_concurrent(coroutines, timeout=timeout)
    
    def close(self):
        """关闭执行器"""
        self._thread_pool.shutdown(wait=False)


async def run_concurrent(
    coroutines: Sequence[Awaitable[T]],
    max_concurrency: int = 10,
    timeout: float = 30.0,
) -> BatchResult[T]:
    """
    并发执行协程的便捷函数
    
    Args:
        coroutines: 协程列表
        max_concurrency: 最大并发数
        timeout: 超时时间
        
    Returns:
        BatchResult
    """
    executor = AsyncExecutor(max_concurrency=max_concurrency, default_timeout=timeout)
    return await executor.run_concurrent(coroutines)


async def batch_process(
    items: Sequence[T],
    processor: Callable[[T], Awaitable[R]],
    batch_size: int = 10,
    max_concurrency: int = 5,
    delay_between_batches: float = 0.0,
) -> BatchResult[R]:
    """
    分批处理数据
    
    Args:
        items: 待处理数据
        processor: 处理函数
        batch_size: 每批大小
        max_concurrency: 每批内的并发数
        delay_between_batches: 批次间延迟
        
    Returns:
        BatchResult
    """
    all_results: List[TaskResult[R]] = []
    start_time = time.time()
    executor = AsyncExecutor(max_concurrency=max_concurrency)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.debug(f"处理批次 {i // batch_size + 1}/{(len(items) - 1) // batch_size + 1}")
        
        batch_result = await executor.map_concurrent(processor, batch)
        all_results.extend(batch_result.results)
        
        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)
    
    executor.close()
    
    return BatchResult(
        results=all_results,
        total_duration=time.time() - start_time,
    )


def with_timeout(timeout: float):
    """
    超时装饰器
    
    示例:
        ```python
        @with_timeout(5.0)
        async def slow_operation():
            await asyncio.sleep(10)
        ```
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
):
    """
    重试装饰器
    
    示例:
        ```python
        @with_retry(max_retries=3)
        async def flaky_operation():
            ...
        ```
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt if exponential_backoff else 1)
                        await asyncio.sleep(delay)
            
            raise last_error
        return wrapper
    return decorator


class AsyncBatcher:
    """
    异步批处理器
    
    自动将多个请求合并为批次执行
    
    示例:
        ```python
        async def batch_embed(texts):
            return model.encode(texts)
        
        batcher = AsyncBatcher(batch_embed, max_batch_size=32, max_wait_time=0.1)
        
        # 多个并发请求会被自动合并
        results = await asyncio.gather(
            batcher.submit("text1"),
            batcher.submit("text2"),
            batcher.submit("text3"),
        )
        ```
    """
    
    def __init__(
        self,
        batch_func: Callable[[List[T]], Awaitable[List[R]]],
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
    ):
        """
        Args:
            batch_func: 批处理函数
            max_batch_size: 最大批次大小
            max_wait_time: 最大等待时间（秒）
        """
        self.batch_func = batch_func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self._queue: List[Tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._processing = False
    
    async def submit(self, item: T) -> R:
        """提交单个请求"""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._queue.append((item, future))
            
            if len(self._queue) >= self.max_batch_size:
                asyncio.create_task(self._process_batch())
            elif not self._processing:
                self._processing = True
                asyncio.create_task(self._wait_and_process())
        
        return await future
    
    async def _wait_and_process(self):
        """等待后处理"""
        await asyncio.sleep(self.max_wait_time)
        await self._process_batch()
        self._processing = False
    
    async def _process_batch(self):
        """处理批次"""
        async with self._lock:
            if not self._queue:
                return
            
            batch = self._queue[:self.max_batch_size]
            self._queue = self._queue[self.max_batch_size:]
        
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]
        
        try:
            results = await self.batch_func(items)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


# ============== 便捷函数 ==============

async def gather_with_concurrency(
    limit: int,
    *coros: Awaitable[T],
) -> List[T]:
    """
    带并发限制的 gather
    
    Args:
        limit: 并发限制
        *coros: 协程
        
    Returns:
        结果列表
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[limited_coro(c) for c in coros])


def run_async(coro: Awaitable[T]) -> T:
    """
    在同步上下文中运行异步代码
    
    Args:
        coro: 协程
        
    Returns:
        协程返回值
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已有事件循环在运行，创建新的
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
