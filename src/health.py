"""
健康检查模块 - 用于服务部署监控
"""
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class HealthStatus:
    """单个服务的健康状态"""

    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """系统整体健康状态"""

    healthy: bool
    services: List[HealthStatus] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "timestamp": self.timestamp,
            "services": [
                {
                    "name": s.name,
                    "healthy": s.healthy,
                    "latency_ms": s.latency_ms,
                    "message": s.message,
                    "details": s.details,
                }
                for s in self.services
            ],
        }


class HealthChecker:
    """
    服务健康检查器

    检查:
    - Milvus 连接
    - Embedding 模型
    - Reranker 模型
    - LLM API
    """

    def __init__(
        self,
        milvus_uri: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        reranker_model_path: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        self.milvus_uri = milvus_uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        self.embedding_model_path = embedding_model_path or os.getenv("EMBEDDING_MODEL_PATH")
        self.reranker_model_path = reranker_model_path or os.getenv("RERANKER_MODEL_PATH")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    def check_milvus(self) -> HealthStatus:
        """检查 Milvus 连接"""
        start = time.time()
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=self.milvus_uri)
            # 简单操作验证连接
            collections = client.list_collections()
            client.close()

            latency = (time.time() - start) * 1000
            return HealthStatus(
                name="milvus",
                healthy=True,
                latency_ms=latency,
                message="连接正常",
                details={"uri": self.milvus_uri, "collections_count": len(collections)},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Milvus 健康检查失败: {e}")
            return HealthStatus(
                name="milvus",
                healthy=False,
                latency_ms=latency,
                message=f"连接失败: {str(e)}",
                details={"uri": self.milvus_uri},
            )

    def check_embedding_model(self) -> HealthStatus:
        """检查 Embedding 模型"""
        start = time.time()
        try:
            if not self.embedding_model_path:
                return HealthStatus(
                    name="embedding",
                    healthy=False,
                    message="EMBEDDING_MODEL_PATH 未配置",
                )

            import os as os_module

            if not os_module.path.exists(self.embedding_model_path):
                return HealthStatus(
                    name="embedding",
                    healthy=False,
                    message=f"模型路径不存在: {self.embedding_model_path}",
                )

            # 检查关键文件
            required_files = ["config.json", "tokenizer.json"]
            missing = [
                f for f in required_files if not os_module.path.exists(os_module.path.join(self.embedding_model_path, f))
            ]

            latency = (time.time() - start) * 1000
            if missing:
                return HealthStatus(
                    name="embedding",
                    healthy=False,
                    latency_ms=latency,
                    message=f"缺少文件: {missing}",
                    details={"path": self.embedding_model_path},
                )

            return HealthStatus(
                name="embedding",
                healthy=True,
                latency_ms=latency,
                message="模型文件存在",
                details={"path": self.embedding_model_path},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Embedding 模型检查失败: {e}")
            return HealthStatus(
                name="embedding",
                healthy=False,
                latency_ms=latency,
                message=f"检查失败: {str(e)}",
            )

    def check_reranker_model(self) -> HealthStatus:
        """检查 Reranker 模型"""
        start = time.time()
        try:
            if not self.reranker_model_path:
                return HealthStatus(
                    name="reranker",
                    healthy=False,
                    message="RERANKER_MODEL_PATH 未配置",
                )

            import os as os_module

            if not os_module.path.exists(self.reranker_model_path):
                return HealthStatus(
                    name="reranker",
                    healthy=False,
                    message=f"模型路径不存在: {self.reranker_model_path}",
                )

            # 检查关键文件
            required_files = ["config.json", "tokenizer.json"]
            missing = [
                f for f in required_files if not os_module.path.exists(os_module.path.join(self.reranker_model_path, f))
            ]

            latency = (time.time() - start) * 1000
            if missing:
                return HealthStatus(
                    name="reranker",
                    healthy=False,
                    latency_ms=latency,
                    message=f"缺少文件: {missing}",
                    details={"path": self.reranker_model_path},
                )

            return HealthStatus(
                name="reranker",
                healthy=True,
                latency_ms=latency,
                message="模型文件存在",
                details={"path": self.reranker_model_path},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Reranker 模型检查失败: {e}")
            return HealthStatus(
                name="reranker",
                healthy=False,
                latency_ms=latency,
                message=f"检查失败: {str(e)}",
            )

    def check_llm_api(self) -> HealthStatus:
        """检查 LLM API 连接"""
        start = time.time()
        try:
            if not self.llm_api_key:
                return HealthStatus(
                    name="llm",
                    healthy=False,
                    message="LLM_API_KEY 未配置",
                )

            from openai import OpenAI

            client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
            # 发送最小请求验证连接
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )

            latency = (time.time() - start) * 1000
            return HealthStatus(
                name="llm",
                healthy=True,
                latency_ms=latency,
                message="API 连接正常",
                details={"base_url": self.llm_base_url},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"LLM API 检查失败: {e}")
            return HealthStatus(
                name="llm",
                healthy=False,
                latency_ms=latency,
                message=f"API 连接失败: {str(e)}",
                details={"base_url": self.llm_base_url},
            )

    def check_all(self, include_llm: bool = False) -> SystemHealth:
        """
        执行所有健康检查

        Args:
            include_llm: 是否包含 LLM API 检查 (会产生少量费用)

        Returns:
            SystemHealth 系统整体健康状态
        """
        services = [
            self.check_milvus(),
            self.check_embedding_model(),
            self.check_reranker_model(),
        ]

        if include_llm:
            services.append(self.check_llm_api())

        all_healthy = all(s.healthy for s in services)

        return SystemHealth(healthy=all_healthy, services=services)

    def liveness(self) -> bool:
        """
        Kubernetes liveness probe
        简单检查服务是否存活
        """
        return True

    def readiness(self) -> bool:
        """
        Kubernetes readiness probe
        检查服务是否准备好接收请求
        """
        # 只检查 Milvus，因为模型加载可能很慢
        milvus_status = self.check_milvus()
        return milvus_status.healthy


# CLI 入口
if __name__ == "__main__":
    import json
    import sys

    checker = HealthChecker()
    health = checker.check_all(include_llm="--include-llm" in sys.argv)

    print(json.dumps(health.to_dict(), indent=2, ensure_ascii=False))
    sys.exit(0 if health.healthy else 1)
