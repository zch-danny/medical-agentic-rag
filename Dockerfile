# Medical Embedding RAG Service
FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# 创建数据目录
RUN mkdir -p data/documents data/parsed data/cache/embeddings logs

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.health import HealthChecker; import sys; checker = HealthChecker(); sys.exit(0 if checker.readiness() else 1)"

# 默认命令
CMD ["python", "scripts/search.py"]
