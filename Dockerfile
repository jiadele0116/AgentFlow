# ============================================================
# Multi-Agent System Dockerfile
# ============================================================
# 多阶段构建：builder 安装依赖 → runtime 仅复制运行时文件
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# 安装系统依赖（编译部分 Python 包需要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖定义，利用 Docker 缓存层
COPY pyproject.toml ./

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装依赖（不安装项目本身，后面再装）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# 创建非 root 用户
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 从 builder 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制源码和配置
COPY src/ ./src/
COPY config.yaml ./

# 创建数据目录
RUN mkdir -p data/chroma data/workspace && \
    chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 8880

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8880/health'); assert r.status_code==200" || exit 1

# 启动命令
CMD ["python", "-m", "src.main"]
