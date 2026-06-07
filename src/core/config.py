"""
配置管理模块 (Configuration Management)

本模块负责加载和管理多Agent系统的全局配置。核心功能：
1. 定义所有子系统的配置模型（使用 Pydantic BaseModel 进行数据验证）
2. 从 YAML 文件加载配置，支持 ${ENV_VAR} 环境变量替换
3. 提供全局单例访问模式，确保配置一致性

配置层级结构：
  Settings (顶层)
  ├── system: SystemConfig      - 系统基本设置（名称、版本、日志级别）
  ├── llm: LLMConfig            - LLM 提供商配置（模型、API密钥、温度等）
  ├── embedding: EmbeddingConfig - 文本嵌入模型配置（bge-m3等）
  ├── vector_store: VectorStoreConfig - 向量数据库配置（ChromaDB）
  ├── rag: RAGConfig            - RAG 检索配置（分块大小、top_k、重排序）
  ├── agents: AgentsConfig      - Agent 运行时配置（并发数、策略、记忆）
  └── mcp: MCPConfig            - MCP 服务器配置（文件系统、Web等）
"""
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """系统级配置：定义整个多Agent系统的基本属性。

    Attributes:
        name: 系统名称，用于日志和标识
        version: 系统版本号，遵循语义化版本规范
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    name: str = "Multi-Agent System"
    version: str = "0.1.0"
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    """大语言模型 (LLM) 配置：定义与 LLM API 交互的参数。

    当前支持 Anthropic Claude 作为默认提供商。通过 api_key 字段
    或环境变量 ANTHROPIC_API_KEY 进行身份验证。

    Attributes:
        provider: LLM 提供商标识符（当前支持 "anthropic"）
        model: 模型ID，如 "claude-sonnet-4-20250514"
        api_key: API 密钥，支持 ${ANTHROPIC_API_KEY} 环境变量替换
        temperature: 生成温度 (0.0-1.0)，越高越随机/创造性
        max_tokens: 单次生成的最大 token 数
        timeout: API 请求超时时间（秒）
    """
    provider: str = "openai"
    model: str = "qwen3.7-plus"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


class EmbeddingConfig(BaseModel):
    """文本嵌入模型配置：用于将文本转换为向量表示。

    默认使用 BAAI/bge-m3 模型，这是一个多语言嵌入模型，
    支持中英文等多种语言的语义相似度计算。

    Attributes:
        model: 嵌入模型名称（HuggingFace 模型ID）
        device: 计算设备 ("cpu" 或 "cuda")
        batch_size: 批量编码时的批次大小，影响内存使用和速度
    """
    model: str = "qwen3-rerank"
    device: str = "cpu"
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    """向量数据库配置：用于存储和检索文档嵌入向量。

    默认使用 ChromaDB 作为向量存储后端，它是一个轻量级的
    开源向量数据库，适合本地开发和中小规模应用。

    Attributes:
        backend: 向量存储后端类型（当前支持 "chroma"）
        persist_dir: 数据持久化目录，ChromaDB 会在此存储索引数据
        collection: 集合名称，用于隔离不同的文档集
    """
    backend: str = "chroma"
    persist_dir: str = "./data/chroma"
    collection: str = "default"


class RAGConfig(BaseModel):
    """RAG (检索增强生成) 配置：控制文档分块和检索行为。

    RAG 流程：文档 → 分块(chunker) → 嵌入(embedder) → 存储(vector_store)
            查询 → 嵌入 → 向量检索 → (可选)重排序 → 返回结果

    Attributes:
        chunk_size: 文档分块大小（字符数），较小的块提高精度但增加存储
        chunk_overlap: 相邻块的重叠字符数，确保上下文连续性
        retrieval_top_k: 检索返回的最相关文档数量
        rerank: 是否启用重排序（使用交叉编码器对结果精排）
        rerank_model: 重排序模型名称
    """
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    rerank: bool = True
    rerank_model: str = "qwen3-omni-flash"


class MemoryConfig(BaseModel):
    """Agent 记忆系统配置。

    记忆分为两层：
    - 短期记忆 (ShortTermMemory): 基于 deque 的会话级记忆，自动淘汰旧消息
    - 长期记忆 (LongTermMemory): 基于 SQLite 的持久化记忆，支持语义检索

    Attributes:
        short_term_max: 短期记忆最大保留的对话轮数
        long_term_backend: 长期记忆存储后端（当前支持 "sqlite"）
    """
    short_term_max: int = 20
    long_term_backend: str = "sqlite"


class AgentsConfig(BaseModel):
    """Agent 运行时配置：控制 Agent 的并发和策略行为。

    Attributes:
        max_concurrent: 最大并发 Agent 数量，防止资源耗尽
        default_strategy: 默认执行策略 ("react"/"cot"/"plan_execute")
        memory: 记忆系统配置
    """
    max_concurrent: int = 5
    default_strategy: str = "react"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class MCPServerConfig(BaseModel):
    """单个 MCP 服务器配置。

    MCP (Model Context Protocol) 是 Agent 与外部工具/数据源通信的标准化协议。
    每个 MCP 服务器提供一组工具和资源。

    Attributes:
        name: 服务器唯一标识名称
        enabled: 是否启用该服务器
        root_dir: 文件系统服务器的根目录（仅 filesystem 类型需要）
        connection_string: 服务器连接字符串（远程服务器使用）
        allowed_domains: Web 服务器允许访问的域名列表（安全限制）
    """
    name: str
    enabled: bool = True
    root_dir: str = ""
    connection_string: str = ""
    allowed_domains: list[str] = Field(default_factory=list)


class MCPConfig(BaseModel):
    """MCP 模块配置：包含所有 MCP 服务器的配置列表。

    Attributes:
        servers: MCP 服务器配置列表
    """
    servers: list[MCPServerConfig] = Field(default_factory=list)


class Settings(BaseModel):
    """系统顶层配置容器：聚合所有子模块的配置。

    这是整个多Agent系统的配置根对象，通过 load_config() 从 YAML 文件加载，
    然后通过 get_settings() 全局访问。

    使用示例：
        settings = load_config(Path("config.yaml"))
        model = settings.llm.model  # 获取 LLM 模型名称
        top_k = settings.rag.retrieval_top_k  # 获取 RAG 检索数量
    """
    system: SystemConfig = Field(default_factory=SystemConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


# 环境变量匹配正则：匹配 ${VAR_NAME} 格式的占位符
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """递归展开配置值中的环境变量占位符。

    支持在 YAML 配置中使用 ${ENV_VAR} 引用环境变量，
    例如: api_key: "${ANTHROPIC_API_KEY}" 会在运行时替换为实际的环境变量值。

    该函数递归处理嵌套的 dict 和 list 结构。

    Args:
        value: 配置值，可以是 str/dict/list/其他基本类型

    Returns:
        替换后的配置值。如果环境变量不存在，保留原始占位符。
    """
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


# 全局配置单例：确保整个系统使用同一份配置
_settings: Settings | None = None


def load_config(path: Path) -> Settings:
    """从 YAML 文件加载系统配置。

    加载流程：
    1. 读取 YAML 文件
    2. 递归展开所有 ${ENV_VAR} 环境变量
    3. 使用 Pydantic 验证并构建 Settings 对象
    4. 存储为全局单例

    Args:
        path: YAML 配置文件路径

    Returns:
        解析后的 Settings 对象

    Raises:
        FileNotFoundError: 配置文件不存在
        pydantic.ValidationError: 配置格式不符合定义
    """
    global _settings
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)
    _settings = Settings(**expanded)
    return _settings


def get_settings() -> Settings:
    """获取当前全局配置单例。

    Returns:
        已加载的 Settings 对象

    Raises:
        RuntimeError: 如果配置尚未加载（未调用 load_config）
    """
    if _settings is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _settings


def reset_settings():
    """重置全局配置单例（主要用于测试）。

    调用后需要重新 load_config() 才能使用 get_settings()。
    """
    global _settings
    _settings = None
