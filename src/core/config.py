import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    name: str = "Multi-Agent System"
    version: str = "0.1.0"
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    backend: str = "chroma"
    persist_dir: str = "./data/chroma"
    collection: str = "default"


class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


class MemoryConfig(BaseModel):
    short_term_max: int = 20
    long_term_backend: str = "sqlite"


class AgentsConfig(BaseModel):
    max_concurrent: int = 5
    default_strategy: str = "react"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class MCPServerConfig(BaseModel):
    name: str
    enabled: bool = True
    root_dir: str = ""
    connection_string: str = ""
    allowed_domains: list[str] = Field(default_factory=list)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class Settings(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} patterns in config values."""
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


_settings: Settings | None = None


def load_config(path: Path) -> Settings:
    """Load configuration from YAML file with env var expansion."""
    global _settings
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)
    _settings = Settings(**expanded)
    return _settings


def get_settings() -> Settings:
    """Get the current settings singleton."""
    if _settings is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _settings


def reset_settings():
    """Reset the settings singleton (for testing)."""
    global _settings
    _settings = None
