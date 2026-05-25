import os
import pytest
from pathlib import Path


def test_load_config_from_file(sample_config):
    """Config loads YAML file and returns typed settings."""
    from src.core.config import load_config

    settings = load_config(sample_config)
    assert settings.system.name == "Test System"
    assert settings.system.log_level == "DEBUG"
    assert settings.llm.provider == "anthropic"
    assert settings.llm.model == "claude-sonnet-4-20250514"


def test_config_expands_env_vars(tmp_path):
    """Config expands ${ENV_VAR} patterns in values."""
    os.environ["TEST_API_KEY"] = "sk-test-123"
    config_content = """
system:
  name: "Test"
  version: "0.1.0"
  log_level: INFO
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_API_KEY}
  temperature: 0.7
  max_tokens: 4096
  timeout: 60
embedding:
  model: BAAI/bge-m3
  device: cpu
  batch_size: 32
vector_store:
  backend: chroma
  persist_dir: ./data/chroma
  collection: default
rag:
  chunk_size: 512
  chunk_overlap: 64
  retrieval_top_k: 5
  rerank: false
agents:
  max_concurrent: 5
  default_strategy: react
  memory:
    short_term_max: 20
    long_term_backend: sqlite
mcp:
  servers:
    - name: filesystem
      enabled: true
      root_dir: ./workspace
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    from src.core.config import load_config
    settings = load_config(config_file)
    assert settings.llm.api_key == "sk-test-123"
    del os.environ["TEST_API_KEY"]


def test_config_singleton(sample_config):
    """get_settings returns the same instance on repeated calls."""
    from src.core.config import load_config, get_settings, reset_settings

    reset_settings()
    load_config(sample_config)
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    reset_settings()


def test_config_missing_file():
    """load_config raises FileNotFoundError for missing file."""
    from src.core.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_config_default_values(sample_config):
    """Config provides sensible defaults for optional fields."""
    from src.core.config import load_config

    settings = load_config(sample_config)
    assert settings.rag.chunk_size == 512
    assert settings.rag.chunk_overlap == 64
    assert settings.agents.max_concurrent == 5
