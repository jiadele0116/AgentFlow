# tests/conftest.py
import pytest
import asyncio


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal config file for testing."""
    config_content = """
system:
  name: "Test System"
  version: "0.1.0"
  log_level: DEBUG

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key
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
    return config_file
