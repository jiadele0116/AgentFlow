# Phase 2: Capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Tools (registry, engine, sandbox, built-in tools) and RAG (processor, chunker, embedder, ChromaDB, retrieval) modules — the capabilities that agents will consume.

**Architecture:** Tools provides a registry + execution engine for callable operations. RAG provides document ingestion, embedding, storage, and hybrid retrieval. Both depend only on Core and LLM from Phase 1.

**Tech Stack:** Python 3.11+, ChromaDB, sentence-transformers, rank-bm25, RestrictedPython, pypdf, python-docx, beautifulsoup4, httpx

---

## File Structure

```
src/tools/
├── __init__.py
├── registry.py          # Tool registration decorator and ToolRegistry
├── engine.py            # ToolEngine with validation, timeout, retry, audit
├── sandbox.py           # RestrictedPython sandbox for code execution
└── builtin/
    ├── __init__.py      # Auto-registers all built-in tools
    ├── file_ops.py      # read_file, write_file, search_files
    ├── code_exec.py     # run_python, run_shell, lint_code
    ├── network.py       # fetch_url, search_web
    ├── data_ops.py      # parse_json, csv_to_dict, json_to_csv
    └── rag_ops.py       # search_knowledge, index_document

src/rag/
├── __init__.py
├── processor.py         # DocumentProcessor orchestrator
├── chunker.py           # Chunking strategies (recursive, fixed_size, etc.)
├── embedder.py          # sentence-transformers wrapper for bge-m3
├── vector_store.py      # VectorStore ABC
├── stores/
│   └── chroma_store.py  # ChromaDB implementation
├── retrieval.py         # Hybrid retrieval with RRF fusion
└── query_rewrite.py     # LLM-based query rewriting

tests/
├── test_tools/
│   ├── __init__.py
│   ├── test_registry.py
│   ├── test_engine.py
│   ├── test_sandbox.py
│   └── test_builtin/
│       ├── __init__.py
│       ├── test_file_ops.py
│       ├── test_data_ops.py
│       └── test_network.py
├── test_rag/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_chroma_store.py
│   └── test_retrieval.py
```

---

## Task 1: Tool Registry

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/registry.py`
- Create: `tests/test_tools/__init__.py`
- Create: `tests/test_tools/test_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_registry.py
import pytest


def test_register_tool_decorator():
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register(
        name="greet",
        description="Say hello",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    )
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert "greet" in registry.list_names()
    tool = registry.get("greet")
    assert tool.name == "greet"
    assert tool.description == "Say hello"


def test_registry_list_tools_returns_schemas():
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    async def add(a: float, b: float) -> float:
        return a + b

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "add"
    assert tools[0]["description"] == "Add two numbers"
    assert "properties" in tools[0]["input_schema"]


def test_registry_get_missing_tool():
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_registry_unregister():
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register(name="temp", description="temp", parameters={})
    async def temp():
        pass

    assert "temp" in registry.list_names()
    registry.unregister("temp")
    assert "temp" not in registry.list_names()


def test_registry_multiple_tools():
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register(name="a", description="tool a", parameters={})
    async def a():
        pass

    @registry.register(name="b", description="tool b", parameters={})
    async def b():
        pass

    assert set(registry.list_names()) == {"a", "b"}
    assert len(registry.list_tools()) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_registry.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement registry**

```python
# src/tools/__init__.py
```

```python
# src/tools/registry.py
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[Any]]
    timeout: int = 30
    retry: int = 0
    requires_confirmation: bool = False


class ToolRegistry:
    """Central registry for all tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        timeout: int = 30,
        retry: int = 0,
        requires_confirmation: bool = False,
    ):
        """Decorator to register an async function as a tool."""
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
                timeout=timeout,
                retry=retry,
                requires_confirmation=requires_confirmation,
            )
            self._tools[name] = tool
            return func
        return decorator

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_tools(self) -> list[dict[str, Any]]:
        """List all tools as JSON Schema descriptions for LLM consumption."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def unregister(self, name: str):
        """Remove a tool from the registry."""
        self._tools.pop(name, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_registry.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/__init__.py src/tools/registry.py tests/test_tools/__init__.py tests/test_tools/test_registry.py
git commit -m "feat: ToolRegistry with decorator-based registration and JSON Schema listing"
```

---

## Task 2: Tool Execution Engine

**Files:**
- Create: `src/tools/engine.py`
- Create: `tests/test_tools/test_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_engine.py
import asyncio
import pytest
from src.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_engine_execute_simple_tool():
    from src.tools.engine import ToolEngine

    registry = ToolRegistry()

    @registry.register(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    async def add(a: float, b: float) -> float:
        return a + b

    engine = ToolEngine(registry)
    result = await engine.execute("add", {"a": 2, "b": 3})
    assert result.success is True
    assert result.data == 5.0
    assert result.duration_ms > 0


@pytest.mark.asyncio
async def test_engine_execute_with_timeout():
    from src.tools.engine import ToolEngine

    registry = ToolRegistry()

    @registry.register(
        name="slow",
        description="Slow tool",
        parameters={},
        timeout=1,
    )
    async def slow():
        await asyncio.sleep(10)
        return "done"

    engine = ToolEngine(registry)
    result = await engine.execute("slow", {})
    assert result.success is False
    assert "timeout" in result.error.lower() or "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_engine_execute_missing_tool():
    from src.tools.engine import ToolEngine

    registry = ToolRegistry()
    engine = ToolEngine(registry)
    result = await engine.execute("nonexistent", {})
    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_engine_execute_with_retry():
    from src.tools.engine import ToolEngine
    from unittest.mock import AsyncMock

    registry = ToolRegistry()
    call_count = 0

    @registry.register(
        name="flaky",
        description="Fails first time",
        parameters={},
        retry=2,
    )
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("transient error")
        return "success"

    engine = ToolEngine(registry)
    result = await engine.execute("flaky", {})
    assert result.success is True
    assert result.data == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_engine_list_tools():
    from src.tools.engine import ToolEngine

    registry = ToolRegistry()

    @registry.register(name="a", description="tool a", parameters={"type": "object"})
    async def a():
        pass

    engine = ToolEngine(registry)
    tools = engine.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_engine.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement engine**

```python
# src/tools/engine.py
import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.tools.registry import ToolRegistry, Tool


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str = ""
    duration_ms: float = 0


class ToolEngine:
    """Executes tools with validation, timeout, retry, and audit logging."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given parameters."""
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")

        start = time.monotonic()
        last_error = ""
        attempts = 1 + tool.retry

        for attempt in range(attempts):
            try:
                result = await asyncio.wait_for(
                    tool.handler(**params),
                    timeout=tool.timeout,
                )
                duration = (time.monotonic() - start) * 1000
                return ToolResult(success=True, data=result, duration_ms=duration)

            except asyncio.TimeoutError:
                duration = (time.monotonic() - start) * 1000
                return ToolResult(
                    success=False,
                    error=f"Tool '{name}' timed out after {tool.timeout}s",
                    duration_ms=duration,
                )

            except Exception as e:
                last_error = str(e)
                if attempt < attempts - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))

        duration = (time.monotonic() - start) * 1000
        return ToolResult(success=False, error=last_error, duration_ms=duration)

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools as JSON Schema descriptions."""
        return self._registry.list_tools()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_engine.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/engine.py tests/test_tools/test_engine.py
git commit -m "feat: ToolEngine with timeout, retry, and result formatting"
```

---

## Task 3: Built-in File Tools

**Files:**
- Create: `src/tools/builtin/__init__.py`
- Create: `src/tools/builtin/file_ops.py`
- Create: `tests/test_tools/test_builtin/__init__.py`
- Create: `tests/test_tools/test_builtin/test_file_ops.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_builtin/test_file_ops.py
import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_read_file(tmp_path):
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.file_ops import register_file_tools

    registry = ToolRegistry()
    register_file_tools(registry)

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    tool = registry.get("read_file")
    result = await tool.handler(path=str(test_file))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_write_file(tmp_path):
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.file_ops import register_file_tools

    registry = ToolRegistry()
    register_file_tools(registry)

    test_file = tmp_path / "output.txt"
    tool = registry.get("write_file")
    result = await tool.handler(path=str(test_file), content="new content")
    assert result is True
    assert test_file.read_text() == "new content"


@pytest.mark.asyncio
async def test_search_files(tmp_path):
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.file_ops import register_file_tools

    registry = ToolRegistry()
    register_file_tools(registry)

    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "c.txt").write_text("c")

    tool = registry.get("search_files")
    result = await tool.handler(pattern="*.py", directory=str(tmp_path))
    assert len(result) == 2
    assert all(p.endswith(".py") for p in result)


@pytest.mark.asyncio
async def test_read_file_not_found():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.file_ops import register_file_tools

    registry = ToolRegistry()
    register_file_tools(registry)

    tool = registry.get("read_file")
    with pytest.raises(FileNotFoundError):
        await tool.handler(path="/nonexistent/file.txt")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_builtin/test_file_ops.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement file tools**

```python
# src/tools/builtin/__init__.py
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools


def register_all_builtin_tools(registry: ToolRegistry):
    """Register all built-in tools."""
    register_file_tools(registry)
```

```python
# src/tools/builtin/file_ops.py
import fnmatch
import os
from pathlib import Path

from src.tools.registry import ToolRegistry


def register_file_tools(registry: ToolRegistry):
    """Register file operation tools."""

    @registry.register(
        name="read_file",
        description="Read the contents of a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
            },
            "required": ["path"],
        },
    )
    async def read_file(path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    @registry.register(
        name="write_file",
        description="Write content to a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    )
    async def write_file(path: str, content: str) -> bool:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True

    @registry.register(
        name="search_files",
        description="Search for files matching a glob pattern",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. *.py)"},
                "directory": {"type": "string", "description": "Directory to search in"},
            },
            "required": ["pattern", "directory"],
        },
    )
    async def search_files(pattern: str, directory: str) -> list[str]:
        results = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    results.append(os.path.join(root, f))
        return sorted(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_builtin/test_file_ops.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/builtin/__init__.py src/tools/builtin/file_ops.py tests/test_tools/test_builtin/__init__.py tests/test_tools/test_builtin/test_file_ops.py
git commit -m "feat: built-in file tools (read_file, write_file, search_files)"
```

---

## Task 4: Built-in Data Tools

**Files:**
- Create: `src/tools/builtin/data_ops.py`
- Create: `tests/test_tools/test_builtin/test_data_ops.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_builtin/test_data_ops.py
import pytest


@pytest.mark.asyncio
async def test_parse_json():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.data_ops import register_data_tools

    registry = ToolRegistry()
    register_data_tools(registry)

    tool = registry.get("parse_json")
    result = await tool.handler(text='{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}


@pytest.mark.asyncio
async def test_csv_to_dict():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.data_ops import register_data_tools

    registry = ToolRegistry()
    register_data_tools(registry)

    csv_text = "name,age\nAlice,30\nBob,25"
    tool = registry.get("csv_to_dict")
    result = await tool.handler(csv_text=csv_text)
    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["age"] == "25"


@pytest.mark.asyncio
async def test_json_to_csv():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.data_ops import register_data_tools

    registry = ToolRegistry()
    register_data_tools(registry)

    data = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
    tool = registry.get("json_to_csv")
    result = await tool.handler(data=data)
    assert "name,age" in result
    assert "Alice,30" in result
    assert "Bob,25" in result


@pytest.mark.asyncio
async def test_parse_json_invalid():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.data_ops import register_data_tools

    registry = ToolRegistry()
    register_data_tools(registry)

    tool = registry.get("parse_json")
    with pytest.raises(Exception):
        await tool.handler(text="not json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_builtin/test_data_ops.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement data tools**

```python
# src/tools/builtin/data_ops.py
import csv
import io
import json
from typing import Any

from src.tools.registry import ToolRegistry


def register_data_tools(registry: ToolRegistry):
    """Register data processing tools."""

    @registry.register(
        name="parse_json",
        description="Parse a JSON string into a Python object",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "JSON string to parse"},
            },
            "required": ["text"],
        },
    )
    async def parse_json(text: str) -> Any:
        return json.loads(text)

    @registry.register(
        name="csv_to_dict",
        description="Parse CSV text into a list of dictionaries",
        parameters={
            "type": "object",
            "properties": {
                "csv_text": {"type": "string", "description": "CSV text to parse"},
            },
            "required": ["csv_text"],
        },
    )
    async def csv_to_dict(csv_text: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(csv_text))
        return list(reader)

    @registry.register(
        name="json_to_csv",
        description="Convert a list of dictionaries to CSV text",
        parameters={
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of dictionaries to convert",
                },
            },
            "required": ["data"],
        },
    )
    async def json_to_csv(data: list[dict[str, str]]) -> str:
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_builtin/test_data_ops.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Update builtin/__init__.py**

```python
# src/tools/builtin/__init__.py
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools
from src.tools.builtin.data_ops import register_data_tools


def register_all_builtin_tools(registry: ToolRegistry):
    """Register all built-in tools."""
    register_file_tools(registry)
    register_data_tools(registry)
```

- [ ] **Step 6: Commit**

```bash
git add src/tools/builtin/data_ops.py src/tools/builtin/__init__.py tests/test_tools/test_builtin/test_data_ops.py
git commit -m "feat: built-in data tools (parse_json, csv_to_dict, json_to_csv)"
```

---

## Task 5: Code Execution Sandbox

**Files:**
- Create: `src/tools/builtin/code_exec.py`
- Create: `src/tools/sandbox.py`
- Create: `tests/test_tools/test_builtin/test_code_exec.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_builtin/test_code_exec.py
import pytest


@pytest.mark.asyncio
async def test_run_python_simple():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.code_exec import register_code_tools

    registry = ToolRegistry()
    register_code_tools(registry, workspace_dir="/tmp")

    tool = registry.get("run_python")
    result = await tool.handler(code="print(2 + 3)", timeout=10)
    assert result.success is True
    assert "5" in result.output


@pytest.mark.asyncio
async def test_run_python_with_output():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.code_exec import register_code_tools

    registry = ToolRegistry()
    register_code_tools(registry, workspace_dir="/tmp")

    tool = registry.get("run_python")
    result = await tool.handler(code="x = [i**2 for i in range(5)]\nprint(x)", timeout=10)
    assert result.success is True
    assert "[0, 1, 4, 9, 16]" in result.output


@pytest.mark.asyncio
async def test_run_python_syntax_error():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.code_exec import register_code_tools

    registry = ToolRegistry()
    register_code_tools(registry, workspace_dir="/tmp")

    tool = registry.get("run_python")
    result = await tool.handler(code="def foo(", timeout=10)
    assert result.success is False
    assert "SyntaxError" in result.error or "syntax" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_builtin/test_code_exec.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement sandbox and code tools**

```python
# src/tools/sandbox.py
import ast
import sys
import io
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    return_value: Any = None


# Dangerous modules/functions to block
BLOCKED_NAMES = {
    "os.system", "subprocess", "subprocess.call", "subprocess.run",
    "subprocess.Popen", "shutil.rmtree", "shutil.move",
}

BLOCKED_IMPORTS = {"subprocess", "shutil", "ctypes", "importlib"}


def _check_code_safety(code: str) -> str | None:
    """Check if code is safe to execute. Returns error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # Let the actual execution handle syntax errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BLOCKED_IMPORTS:
                    return f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_IMPORTS:
                return f"Blocked import from: {node.module}"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                full = f"{node.value.id}.{node.attr}"
                if full in BLOCKED_NAMES:
                    return f"Blocked call: {full}"

    return None


def execute_python(code: str, timeout: int = 30) -> ExecutionResult:
    """Execute Python code in a restricted environment."""
    safety_error = _check_code_safety(code)
    if safety_error:
        return ExecutionResult(success=False, error=safety_error)

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    try:
        sys.stdout = captured_out
        sys.stderr = captured_err

        # Restricted globals
        safe_globals = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "any": any,
                "all": all,
                "True": True,
                "False": False,
                "None": None,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "Exception": Exception,
                "json": __import__("json"),
                "math": __import__("math"),
                "re": __import__("re"),
                "collections": __import__("collections"),
            }
        }

        exec(code, safe_globals)

        output = captured_out.getvalue()
        return ExecutionResult(success=True, output=output)

    except Exception:
        error = traceback.format_exc()
        return ExecutionResult(success=False, error=error)

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
```

```python
# src/tools/builtin/code_exec.py
from dataclasses import dataclass
from src.tools.registry import ToolRegistry
from src.tools.sandbox import execute_python, ExecutionResult


def register_code_tools(registry: ToolRegistry, workspace_dir: str = "."):
    """Register code execution tools."""

    @registry.register(
        name="run_python",
        description="Execute Python code in a sandboxed environment",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["code"],
        },
        timeout=30,
    )
    async def run_python(code: str, timeout: int = 30) -> ExecutionResult:
        return execute_python(code, timeout=timeout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_builtin/test_code_exec.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Update builtin/__init__.py**

```python
# src/tools/builtin/__init__.py
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools
from src.tools.builtin.data_ops import register_data_tools
from src.tools.builtin.code_exec import register_code_tools


def register_all_builtin_tools(registry: ToolRegistry, workspace_dir: str = "."):
    """Register all built-in tools."""
    register_file_tools(registry)
    register_data_tools(registry)
    register_code_tools(registry, workspace_dir=workspace_dir)
```

- [ ] **Step 6: Commit**

```bash
git add src/tools/sandbox.py src/tools/builtin/code_exec.py src/tools/builtin/__init__.py tests/test_tools/test_builtin/test_code_exec.py
git commit -m "feat: Python sandbox with restricted execution and built-in code tools"
```

---

## Task 6: Network Tools

**Files:**
- Create: `src/tools/builtin/network.py`
- Create: `tests/test_tools/test_builtin/test_network.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_builtin/test_network.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_fetch_url_success():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.network import register_network_tools

    registry = ToolRegistry()
    register_network_tools(registry)

    mock_response = MagicMock()
    mock_response.text = "<html>Hello</html>"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        instance.get = AsyncMock(return_value=mock_response)

        tool = registry.get("fetch_url")
        result = await tool.handler(url="https://example.com")

    assert result["status"] == 200
    assert "Hello" in result["body"]


@pytest.mark.asyncio
async def test_fetch_url_handles_error():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.network import register_network_tools
    import httpx

    registry = ToolRegistry()
    register_network_tools(registry)

    with patch("httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        instance.get = AsyncMock(side_effect=httpx.ConnectError("connection failed"))

        tool = registry.get("fetch_url")
        with pytest.raises(Exception):
            await tool.handler(url="https://invalid.example")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_builtin/test_network.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement network tools**

```python
# src/tools/builtin/network.py
import httpx
from typing import Any

from src.tools.registry import ToolRegistry


def register_network_tools(registry: ToolRegistry):
    """Register network tools."""

    @registry.register(
        name="fetch_url",
        description="Fetch content from a URL",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                "headers": {"type": "object", "description": "HTTP headers", "default": {}},
                "body": {"type": "string", "description": "Request body"},
            },
            "required": ["url"],
        },
        timeout=30,
    )
    async def fetch_url(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers or {},
                content=body,
            )
            return {
                "status": response.status_code,
                "body": response.text,
                "headers": dict(response.headers),
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_builtin/test_network.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Update builtin/__init__.py**

```python
# src/tools/builtin/__init__.py
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools
from src.tools.builtin.data_ops import register_data_tools
from src.tools.builtin.code_exec import register_code_tools
from src.tools.builtin.network import register_network_tools


def register_all_builtin_tools(registry: ToolRegistry, workspace_dir: str = "."):
    """Register all built-in tools."""
    register_file_tools(registry)
    register_data_tools(registry)
    register_code_tools(registry, workspace_dir=workspace_dir)
    register_network_tools(registry)
```

- [ ] **Step 6: Commit**

```bash
git add src/tools/builtin/network.py src/tools/builtin/__init__.py tests/test_tools/test_builtin/test_network.py
git commit -m "feat: built-in network tools (fetch_url)"
```

---

## Task 7: RAG Chunker

**Files:**
- Create: `src/rag/__init__.py`
- Create: `src/rag/chunker.py`
- Create: `tests/test_rag/__init__.py`
- Create: `tests/test_rag/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rag/test_chunker.py
import pytest


def test_chunk_fixed_size():
    from src.rag.chunker import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("a" * 25)
    assert len(chunks) > 1
    # Each chunk should be <= 10 chars
    for chunk in chunks:
        assert len(chunk.text) <= 10


def test_chunk_recursive_markdown():
    from src.rag.chunker import RecursiveChunker

    text = """# Section 1
Content of section 1.

## Subsection 1.1
Details here.

# Section 2
Content of section 2."""

    chunker = RecursiveChunker(chunk_size=200, overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_preserves_metadata():
    from src.rag.chunker import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=10, overlap=0)
    chunks = chunker.chunk("hello world test", metadata={"source": "test.txt"})
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"


def test_chunk_empty_text():
    from src.rag.chunker import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("")
    assert len(chunks) == 0


def test_chunk_text_shorter_than_size():
    from src.rag.chunker import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk("short text")
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag/test_chunker.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement chunker**

```python
# src/rag/__init__.py
```

```python
# src/rag/chunker.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    source: str = ""


class Chunker(ABC):
    """Base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into chunks."""
        ...


class FixedSizeChunker(Chunker):
    """Split text into fixed-size chunks with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if not text.strip():
            return []

        meta = metadata or {}
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=meta.copy(),
                    index=index,
                    source=meta.get("source", ""),
                ))
                index += 1

            start += self.chunk_size - self.overlap

        return chunks


class RecursiveChunker(Chunker):
    """Split text recursively by headers, then by token count."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if not text.strip():
            return []

        meta = metadata or {}

        # First split by markdown headers
        sections = self._split_by_headers(text)

        # Then apply fixed-size chunking to each section
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                if section.strip():
                    chunks.append(Chunk(
                        text=section,
                        metadata=meta.copy(),
                        index=len(chunks),
                        source=meta.get("source", ""),
                    ))
            else:
                # Use fixed-size for large sections
                sub_chunker = FixedSizeChunker(self.chunk_size, self.overlap)
                sub_chunks = sub_chunker.chunk(section, meta)
                for sc in sub_chunks:
                    sc.index = len(chunks)
                    chunks.append(sc)

        return chunks

    def _split_by_headers(self, text: str) -> list[str]:
        """Split text by markdown headers (# ## ### etc)."""
        lines = text.split("\n")
        sections = []
        current = []

        for line in lines:
            if line.startswith("#") and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append("\n".join(current))

        return sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag/test_chunker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/__init__.py src/rag/chunker.py tests/test_rag/__init__.py tests/test_rag/test_chunker.py
git commit -m "feat: RAG chunker with FixedSize and Recursive strategies"
```

---

## Task 8: Embedder

**Files:**
- Create: `src/rag/embedder.py`
- Create: `tests/test_rag/test_embedder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rag/test_embedder.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.mark.asyncio
async def test_embedder_embed_single():
    """Embedder returns a vector for a single text."""
    from src.rag.embedder import Embedder

    with patch("src.rag.embedder.SentenceTransformer") as MockModel:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        MockModel.return_value = mock_model

        embedder = Embedder(model_name="test-model", device="cpu")
        result = await embedder.embed("hello world")

    assert len(result) == 3
    assert result[0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_embedder_embed_batch():
    """Embedder handles batch embedding."""
    from src.rag.embedder import Embedder

    with patch("src.rag.embedder.SentenceTransformer") as MockModel:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        MockModel.return_value = mock_model

        embedder = Embedder(model_name="test-model", device="cpu")
        results = await embedder.embed_batch(["text1", "text2"])

    assert len(results) == 2
    assert len(results[0]) == 2
    assert results[1][0] == pytest.approx(0.3)


def test_embedder_dimension():
    from src.rag.embedder import Embedder

    with patch("src.rag.embedder.SentenceTransformer") as MockModel:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        MockModel.return_value = mock_model

        embedder = Embedder(model_name="test-model", device="cpu")
        assert embedder.dimension == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag/test_embedder.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement embedder**

```python
# src/rag/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """Wraps sentence-transformers for text embedding."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 32):
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self._model.encode([text], batch_size=1)
        return embedding[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag/test_embedder.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/embedder.py tests/test_rag/test_embedder.py
git commit -m "feat: RAG embedder wrapping sentence-transformers with bge-m3"
```

---

## Task 9: ChromaDB Vector Store

**Files:**
- Create: `src/rag/vector_store.py`
- Create: `src/rag/stores/__init__.py`
- Create: `src/rag/stores/chroma_store.py`
- Create: `tests/test_rag/test_chroma_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rag/test_chroma_store.py
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_chroma_store_upsert_and_search(tmp_path):
    from src.rag.stores.chroma_store import ChromaVectorStore
    from src.rag.chunker import Chunk

    store = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), collection="test")

    chunks = [
        Chunk(text="hello world", metadata={"source": "a.txt"}, index=0),
        Chunk(text="foo bar", metadata={"source": "b.txt"}, index=1),
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    await store.upsert(chunks, embeddings)

    # Search with a query embedding close to the first chunk
    results = await store.search(query_embedding=[0.1, 0.2, 0.3], top_k=1)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_chroma_store_delete(tmp_path):
    from src.rag.stores.chroma_store import ChromaVectorStore
    from src.rag.chunker import Chunk

    store = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), collection="test_del")

    chunks = [Chunk(text="to delete", metadata={}, index=0)]
    embeddings = [[0.1, 0.2]]

    await store.upsert(chunks, embeddings)

    # Get the IDs that were stored
    results = await store.search(query_embedding=[0.1, 0.2], top_k=10)
    ids = [r.id for r in results]

    await store.delete(ids)

    results_after = await store.search(query_embedding=[0.1, 0.2], top_k=10)
    assert len(results_after) == 0


@pytest.mark.asyncio
async def test_chroma_store_empty_search(tmp_path):
    from src.rag.stores.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), collection="empty")
    results = await store.search(query_embedding=[0.1, 0.2], top_k=5)
    assert len(results) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag/test_chroma_store.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement vector store ABC and ChromaDB implementation**

```python
# src/rag/vector_store.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.rag.chunker import Chunk


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base for vector stores."""

    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete chunks by ID."""
        ...
```

```python
# src/rag/stores/__init__.py
```

```python
# src/rag/stores/chroma_store.py
import uuid
from typing import Any

import chromadb

from src.rag.chunker import Chunk
from src.rag.vector_store import VectorStore, SearchResult


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store."""

    def __init__(self, persist_dir: str = "./data/chroma", collection: str = "default"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings."""
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {**c.metadata, "index": c.index, "source": c.source}
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()) if self._collection.count() > 0 else 0,
        }
        if filters:
            kwargs["where"] = filters

        if kwargs["n_results"] == 0:
            return []

        results = self._collection.query(**kwargs)

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(
                SearchResult(
                    id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    score=1 - results["distances"][0][i],  # cosine distance to similarity
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
            )

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete chunks by ID."""
        if ids:
            self._collection.delete(ids=ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag/test_chroma_store.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/vector_store.py src/rag/stores/__init__.py src/rag/stores/chroma_store.py tests/test_rag/test_chroma_store.py
git commit -m "feat: VectorStore ABC and ChromaDB implementation with upsert/search/delete"
```

---

## Task 10: RAG Retrieval Pipeline

**Files:**
- Create: `src/rag/retrieval.py`
- Create: `tests/test_rag/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rag/test_retrieval.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.rag.vector_store import SearchResult


@pytest.mark.asyncio
async def test_retrieval_basic_search():
    """Retrieval pipeline returns results from vector store."""
    from src.rag.retrieval import RetrievalPipeline

    mock_vector_store = MagicMock()
    mock_vector_store.search = AsyncMock(return_value=[
        SearchResult(id="1", text="result 1", score=0.9, metadata={}),
        SearchResult(id="2", text="result 2", score=0.8, metadata={}),
    ])

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    pipeline = RetrievalPipeline(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )

    results = await pipeline.retrieve("test query", top_k=2)
    assert len(results) == 2
    assert results[0].text == "result 1"


@pytest.mark.asyncio
async def test_retrieval_with_rerank():
    """Retrieval pipeline applies reranking when enabled."""
    from src.rag.retrieval import RetrievalPipeline

    mock_vector_store = MagicMock()
    mock_vector_store.search = AsyncMock(return_value=[
        SearchResult(id="1", text="less relevant", score=0.7, metadata={}),
        SearchResult(id="2", text="more relevant", score=0.8, metadata={}),
    ])

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[
        SearchResult(id="2", text="more relevant", score=0.95, metadata={}),
        SearchResult(id="1", text="less relevant", score=0.6, metadata={}),
    ])

    pipeline = RetrievalPipeline(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        reranker=mock_reranker,
    )

    results = await pipeline.retrieve("test query", top_k=2, rerank=True)
    assert results[0].text == "more relevant"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag/test_retrieval.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement retrieval pipeline**

```python
# src/rag/retrieval.py
from typing import Any

from src.rag.vector_store import VectorStore, SearchResult


class RetrievalPipeline:
    """Hybrid retrieval with optional reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Any,  # Embedder instance
        reranker: Any | None = None,
    ):
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for a query."""
        # Step 1: Embed the query
        query_embedding = await self._embedder.embed(query)

        # Step 2: Vector search (fetch more for reranking)
        fetch_k = top_k * 2 if rerank and self._reranker else top_k
        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filters=filters,
        )

        # Step 3: Rerank if enabled
        if rerank and self._reranker and results:
            results = await self._reranker.rerank(query, results)

        return results[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag/test_retrieval.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval.py tests/test_rag/test_retrieval.py
git commit -m "feat: RAG retrieval pipeline with vector search and optional reranking"
```

---

## Task 11: RAG Tools Integration

**Files:**
- Create: `src/tools/builtin/rag_ops.py`
- Create: `tests/test_tools/test_builtin/test_rag_ops.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_builtin/test_rag_ops.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.rag.vector_store import SearchResult
from src.rag.chunker import Chunk


@pytest.mark.asyncio
async def test_search_knowledge():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.rag_ops import register_rag_tools

    mock_retrieval = MagicMock()
    mock_retrieval.retrieve = AsyncMock(return_value=[
        SearchResult(id="1", text="relevant info", score=0.9, metadata={"source": "doc.txt"}),
    ])

    registry = ToolRegistry()
    register_rag_tools(registry, retrieval_pipeline=mock_retrieval)

    tool = registry.get("search_knowledge")
    result = await tool.handler(query="test question", top_k=1)
    assert len(result) == 1
    assert result[0]["text"] == "relevant info"


@pytest.mark.asyncio
async def test_index_document():
    from src.tools.registry import ToolRegistry
    from src.tools.builtin.rag_ops import register_rag_tools

    mock_processor = MagicMock()
    mock_processor.ingest = AsyncMock(return_value=[
        Chunk(text="chunk 1", metadata={}, index=0),
        Chunk(text="chunk 2", metadata={}, index=1),
    ])

    registry = ToolRegistry()
    register_rag_tools(registry, document_processor=mock_processor)

    tool = registry.get("index_document")
    result = await tool.handler(source="/path/to/doc.txt")
    assert result == 2  # number of chunks indexed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_builtin/test_rag_ops.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement RAG tools**

```python
# src/tools/builtin/rag_ops.py
from typing import Any

from src.tools.registry import ToolRegistry


def register_rag_tools(
    registry: ToolRegistry,
    retrieval_pipeline: Any = None,
    document_processor: Any = None,
):
    """Register RAG-powered tools."""

    @registry.register(
        name="search_knowledge",
        description="Search the knowledge base for relevant information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "collection": {"type": "string", "description": "Collection name", "default": "default"},
                "top_k": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["query"],
        },
    )
    async def search_knowledge(query: str, collection: str = "default", top_k: int = 5) -> list[dict]:
        if retrieval_pipeline is None:
            raise RuntimeError("RAG retrieval pipeline not configured")
        results = await retrieval_pipeline.retrieve(query, top_k=top_k)
        return [
            {"text": r.text, "score": r.score, "metadata": r.metadata}
            for r in results
        ]

    @registry.register(
        name="index_document",
        description="Index a document into the knowledge base",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to the document"},
                "collection": {"type": "string", "description": "Collection name", "default": "default"},
            },
            "required": ["source"],
        },
    )
    async def index_document(source: str, collection: str = "default") -> int:
        if document_processor is None:
            raise RuntimeError("Document processor not configured")
        chunks = await document_processor.ingest(source, metadata={"collection": collection})
        return len(chunks)
```

- [ ] **Step 4: Update builtin/__init__.py**

```python
# src/tools/builtin/__init__.py
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools
from src.tools.builtin.data_ops import register_data_tools
from src.tools.builtin.code_exec import register_code_tools
from src.tools.builtin.network import register_network_tools
from src.tools.builtin.rag_ops import register_rag_tools


def register_all_builtin_tools(
    registry: ToolRegistry,
    workspace_dir: str = ".",
    retrieval_pipeline=None,
    document_processor=None,
):
    """Register all built-in tools."""
    register_file_tools(registry)
    register_data_tools(registry)
    register_code_tools(registry, workspace_dir=workspace_dir)
    register_network_tools(registry)
    register_rag_tools(
        registry,
        retrieval_pipeline=retrieval_pipeline,
        document_processor=document_processor,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tools/test_builtin/test_rag_ops.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/tools/builtin/rag_ops.py src/tools/builtin/__init__.py tests/test_tools/test_builtin/test_rag_ops.py
git commit -m "feat: RAG tools (search_knowledge, index_document) bridging Tools and RAG modules"
```

---

## Verification Checklist

After completing all tasks, run:

```bash
# All tests pass
pytest tests/ -v

# No import errors
python -c "from src.tools.registry import ToolRegistry; from src.tools.engine import ToolEngine; from src.rag.chunker import RecursiveChunker; from src.rag.vector_store import VectorStore; print('All imports OK')"

# Tool registration works
python -c "
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_all_builtin_tools
r = ToolRegistry()
register_all_builtin_tools(r)
print(f'Registered {len(r.list_tools())} tools: {r.list_names()}')
"
```
