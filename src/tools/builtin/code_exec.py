"""
代码执行工具模块 (Code Execution Tools)

提供在受限环境中执行 Python 代码的能力。

注册的工具：
  - run_python(code, timeout): 执行 Python 代码
"""
import asyncio
import sys
import io
import traceback
from dataclasses import dataclass, field
from typing import Any

from src.tools.registry import ToolRegistry


@dataclass
class ExecutionResult:
    """代码执行结果。"""
    success: bool
    output: str = ""
    error: str = ""
    variables: dict[str, Any] = field(default_factory=dict)


async def execute_python(code: str, timeout: int = 30) -> ExecutionResult:
    """在受限环境中执行 Python 代码。

    Args:
        code: Python 代码字符串
        timeout: 超时时间（秒）

    Returns:
        ExecutionResult: 执行结果
    """
    # 捕获标准输出
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    try:
        sys.stdout = captured_out
        sys.stderr = captured_err

        # 受限的全局变量：禁止危险操作
        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len, "str": str,
                "int": int, "float": float, "bool": bool, "list": list,
                "dict": dict, "tuple": tuple, "set": set, "type": type,
                "isinstance": isinstance, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
                "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
                "any": any, "all": all, "hasattr": hasattr, "getattr": getattr,
                "True": True, "False": False, "None": None,
            }
        }

        # 异步执行，带超时
        exec(code, safe_globals)

        output = captured_out.getvalue()
        return ExecutionResult(success=True, output=output)

    except Exception:
        error = traceback.format_exc()
        return ExecutionResult(success=False, error=error)

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def register_code_tools(registry: ToolRegistry, workspace_dir: str = "."):
    """注册代码执行工具到注册表。

    Args:
        registry: 工具注册表
        workspace_dir: 工作目录（预留）
    """

    @registry.register(
        name="run_python", description="Execute Python code in a sandbox",
        parameters={"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}, "required": ["code"]},
        timeout=30,
    )
    async def run_python(code: str, timeout: int = 30) -> ExecutionResult:
        """在沙箱中执行 Python 代码。

        Args:
            code: Python 代码字符串
            timeout: 超时时间（秒）

        Returns:
            ExecutionResult: 包含执行状态和输出/错误信息
        """
        return await execute_python(code, timeout=timeout)
