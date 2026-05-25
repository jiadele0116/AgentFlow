"""Code execution tools."""
from src.tools.registry import ToolRegistry
from src.tools.sandbox import execute_python, ExecutionResult


def register_code_tools(registry: ToolRegistry, workspace_dir: str = "."):
    @registry.register(
        name="run_python", description="Execute Python code in a sandbox",
        parameters={"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}, "required": ["code"]},
        timeout=30,
    )
    async def run_python(code: str, timeout: int = 30) -> ExecutionResult:
        return execute_python(code, timeout=timeout)
