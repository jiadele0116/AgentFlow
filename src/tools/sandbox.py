"""Python code execution sandbox."""
import ast
import sys
import io
import traceback
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    return_value: object = None


BLOCKED_IMPORTS = {"subprocess", "shutil", "ctypes", "importlib"}


def _check_code_safety(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BLOCKED_IMPORTS:
                    return f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_IMPORTS:
                return f"Blocked import from: {node.module}"
    return None


def execute_python(code: str, timeout: int = 30) -> ExecutionResult:
    safety_error = _check_code_safety(code)
    if safety_error:
        return ExecutionResult(success=False, error=safety_error)

    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_out, captured_err = io.StringIO(), io.StringIO()
    try:
        sys.stdout, sys.stderr = captured_out, captured_err
        safe_builtins = {
            "print": print, "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "tuple": tuple, "set": set,
            "bool": bool, "type": type, "isinstance": isinstance, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "any": any, "all": all, "True": True, "False": False, "None": None,
            "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "Exception": Exception,
            "json": __import__("json"), "math": __import__("math"), "re": __import__("re"),
        }
        exec(code, {"__builtins__": safe_builtins})
        return ExecutionResult(success=True, output=captured_out.getvalue())
    except Exception:
        return ExecutionResult(success=False, error=traceback.format_exc())
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
