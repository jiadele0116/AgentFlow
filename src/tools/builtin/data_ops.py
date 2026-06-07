"""
数据处理工具模块 (Data Processing Tools)

提供常见的数据格式转换能力，支持 JSON 和 CSV 之间的相互转换。

注册的工具：
  - parse_json(text): 解析 JSON 字符串为 Python 对象
  - csv_to_dict(csv_text): 解析 CSV 文本为字典列表
  - json_to_csv(data): 将字典列表转换为 CSV 文本
"""
import csv
import io
import json
from typing import Any
from src.tools.registry import ToolRegistry


def register_data_tools(registry: ToolRegistry):
    """注册数据处理工具到注册表。"""

    @registry.register(
        name="parse_json", description="Parse a JSON string",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    async def parse_json(text: str) -> Any:
        """解析 JSON 字符串为 Python 对象。

        Args:
            text: JSON 格式的字符串

        Returns:
            解析后的 Python 对象（dict/list/str/int/float/bool/None）

        Raises:
            json.JSONDecodeError: JSON 格式错误
        """
        return json.loads(text)

    @registry.register(
        name="csv_to_dict", description="Parse CSV text into list of dicts",
        parameters={"type": "object", "properties": {"csv_text": {"type": "string"}}, "required": ["csv_text"]},
    )
    async def csv_to_dict(csv_text: str) -> list[dict[str, str]]:
        """将 CSV 文本解析为字典列表。

        CSV 的第一行作为键名，后续每行对应一个字典。

        Args:
            csv_text: CSV 格式的文本

        Returns:
            字典列表，每个字典对应 CSV 的一行
        """
        return list(csv.DictReader(io.StringIO(csv_text)))

    @registry.register(
        name="json_to_csv", description="Convert list of dicts to CSV text",
        parameters={"type": "object", "properties": {"data": {"type": "array", "items": {"type": "object"}}}, "required": ["data"]},
    )
    async def json_to_csv(data: list[dict[str, str]]) -> str:
        """将字典列表转换为 CSV 文本。

        使用第一个字典的键作为 CSV 表头。

        Args:
            data: 字典列表

        Returns:
            CSV 格式的文本字符串
        """
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
