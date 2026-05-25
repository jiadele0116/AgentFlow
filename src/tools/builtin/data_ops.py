"""Data processing tools."""
import csv
import io
import json
from typing import Any
from src.tools.registry import ToolRegistry


def register_data_tools(registry: ToolRegistry):
    @registry.register(
        name="parse_json", description="Parse a JSON string",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    async def parse_json(text: str) -> Any:
        return json.loads(text)

    @registry.register(
        name="csv_to_dict", description="Parse CSV text into list of dicts",
        parameters={"type": "object", "properties": {"csv_text": {"type": "string"}}, "required": ["csv_text"]},
    )
    async def csv_to_dict(csv_text: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(csv_text)))

    @registry.register(
        name="json_to_csv", description="Convert list of dicts to CSV text",
        parameters={"type": "object", "properties": {"data": {"type": "array", "items": {"type": "object"}}}, "required": ["data"]},
    )
    async def json_to_csv(data: list[dict[str, str]]) -> str:
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
