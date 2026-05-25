"""Skill execution engine."""
import re
from typing import Any
from src.llm.provider import LLMProvider


class SkillContext:
    def __init__(self, input_data: dict[str, Any]):
        self._data: dict[str, Any] = {"input": input_data}
        self.iteration = 0

    def set(self, key: str, value: Any):
        self._data[key] = value

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def resolve_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {k: self._resolve_template(v) if isinstance(v, str) else v for k, v in params.items()}

    def _resolve_template(self, template: str) -> str:
        def replacer(match):
            path = match.group(1).strip()
            parts = path.split(".")
            value = self._data
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, match.group(0))
                else:
                    return match.group(0)
            return str(value)
        return re.sub(r"\{\{(.*?)\}\}", replacer, template)


class SkillEngine:
    def __init__(self, tool_engine: Any, llm: LLMProvider):
        self._tool_engine = tool_engine
        self._llm = llm

    async def execute_skill(self, skill_def: dict[str, Any], input_data: dict[str, Any]) -> dict[str, Any]:
        context = SkillContext(input_data)
        for step in skill_def.get("steps", []):
            await self._execute_step(step, context)
        return context._data

    async def _execute_step(self, step: dict[str, Any], context: SkillContext):
        step_type = step["type"]
        if step_type == "tool_call":
            params = context.resolve_params(step.get("params", {}))
            result = await self._tool_engine.execute(step["tool"], params)
            context.set(step["output"], result.data if result.success else result.error)
        elif step_type == "llm_reasoning":
            prompt = context._resolve_template(step["prompt"])
            response = await self._llm.complete(messages=[{"role": "user", "content": prompt}])
            context.set(step["output"], response.content)
        elif step_type == "conditional":
            condition = context._resolve_template(step.get("condition", "false"))
            if condition.lower() in ("true", "1", "yes"):
                for sub in step.get("then_steps", []):
                    await self._execute_step(sub, context)
            else:
                for sub in step.get("else_steps", []):
                    await self._execute_step(sub, context)
        elif step_type == "loop":
            for _ in range(step.get("max_iterations", 10)):
                context.iteration += 1
                for sub in step.get("body_steps", []):
                    await self._execute_step(sub, context)
