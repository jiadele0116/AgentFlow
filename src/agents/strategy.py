"""Agent execution strategies: ReAct, CoT, PlanExecute."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from src.llm.provider import LLMProvider


@dataclass
class StrategyResult:
    content: str
    tool_calls_made: int = 0
    iterations: int = 0


class ExecutionStrategy(ABC):
    @abstractmethod
    async def execute(self, task_description: str, context: list[dict[str, Any]]) -> StrategyResult: ...


class ReActStrategy(ExecutionStrategy):
    def __init__(self, llm: LLMProvider, tool_engine: Any, max_iterations: int = 10):
        self._llm = llm
        self._tool_engine = tool_engine
        self._max_iterations = max_iterations

    async def execute(self, task_description: str, context: list[dict[str, Any]]) -> StrategyResult:
        tools = self._tool_engine.list_tools()
        messages = list(context)
        messages.append({"role": "user", "content": task_description})
        tool_calls_made = 0

        for iteration in range(self._max_iterations):
            response = await self._llm.complete(messages=messages, tools=tools if tools else None)
            if not response.tool_calls:
                return StrategyResult(content=response.content, tool_calls_made=tool_calls_made, iterations=iteration + 1)
            messages.append({"role": "assistant", "content": response.content or ""})
            for tc in response.tool_calls:
                result = await self._tool_engine.execute(tc.name, tc.arguments)
                tool_calls_made += 1
                messages.append({"role": "user", "content": f"Tool result for {tc.name}: {result.data if result.success else result.error}"})
        return StrategyResult(content=response.content if response else "Max iterations reached",
                              tool_calls_made=tool_calls_made, iterations=self._max_iterations)


class CoTStrategy(ExecutionStrategy):
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def execute(self, task_description: str, context: list[dict[str, Any]]) -> StrategyResult:
        messages = list(context)
        messages.append({"role": "user", "content": f"Think step by step about this task:\n\n{task_description}"})
        response = await self._llm.complete(messages=messages)
        return StrategyResult(content=response.content, tool_calls_made=0, iterations=1)


class PlanExecuteStrategy(ExecutionStrategy):
    def __init__(self, llm: LLMProvider, tool_engine: Any, max_iterations: int = 10):
        self._llm = llm
        self._tool_engine = tool_engine
        self._max_iterations = max_iterations

    async def execute(self, task_description: str, context: list[dict[str, Any]]) -> StrategyResult:
        tools = self._tool_engine.list_tools()
        plan_messages = list(context)
        plan_messages.append({"role": "user", "content": f"Create a step-by-step plan. Output each step on a numbered line.\n\nTask: {task_description}"})
        plan_response = await self._llm.complete(messages=plan_messages)
        plan_steps = [l.strip() for l in plan_response.content.split("\n") if l.strip() and l.strip()[0].isdigit()]

        results, tool_calls_made = [], 0
        for i, step in enumerate(plan_steps[:self._max_iterations]):
            step_messages = list(context)
            step_messages.append({"role": "user", "content": f"Execute this step: {step}\n\nPrevious results: {results}"})
            step_response = await self._llm.complete(messages=step_messages, tools=tools if tools else None)
            if step_response.tool_calls:
                for tc in step_response.tool_calls:
                    r = await self._tool_engine.execute(tc.name, tc.arguments)
                    tool_calls_made += 1
                    results.append(f"Step {i+1}: {r.data if r.success else r.error}")
            else:
                results.append(f"Step {i+1}: {step_response.content}")
        return StrategyResult(content="\n".join(results), tool_calls_made=tool_calls_made, iterations=len(plan_steps))
