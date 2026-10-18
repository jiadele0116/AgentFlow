"""
技能执行引擎模块 (Skill Engine)

Skill 是 Tool 的高阶编排：多个 Tool 调用 + 控制逻辑 + LLM 推理组合而成。
本模块提供 Skill 的数据定义和执行引擎。

核心组件：
  - SkillStep: 技能步骤  定义（tool_call / llm_reasoning / conditional / loop）
  - Skill: 技能完整定义（名称、描述、步骤列表）
  - SkillContext: 执行上下文（步骤间传递数据）
  - SkillResult: 执行结果
  - SkillEngine: 技能执行引擎（按步骤执行技能流程）

执行流程：
  YAML 定义 → SkillLoader 解析 → Skill 对象 → SkillEngine.execute() → SkillResult
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from jinja2 import Template, TemplateError

from src.core.exceptions import SkillError
from src.core.logging import get_logger

logger = get_logger("skills.engine")


# ============ 步骤类型枚举 ============

class StepType(str, Enum):
    """技能步骤类型。"""
    TOOL_CALL = "tool_call"
    LLM_REASONING = "llm_reasoning"
    CONDITIONAL = "conditional"
    LOOP = "loop"


# ============ 数据类 ============

@dataclass
class SkillStep:
    """技能步骤定义。

    Attributes:
        id: 步骤唯一标识（同一 Skill 内唯一）
        type: 步骤类型（tool_call / llm_reasoning / conditional / loop）
        tool: 工具名称（tool_call 类型时使用）
        params: 工具参数模板字典（支持 Jinja2 模板引用上下文变量）
        prompt: LLM 提示词模板（llm_reasoning 类型时使用）
        condition: 条件表达式字符串（conditional / loop 类型时使用）
        then_steps: 条件为真时执行的子步骤列表
        else_steps: 条件为假时执行的子步骤列表（conditional 类型）
        body_steps: 循环体步骤列表（loop 类型）
        max_iterations: 最大循环次数（loop 类型，默认 10）
        output: 输出变量名（步骤结果存入上下文的 key）
    """
    id: str
    type: StepType
    tool: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    condition: str = ""
    then_steps: list[SkillStep] = field(default_factory=list)
    else_steps: list[SkillStep] = field(default_factory=list)
    body_steps: list[SkillStep] = field(default_factory=list)
    max_iterations: int = 10
    output: str = ""


@dataclass
class Skill:
    """技能完整定义。

    Attributes:
        name: 技能唯一标识
        description: 技能功能描述
        required_tools: 依赖的工具名称列表
        steps: 执行步骤列表
        input_schema: 输入参数 JSON Schema（可选）
        output_schema: 输出参数 JSON Schema（可选）
    """
    name: str
    description: str
    required_tools: list[str] = field(default_factory=list)
    steps: list[SkillStep] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillContext:
    """技能执行上下文：在步骤之间传递数据。

    Attributes:
        input: 用户输入参数
        variables: 步骤输出变量字典（步骤的 output 字段写入此处）
        iteration: 当前循环迭代次数
        history: 执行历史记录
    """
    input: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """从上下文中获取变量。支持 input.xxx 和直接变量名。"""
        if key.startswith("input."):
            sub_key = key[6:]
            return self.input.get(sub_key, default)
        return self.variables.get(key, self.input.get(key, default))

    def set(self, key: str, value: Any):
        """设置变量到上下文。"""
        self.variables[key] = value

    def render_template(self, template_str: str) -> str:
        """渲染 Jinja2 模板字符串。

        模板中可用变量：
          - input: 用户输入参数字典
          - 步骤输出变量名（直接引用）
        """
        try:
            template = Template(template_str)
            return template.render(
                input=self.input,
                **self.variables,
            )
        except TemplateError as e:
            raise SkillError(
                f"模板渲染失败: {e}",
                skill_name="",
                code="TEMPLATE_ERROR",
                context={"template": template_str[:200]},
            )

    def record(self, step_id: str, step_type: str, output: Any, duration_ms: float):
        """记录步骤执行历史。"""
        self.history.append({
            "step_id": step_id,
            "type": step_type,
            "output": str(output)[:500],
            "duration_ms": round(duration_ms, 2),
        })


@dataclass
class SkillResult:
    """技能执行结果。

    Attributes:
        success: 是否执行成功
        skill_name: 技能名称
        output: 最终输出（最后一个步骤的结果或指定输出变量）
        context: 完整执行上下文
        duration_ms: 执行耗时（毫秒）
        error: 错误信息（失败时）
    """
    success: bool
    skill_name: str
    output: Any = None
    context: SkillContext | None = None
    duration_ms: float = 0.0
    error: str = ""


# ============ 工具和 LLM 回调类型 ============

# 工具调用回调: async (tool_name, params) -> result
ToolCallable = Callable[[str, dict[str, Any]], Awaitable[Any]]

# LLM 调用回调: async (prompt) -> response_text
LLMCallable = Callable[[str], Awaitable[str]]


# ============ 执行引擎 ============

class SkillEngine:
    """技能执行引擎。

    负责按步骤执行 Skill 定义的流程，支持四种步骤类型：
      - tool_call: 调用工具，参数支持 Jinja2 模板
      - llm_reasoning: 调用 LLM，提示词支持 Jinja2 模板
      - conditional: 条件分支，根据表达式选择 then/else 分支
      - loop: 循环执行，满足条件时重复执行 body 步骤

    Args:
        tool_caller: 工具调用回调函数
        llm_caller: LLM 调用回调函数
    """

    def __init__(self, tool_caller: ToolCallable, llm_caller: LLMCallable):
        self._tool_caller = tool_caller
        self._llm_caller = llm_caller
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """注册技能到引擎。"""
        self._skills[skill.name] = skill
        logger.info(f"技能已注册: {skill.name}")

    def get(self, name: str) -> Skill | None:
        """根据名称获取技能。"""
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, Any]]:
        """列出所有已注册技能。"""
        return [
            {"name": s.name, "description": s.description, "required_tools": s.required_tools}
            for s in self._skills.values()
        ]

    async def execute(self, skill_name: str, input_data: dict[str, Any] | None = None) -> SkillResult:
        """执行指定技能。

        Args:
            skill_name: 技能名称
            input_data: 输入参数字典

        Returns:
            SkillResult 执行结果
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return SkillResult(
                success=False,
                skill_name=skill_name,
                error=f"技能 '{skill_name}' 未注册",
            )

        context = SkillContext(input=input_data or {})
        start_time = time.time()

        try:
            logger.info(f"正在执行技能: {skill_name}")
            output = await self._execute_steps(skill.steps, context, skill.name)
            duration = (time.time() - start_time) * 1000

            logger.info(f"技能 {skill_name} 执行完成，耗时 {duration:.0f}ms")
            return SkillResult(
                success=True,
                skill_name=skill_name,
                output=output,
                context=context,
                duration_ms=duration,
            )

        except SkillError:
            raise
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"技能 {skill_name} 执行失败: {e}")
            return SkillResult(
                success=False,
                skill_name=skill_name,
                context=context,
                duration_ms=duration,
                error=str(e),
            )

    async def _execute_steps(self, steps: list[SkillStep], context: SkillContext, skill_name: str) -> Any:
        """按顺序执行一组步骤。返回最后一个步骤的输出。"""
        last_output = None
        for step in steps:
            last_output = await self._execute_step(step, context, skill_name)
        return last_output

    async def _execute_step(self, step: SkillStep, context: SkillContext, skill_name: str) -> Any:
        """执行单个步骤。根据步骤类型分发到对应的处理方法。"""
        step_start = time.time()
        logger.debug(f"步骤 [{step.id}] 类型={step.type.value}")

        try:
            match step.type:
                case StepType.TOOL_CALL:
                    output = await self._execute_tool_call(step, context, skill_name)
                case StepType.LLM_REASONING:
                    output = await self._execute_llm_reasoning(step, context, skill_name)
                case StepType.CONDITIONAL:
                    output = await self._execute_conditional(step, context, skill_name)
                case StepType.LOOP:
                    output = await self._execute_loop(step, context, skill_name)
                case _:
                    raise SkillError(
                        f"未知步骤类型: {step.type}",
                        skill_name=skill_name,
                        code="UNKNOWN_STEP_TYPE",
                    )

            # 将输出存入上下文
            if step.output:
                context.set(step.output, output)

            duration = (time.time() - step_start) * 1000
            context.record(step.id, step.type.value, output, duration)
            logger.debug(f"步骤 [{step.id}] 完成，耗时 {duration:.0f}ms")
            return output

        except SkillError:
            raise
        except Exception as e:
            duration = (time.time() - step_start) * 1000
            context.record(step.id, step.type.value, f"ERROR: {e}", duration)
            raise SkillError(
                f"步骤 [{step.id}] 执行失败: {e}",
                skill_name=skill_name,
                code="STEP_EXECUTION_ERROR",
                context={"step_id": step.id, "step_type": step.type.value},
            )

    async def _execute_tool_call(self, step: SkillStep, context: SkillContext, skill_name: str) -> Any:
        """执行工具调用步骤。"""
        # 渲染参数模板
        resolved_params = {}
        for key, value in step.params.items():
            if isinstance(value, str) and "{{" in value:
                resolved_params[key] = context.render_template(value)
            else:
                resolved_params[key] = value

        return await self._tool_caller(step.tool, resolved_params)

    async def _execute_llm_reasoning(self, step: SkillStep, context: SkillContext, skill_name: str) -> str:
        """执行 LLM 推理步骤。"""
        prompt = context.render_template(step.prompt)
        return await self._llm_caller(prompt)

    async def _execute_conditional(self, step: SkillStep, context: SkillContext, skill_name: str) -> Any:
        """执行条件分支步骤。"""
        condition_result = self._evaluate_condition(step.condition, context)

        if condition_result:
            if step.then_steps:
                return await self._execute_steps(step.then_steps, context, skill_name)
        else:
            if step.else_steps:
                return await self._execute_steps(step.else_steps, context, skill_name)
        return None

    async def _execute_loop(self, step: SkillStep, context: SkillContext, skill_name: str) -> Any:
        """执行循环步骤。"""
        last_output = None
        for i in range(step.max_iterations):
            # 检查循环条件
            if not self._evaluate_condition(step.condition, context):
                break

            context.iteration = i
            last_output = await self._execute_steps(step.body_steps, context, skill_name)

        context.iteration = 0
        return last_output

    @staticmethod
    def _evaluate_condition(condition: str, context: SkillContext) -> bool:
        """评估条件表达式。

        支持的表达式格式：
          - 变量存在性: "variable_name"
          - 变量比较: "variable_name == 'value'"
          - 变量非空: "variable_name != ''"
          - Jinja2 表达式: "{{ variable_name }}"

        安全限制：仅允许简单表达式，不支持 exec/eval 的危险操作。
        """
        if not condition:
            return True

        # 渲染模板变量
        rendered = context.render_template(condition)

        # 简单的条件评估
        rendered = rendered.strip()

        # 布尔字面量
        if rendered.lower() in ("true", "1", "yes"):
            return True
        if rendered.lower() in ("false", "0", "no", ""):
            return False

        # 比较运算
        for op in ("==", "!=", ">=", "<=", ">", "<"):
            if op in rendered:
                left, right = rendered.split(op, 1)
                left = left.strip().strip("'\"")
                right = right.strip().strip("'\"")
                try:
                    left_num = float(left)
                    right_num = float(right)
                    match op:
                        case "==": return left_num == right_num
                        case "!=": return left_num != right_num
                        case ">=": return left_num >= right_num
                        case "<=": return left_num <= right_num
                        case ">": return left_num > right_num
                        case "<": return left_num < right_num
                except ValueError:
                    match op:
                        case "==": return left == right
                        case "!=": return left != right
                        case _: return False

        # 变量存在性检查
        return bool(rendered)
