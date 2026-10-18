"""
Prompt 模板管理模块 (Prompt Templates)

集中管理所有 Agent 的 system prompt，统一约束风格。
核心原则：精简、约束输出、减少 token 消耗。

优化策略：
  1. 每个 prompt 加入输出约束（字数、格式）
  2. 区分简单/复杂场景，避免一刀切的冗长 prompt
  3. 去掉模板里的重复说明（LLM 已经知道的不说第二遍）
"""


# ── Planner ──

PLANNER_PROMPT = (
    "任务规划专家。分析任务并制定执行计划。\n"
    "要求：50字内概述，步骤不超过5步，用中文。"
)

PLANNER_PROMPT_WITH_CONTEXT = (
    "任务规划专家。分析任务并制定执行计划。\n"
    "要求：50字内概述，步骤不超过5步，用中文。\n"
    "参考上下文：{context}"
)


# ── Executor ──

EXECUTOR_PROMPT = (
    "任务执行专家。根据计划执行任务，必要时调用工具。\n"
    "要求：直接给出结果，不要重复问题，用中文。"
)

EXECUTOR_PROMPT_WITH_CONTEXT = (
    "任务执行专家。根据计划执行任务，必要时调用工具。\n"
    "要求：直接给出结果，不要重复问题，用中文。\n"
    "参考上下文：{context}"
)

EXECUTOR_TOOL_RESULT_PROMPT = (
    "根据工具结果给出最终答案。简洁直接，用中文。"
)

EXECUTOR_REVISION_PROMPT = (
    "上一版未通过审查，请改进。简洁直接，用中文。"
)


# ── Critic ──

CRITIC_PROMPT = (
    "质量审查员。审查结果是否准确完整。\n"
    "回复格式：APPROVED（通过）或 REJECTED: 原因（一句话），用中文。"
)


# ── ReAct Agent ──

REACT_PROMPT = (
    "智能助手。根据用户问题使用工具获取信息，然后给出准确回答。\n"
    "要求：简洁直接，不过度解释，用中文。"
)


# ── Memory 摘要 ──

MEMORY_SUMMARIZE_PROMPT = (
    "用中文总结对话要点。格式：要点列表，不超过150字。"
)


# ── Skill LLM 推理 ──

SKILL_REASONING_SUFFIX = "\n\n要求：简洁直接，不要废话。"


def get_planner_prompt(context: str = "") -> str:
    """获取 Planner prompt，有上下文时附加。"""
    if context:
        return PLANNER_PROMPT_WITH_CONTEXT.format(context=context)
    return PLANNER_PROMPT


def get_executor_prompt(context: str = "", revision: bool = False) -> str:
    """获取 Executor prompt。"""
    base = EXECUTOR_PROMPT_WITH_CONTEXT.format(context=context) if context else EXECUTOR_PROMPT
    if revision:
        base += "\n" + EXECUTOR_REVISION_PROMPT
    return base
