"""
技能编排模块 (Skills Module)

Skill 是 Tool 的高阶编排，由多个 Tool 调用 + 控制逻辑 + LLM 推理组合而成。

核心组件：
  - engine: 技能执行引擎（SkillEngine, Skill, SkillStep, SkillContext, SkillResult）
  - loader: YAML 技能定义加载器
  - definitions/: 内置技能 YAML 定义

使用示例：
    from src.skills.engine import SkillEngine
    from src.skills.loader import load_skills_from_directory

    engine = SkillEngine(tool_caller=my_tool_fn, llm_caller=my_llm_fn)
    for skill in load_skills_from_directory("src/skills/definitions"):
        engine.register(skill)

    result = await engine.execute("code_review", {"file_path": "main.py", "language": "python"})
"""
from src.skills.engine import Skill, SkillStep, SkillContext, SkillResult, SkillEngine, StepType
from src.skills.loader import load_skill_from_yaml, load_skills_from_directory

__all__ = [
    "Skill", "SkillStep", "SkillContext", "SkillResult", "SkillEngine", "StepType",
    "load_skill_from_yaml", "load_skills_from_directory",
]
