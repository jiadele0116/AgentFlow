"""
技能 YAML 加载器模块 (Skill Loader)

从 YAML 文件或目录加载 Skill 定义，解析为 Skill 对象。
支持单文件加载和目录批量加载。

YAML 格式示例：
  name: code_review
  description: "审查代码质量和潜在问题"
  required_tools:
    - read_file
    - lint_code
  steps:
    - id: read_code
      type: tool_call
      tool: read_file
      params:
        path: "{{ input.file_path }}"
      output: code_content

    - id: analyze
      type: llm_reasoning
      prompt: |
        审查以下代码：{{ code_content }}
      output: review_result
"""
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import SkillError
from src.core.logging import get_logger
from src.skills.engine import Skill, SkillStep, StepType

logger = get_logger("skills.loader")


def load_skill_from_yaml(yaml_path: str | Path) -> Skill:
    """从 YAML 文件加载单个 Skill 定义。

    Args:
        yaml_path: YAML 文件路径

    Returns:
        Skill 对象

    Raises:
        SkillError: YAML 格式错误或字段缺失
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise SkillError(
            f"Skill 定义文件不存在: {yaml_path}",
            code="FILE_NOT_FOUND",
            context={"path": str(yaml_path)},
        )

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SkillError(
            f"YAML 解析失败: {e}",
            code="YAML_PARSE_ERROR",
            context={"path": str(yaml_path)},
        )

    if not isinstance(data, dict):
        raise SkillError(
            f"YAML 文件格式错误，期望字典类型",
            code="YAML_FORMAT_ERROR",
            context={"path": str(yaml_path)},
        )

    return _parse_skill(data, source=str(yaml_path))


def load_skills_from_directory(directory: str | Path) -> list[Skill]:
    """从目录批量加载所有 .yaml / .yml Skill 定义。

    Args:
        directory: Skill 定义目录路径

    Returns:
        Skill 对象列表
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning(f"Skill 定义目录不存在: {directory}")
        return []

    skills = []
    for yaml_file in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            skill = load_skill_from_yaml(yaml_file)
            skills.append(skill)
            logger.info(f"已加载技能: {skill.name} (来自 {yaml_file.name})")
        except SkillError as e:
            logger.warning(f"跳过无效 Skill 文件 {yaml_file.name}: {e}")

    return skills


def _parse_skill(data: dict[str, Any], source: str = "") -> Skill:
    """将 YAML 解析后的字典转换为 Skill 对象。"""
    name = data.get("name")
    if not name:
        raise SkillError(
            "Skill 定义缺少 'name' 字段",
            code="MISSING_FIELD",
            context={"source": source},
        )

    description = data.get("description", "")
    required_tools = data.get("required_tools", [])
    input_schema = data.get("input_schema", {})
    output_schema = data.get("output_schema", {})

    raw_steps = data.get("steps", [])
    steps = [_parse_step(s, source) for s in raw_steps]

    return Skill(
        name=name,
        description=description,
        required_tools=required_tools,
        steps=steps,
        input_schema=input_schema,
        output_schema=output_schema,
    )


def _parse_step(data: dict[str, Any], source: str = "") -> SkillStep:
    """将步骤字典转换为 SkillStep 对象。"""
    step_id = data.get("id")
    if not step_id:
        raise SkillError(
            "步骤定义缺少 'id' 字段",
            code="MISSING_FIELD",
            context={"source": source},
        )

    step_type_str = data.get("type")
    if not step_type_str:
        raise SkillError(
            f"步骤 [{step_id}] 缺少 'type' 字段",
            code="MISSING_FIELD",
            context={"source": source, "step_id": step_id},
        )

    try:
        step_type = StepType(step_type_str)
    except ValueError:
        raise SkillError(
            f"步骤 [{step_id}] 类型无效: {step_type_str}，"
            f"支持的类型: {[t.value for t in StepType]}",
            code="INVALID_STEP_TYPE",
            context={"source": source, "step_id": step_id, "type": step_type_str},
        )

    # 递归解析子步骤
    then_steps = [_parse_step(s, source) for s in data.get("then_steps", [])]
    else_steps = [_parse_step(s, source) for s in data.get("else_steps", [])]
    body_steps = [_parse_step(s, source) for s in data.get("body_steps", [])]

    return SkillStep(
        id=step_id,
        type=step_type,
        tool=data.get("tool", ""),
        params=data.get("params", {}),
        prompt=data.get("prompt", ""),
        condition=data.get("condition", ""),
        then_steps=then_steps,
        else_steps=else_steps,
        body_steps=body_steps,
        max_iterations=data.get("max_iterations", 10),
        output=data.get("output", ""),
    )
