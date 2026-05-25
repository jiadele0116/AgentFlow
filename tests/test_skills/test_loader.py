"""Tests for skill loader."""
import pytest
import yaml
from pathlib import Path
from src.skills.loader import SkillLoader


def test_loader_loads_yaml(tmp_path):
    skill_def = {"name": "test_skill", "steps": [{"id": "s1", "type": "tool_call", "tool": "read_file", "params": {"path": "{{ input.path }}"}, "output": "content"}]}
    (tmp_path / "test_skill.yaml").write_text(yaml.dump(skill_def))
    loader = SkillLoader(str(tmp_path))
    loaded = loader.load("test_skill")
    assert loaded["name"] == "test_skill"


def test_loader_lists_skills(tmp_path):
    for name in ["skill_a", "skill_b"]:
        (tmp_path / f"{name}.yaml").write_text(yaml.dump({"name": name, "steps": []}))
    loader = SkillLoader(str(tmp_path))
    assert set(loader.list_skills()) == {"skill_a", "skill_b"}


def test_loader_missing_skill(tmp_path):
    loader = SkillLoader(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")
