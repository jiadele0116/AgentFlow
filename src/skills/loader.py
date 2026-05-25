"""YAML skill definition loader."""
import yaml
from pathlib import Path
from typing import Any


class SkillLoader:
    def __init__(self, definitions_dir: str):
        self._dir = Path(definitions_dir)

    def load(self, name: str) -> dict[str, Any]:
        file_path = self._dir / f"{name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Skill definition not found: {name}")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def list_skills(self) -> list[str]:
        return [f.stem for f in self._dir.glob("*.yaml") if f.is_file()]

    def load_all(self) -> dict[str, dict[str, Any]]:
        return {name: self.load(name) for name in self.list_skills()}
