from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import SkillPack


@dataclass
class SkillRegistry:
    _skills: dict[str, SkillPack] = field(default_factory=dict)

    def register(self, skill: SkillPack) -> None:
        self._skills[skill.skill_id] = skill

    def resolve(self, skill_ids: tuple[str, ...]) -> tuple[SkillPack, ...]:
        resolved: list[SkillPack] = []
        for skill_id in skill_ids:
            try:
                resolved.append(self._skills[skill_id])
            except KeyError as exc:
                raise KeyError(f"Unknown skill: {skill_id}") from exc
        return tuple(resolved)

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

