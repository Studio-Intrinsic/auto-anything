from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    CandidateSnapshot,
    EvaluationReport,
    ObjectiveBrief,
    RolePass,
    SkillContribution,
    TaskCharter,
)


class SkillPack(Protocol):
    skill_id: str
    description: str

    def contribute(self, brief: ObjectiveBrief) -> SkillContribution:
        ...


class ObjectiveCompiler(Protocol):
    def compile(self, brief: ObjectiveBrief, skills: tuple[SkillPack, ...] = ()) -> TaskCharter:
        ...


class TaskAdapter(Protocol):
    adapter_id: str

    def setup(self, charter: TaskCharter, work_dir: Path) -> None:
        ...

    def baseline_candidate(self, charter: TaskCharter, work_dir: Path) -> CandidateSnapshot:
        ...

    def materialize_candidate(self, charter: TaskCharter, work_dir: Path) -> CandidateSnapshot:
        ...

    def run_candidate(self, snapshot: CandidateSnapshot, charter: TaskCharter) -> EvaluationReport:
        ...


class ExperimentAgent(Protocol):
    agent_id: str

    def propose_next_step(self, charter: TaskCharter, snapshot: CandidateSnapshot, role_pass: RolePass) -> str:
        ...
