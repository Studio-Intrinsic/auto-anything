from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import WorkspaceLayout


@dataclass(frozen=True)
class WorkspacePaths:
    root_dir: Path
    candidate_dir: Path
    evaluator_dir: Path
    artifacts_dir: Path
    replay_dir: Path

    def ensure_exists(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)


def resolve_workspace_paths(base_dir: Path, layout: WorkspaceLayout) -> WorkspacePaths:
    root_dir = (base_dir / layout.root_dir).resolve()
    return WorkspacePaths(
        root_dir=root_dir,
        candidate_dir=(root_dir / layout.candidate_dir).resolve(),
        evaluator_dir=(root_dir / layout.evaluator_dir).resolve(),
        artifacts_dir=(root_dir / layout.artifacts_dir).resolve(),
        replay_dir=(root_dir / layout.replay_dir).resolve(),
    )
