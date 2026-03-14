from __future__ import annotations

from pathlib import Path
import json


def load_task_inputs(task_root: Path) -> dict:
    """Return the raw referenced assets so the agent can shape task-specific ingestion."""
    charter = json.loads((task_root / "task_charter.json").read_text(encoding="utf-8"))
    return {
        "objective": charter["objective_statement"],
        "data_assets": charter.get("data_assets", []),
    }
