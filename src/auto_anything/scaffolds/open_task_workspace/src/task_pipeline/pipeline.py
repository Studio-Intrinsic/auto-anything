from __future__ import annotations

from pathlib import Path

from .data_ingestion import load_task_inputs


def run_candidate(task_root: Path) -> dict:
    """Placeholder candidate surface for a synthesized task workspace."""
    return {
        "status": "unimplemented",
        "next_step": "Replace this placeholder with the real task pipeline.",
        "inputs": load_task_inputs(task_root),
    }
