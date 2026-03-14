from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .engine import ExperimentEngine
from .execution import run_task_command
from .history import (
    build_experiment_context,
    choose_primary_signal_from_charter,
    load_experiment_history,
    record_experiment_record,
)
from .models import (
    AgentRole,
    CounterbalanceMode,
    CounterbalanceReport,
    CritiqueFinding,
    CritiqueSeverity,
    ExperimentRecord,
    IterationStep,
    TaskCharter,
)
from .serialization import evaluation_report_from_summary, load_task_charter


def _load_best_entry(task_root: Path) -> dict | None:
    best_path = task_root / "artifacts" / "best_experiment.json"
    if best_path.is_file():
        return json.loads(best_path.read_text(encoding="utf-8"))
    history = load_experiment_history(task_root)
    if not history:
        return None
    accepted = [entry for entry in history if entry.get("accepted")]
    return accepted[-1] if accepted else history[-1]


def _worktree_paths(task_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(task_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ()
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        raw_path = line[3:].strip()
        if raw_path.startswith("artifacts/"):
            continue
        if raw_path not in paths:
            paths.append(raw_path)
    return tuple(paths)


def _path_matches_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def run_self_critic(
    *,
    task_root: Path,
    charter: TaskCharter,
    touched_paths: tuple[str, ...],
    focus_subsystems: tuple[str, ...],
    change_summary: str,
) -> CounterbalanceReport:
    findings: list[CritiqueFinding] = []
    protected_paths = charter.search_surface.protected_paths
    for path in touched_paths:
        if _path_matches_prefix(path, protected_paths):
            findings.append(
                CritiqueFinding(
                    finding_id=f"protected-path-{path}",
                    summary=f"Modified protected path: {path}",
                    severity=CritiqueSeverity.CRITICAL,
                    rationale="Protected paths should not be edited during pipeline iteration.",
                )
            )

    for path in touched_paths:
        absolute = task_root / path
        if absolute.suffix == ".py" and absolute.is_file():
            line_count = len(absolute.read_text(encoding="utf-8").splitlines())
            if line_count > 220:
                findings.append(
                    CritiqueFinding(
                        finding_id=f"monolith-risk-{absolute.name}",
                        summary=f"Python file grew to {line_count} lines: {path}",
                        severity=CritiqueSeverity.HIGH,
                        rationale="Large pipeline files increase coupling and hurt subsystem iteration.",
                    )
                )

    owned_paths = tuple(
        path
        for subsystem in charter.search_surface.subsystems
        if subsystem.subsystem_id in focus_subsystems
        for path in subsystem.owned_paths
    )
    if focus_subsystems and owned_paths and not any(
        _path_matches_prefix(path, owned_paths)
        for path in touched_paths
    ):
        findings.append(
            CritiqueFinding(
                finding_id="focus-drift",
                summary="Focused iteration did not touch any files owned by the target subsystem.",
                severity=CritiqueSeverity.MEDIUM,
                rationale="The iteration may not be aligned with its declared subsystem focus.",
            )
        )

    if "monolith" in change_summary.lower() or "all-in-one" in change_summary.lower():
        findings.append(
            CritiqueFinding(
                finding_id="architecture-regression-intent",
                summary="Change summary suggests collapsing logic into a larger single component.",
                severity=CritiqueSeverity.HIGH,
                rationale="The system should resist monolithic pipeline growth.",
            )
        )

    return CounterbalanceReport(
        mode=CounterbalanceMode.SELF_CRITIC,
        findings=tuple(findings),
        notes=("Automatic self-critic pass over path hygiene and modularity risks.",),
    )


def _knowledge_items(
    experiment: ExperimentRecord,
    *,
    metric_name: str,
    metric_value: float,
    focus_subsystems: tuple[str, ...],
    counterbalance_report: CounterbalanceReport,
) -> tuple[str, ...]:
    items: list[str] = []
    if experiment.decision.accepted:
        if focus_subsystems:
            items.append(
                f"Focused work on {', '.join(focus_subsystems)} produced an accepted candidate with {metric_name}={metric_value}."
            )
        else:
            items.append(f"This candidate was accepted with {metric_name}={metric_value}.")
    else:
        items.append(
            f"This candidate was rejected. Reasons: {', '.join(experiment.decision.reasons) or 'none recorded'}."
        )
    for finding in counterbalance_report.findings:
        items.append(f"Critic finding: {finding.summary}")
    return tuple(items)


def run_task_iteration(
    *,
    task_root: Path,
    hypothesis: str,
    change_summary: str,
    label: str = "",
    focus_subsystems: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
    command_name: str = "evaluate",
) -> dict:
    task_root = task_root.expanduser().resolve()
    charter = load_task_charter(task_root / "task_charter.json")
    resolved_focus = focus_subsystems or charter.focus_subsystems
    touched_paths = _worktree_paths(task_root)

    execution = run_task_command(
        task_root=task_root,
        charter=charter,
        command_name=command_name,
        extra_env={"AUTO_ANYTHING_SKIP_AUTO_RECORD": "1"},
    )
    execution.check_returncode()
    candidate_summary = json.loads((task_root / "artifacts" / "eval_summary.json").read_text(encoding="utf-8"))
    candidate_report = evaluation_report_from_summary(candidate_summary)

    baseline_entry = _load_best_entry(task_root)
    if baseline_entry is None:
        baseline_summary = candidate_summary
    else:
        baseline_summary = {
            "candidate_id": baseline_entry.get("candidate_id", "baseline"),
            "signals": baseline_entry.get("signals", {}),
        }
    baseline_report = evaluation_report_from_summary(baseline_summary)

    counterbalance_report = run_self_critic(
        task_root=task_root,
        charter=charter,
        touched_paths=touched_paths,
        focus_subsystems=resolved_focus,
        change_summary=change_summary,
    )
    engine = ExperimentEngine()
    recorded = engine.record(
        charter=charter,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        counterbalance_report=counterbalance_report,
        focus_subsystems=resolved_focus,
        notes=notes,
    )
    experiment = ExperimentRecord(
        candidate_id=recorded.candidate_id,
        baseline_candidate_id=recorded.baseline_candidate_id,
        decision=recorded.decision,
        baseline_report=recorded.baseline_report,
        candidate_report=recorded.candidate_report,
        counterbalance_report=counterbalance_report,
        candidate_snapshot=recorded.candidate_snapshot,
        iteration_steps=(
            IterationStep(
                summary=change_summary,
                role=AgentRole.BUILDER,
                touched_paths=touched_paths,
                executed_commands=(
                    " ".join(execution.command),
                    f"backend={execution.backend_kind.value}",
                ),
                notes=(
                    hypothesis,
                    f"duration_seconds={execution.duration_seconds:.3f}",
                    *(f"sync_back={path}" for path in execution.synced_paths),
                ),
            ),
            IterationStep(
                summary="Automatic self-critic pass",
                role=AgentRole.CRITIC,
                touched_paths=touched_paths,
                notes=tuple(finding.summary for finding in counterbalance_report.findings) or counterbalance_report.notes,
            ),
            IterationStep(
                summary="Engine acceptance decision",
                role=AgentRole.JUDGE,
                notes=recorded.decision.reasons,
            ),
        ),
        focus_subsystems=recorded.focus_subsystems,
        notes=notes,
    )
    metric_name, _ = choose_primary_signal_from_charter(charter)
    history_entry = record_experiment_record(
        task_root=task_root,
        charter=charter,
        experiment=experiment,
        label=label or candidate_report.candidate_id,
        hypothesis=hypothesis,
        change_summary=change_summary,
        knowledge_items=_knowledge_items(
            experiment,
            metric_name=metric_name,
            metric_value=float(candidate_summary["signals"][metric_name]),
            focus_subsystems=resolved_focus,
            counterbalance_report=counterbalance_report,
        ),
        notes=notes,
    )
    context = build_experiment_context(task_root)
    return {
        "decision": {
            "accepted": experiment.decision.accepted,
            "reasons": list(experiment.decision.reasons),
            "blocking_signals": list(experiment.decision.blocking_signals),
            "blocking_findings": list(experiment.decision.blocking_findings),
            "utility_gain": experiment.decision.utility_gain,
        },
        "history_entry": history_entry,
        "context": context,
    }
