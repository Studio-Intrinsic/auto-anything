from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .models import ExperimentRecord, ObjectiveSignal, SignalDirection, TaskCharter


DEFAULT_TASK_GITIGNORE = (
    "artifacts/",
    "__pycache__/",
    "*.pyc",
)


def _run_git(task_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(task_root),
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_task_git_repo(task_root: Path, ignore_entries: tuple[str, ...] = DEFAULT_TASK_GITIGNORE) -> None:
    task_root = task_root.expanduser().resolve()
    if not (task_root / ".git").is_dir():
        _run_git(task_root, "init", "-b", "main")

    ignore_path = task_root / ".gitignore"
    existing = set()
    if ignore_path.is_file():
        existing = {line.strip() for line in ignore_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    lines = list(ignore_path.read_text(encoding="utf-8").splitlines()) if ignore_path.is_file() else []
    changed = False
    for entry in ignore_entries:
        if entry not in existing:
            lines.append(entry)
            changed = True
    if changed or not ignore_path.is_file():
        ignore_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    for key, value in (("user.name", "auto-anything"), ("user.email", "auto-anything@local")):
        completed = subprocess.run(
            ["git", "config", "--get", key],
            cwd=str(task_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            _run_git(task_root, "config", key, value)


def _has_head(task_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=str(task_root),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def current_commit_sha(task_root: Path) -> str:
    return _run_git(task_root, "rev-parse", "HEAD").stdout.strip()


def snapshot_experiment_workspace(task_root: Path, *, experiment_id: str, label: str) -> str:
    ensure_task_git_repo(task_root)
    _run_git(task_root, "add", "-A")
    _run_git(
        task_root,
        "commit",
        "--allow-empty",
        "-m",
        f"[auto-anything] {experiment_id} {label}".strip(),
    )
    commit_sha = current_commit_sha(task_root)
    _run_git(task_root, "tag", "-f", experiment_id, commit_sha)
    return commit_sha


def choose_primary_signal_from_charter(charter: TaskCharter | dict[str, Any]) -> tuple[str, SignalDirection]:
    raw_signals = charter.evaluation_plan.signals if isinstance(charter, TaskCharter) else charter["evaluation_plan"]["signals"]
    best_signal: ObjectiveSignal | dict[str, Any] | None = None
    best_rank: tuple[int, float] | None = None
    for signal in raw_signals:
        hard_gate = signal.hard_gate if isinstance(signal, ObjectiveSignal) else bool(signal.get("hard_gate"))
        weight = signal.weight if isinstance(signal, ObjectiveSignal) else float(signal.get("weight", 1.0))
        rank = (0 if hard_gate else 1, weight)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_signal = signal
    assert best_signal is not None
    if isinstance(best_signal, ObjectiveSignal):
        return best_signal.name, best_signal.direction
    return best_signal["name"], SignalDirection(best_signal["direction"])


def _metric_improved(value: float, best_value: float, direction: SignalDirection) -> bool:
    if direction in {SignalDirection.MAXIMIZE, SignalDirection.SATISFY}:
        return value > best_value
    return value < best_value


def _history_paths(task_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    artifacts_dir = task_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = artifacts_dir / "experiments"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return (
        artifacts_dir / "experiment_history.jsonl",
        artifacts_dir / "experiment_history.json",
        artifacts_dir / "progress_curve.svg",
        artifacts_dir / "best_experiment.json",
        reports_dir,
    )


def load_experiment_history(task_root: Path) -> list[dict[str, Any]]:
    history_path, _, _, _, _ = _history_paths(task_root)
    if not history_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def _write_history_views(task_root: Path, entries: list[dict[str, Any]]) -> None:
    history_path, latest_path, _, best_path, _ = _history_paths(task_root)
    history_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    latest_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    kept_entries = [entry for entry in entries if entry.get("accepted")]
    best_entry = kept_entries[-1] if kept_entries else (entries[-1] if entries else {})
    best_path.write_text(json.dumps(best_entry, indent=2, sort_keys=True), encoding="utf-8")


def _experiment_report_paths(task_root: Path, experiment_id: str) -> tuple[Path, Path]:
    _, _, _, _, reports_dir = _history_paths(task_root)
    return (
        reports_dir / f"{experiment_id}.json",
        reports_dir / f"{experiment_id}.md",
    )


def _knowledge_summary_path(task_root: Path) -> Path:
    artifacts_dir = task_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir / "knowledge_base.md"


def _previous_commit(task_root: Path) -> str:
    if not _has_head(task_root):
        return ""
    return current_commit_sha(task_root)


def _safe_git_output(task_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(task_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _collect_diff_summary(task_root: Path, previous_commit: str, commit_sha: str) -> dict[str, Any]:
    if not previous_commit:
        return {
            "previous_commit": "",
            "diff_stat": "Initial workspace snapshot.",
            "changed_files": [],
        }
    diff_stat = _safe_git_output(task_root, "diff", "--stat", previous_commit, commit_sha)
    changed_files_text = _safe_git_output(task_root, "diff", "--name-only", previous_commit, commit_sha)
    changed_files = tuple(line.strip() for line in changed_files_text.splitlines() if line.strip())
    return {
        "previous_commit": previous_commit,
        "diff_stat": diff_stat,
        "changed_files": changed_files,
    }


def _format_experiment_markdown(entry: dict[str, Any]) -> str:
    signals = entry.get("signals", {})
    signal_lines = "\n".join(
        f"- `{name}`: {value}"
        for name, value in sorted(signals.items())
    ) or "- none"
    decision_reasons = entry.get("decision_reasons", [])
    reason_lines = "\n".join(f"- {reason}" for reason in decision_reasons) or "- none"
    findings = entry.get("blocking_findings", [])
    finding_lines = "\n".join(f"- {item}" for item in findings) or "- none"
    notes = entry.get("notes", [])
    note_lines = "\n".join(f"- {item}" for item in notes) or "- none"
    learning_lines = "\n".join(f"- {item}" for item in entry.get("knowledge_items", [])) or "- none"
    touched_paths = "\n".join(f"- `{item}`" for item in entry.get("touched_paths", [])) or "- none"
    return (
        f"# {entry['experiment_id']}\n\n"
        f"- Label: `{entry.get('label', '')}`\n"
        f"- Accepted: `{entry.get('accepted')}`\n"
        f"- Candidate: `{entry.get('candidate_id', '')}`\n"
        f"- Focus subsystems: {', '.join(entry.get('focus_subsystems', [])) or 'none'}\n"
        f"- Git commit: `{entry.get('git_commit', '')}`\n"
        f"- Previous commit: `{entry.get('previous_commit', '')}`\n"
        f"- Metric: `{entry.get('metric_name', '')}` = `{entry.get('metric_value', '')}`\n\n"
        f"## Hypothesis\n\n{entry.get('hypothesis', '').strip() or 'Not recorded.'}\n\n"
        f"## Change Summary\n\n{entry.get('change_summary', '').strip() or 'Not recorded.'}\n\n"
        f"## Signals\n\n{signal_lines}\n\n"
        f"## Decision Reasons\n\n{reason_lines}\n\n"
        f"## Blocking Findings\n\n{finding_lines}\n\n"
        f"## Changed Paths\n\n{touched_paths}\n\n"
        f"## Diff Stat\n\n```\n{entry.get('diff_stat', '').strip() or 'No diff stat available.'}\n```\n\n"
        f"## Learnings\n\n{learning_lines}\n\n"
        f"## Notes\n\n{note_lines}\n"
    )


def _write_experiment_report(task_root: Path, entry: dict[str, Any]) -> None:
    json_path, markdown_path = _experiment_report_paths(task_root, entry["experiment_id"])
    json_path.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_format_experiment_markdown(entry), encoding="utf-8")


def _write_knowledge_base(task_root: Path, entries: list[dict[str, Any]]) -> None:
    path = _knowledge_summary_path(task_root)
    if not entries:
        path.write_text("# Experiment Knowledge Base\n\nNo experiments recorded yet.\n", encoding="utf-8")
        return
    kept_entries = [entry for entry in entries if entry.get("accepted")]
    best_entry = kept_entries[-1] if kept_entries else entries[-1]
    recent_entries = entries[-5:]
    subsystem_notes: dict[str, list[str]] = {}
    for entry in entries:
        for subsystem in entry.get("focus_subsystems", []):
            bucket = subsystem_notes.setdefault(subsystem, [])
            for item in entry.get("knowledge_items", []):
                if item not in bucket:
                    bucket.append(item)
    recent_lines = "\n".join(
        f"- `{entry['experiment_id']}` `{entry.get('label', '')}` accepted=`{entry.get('accepted')}` metric=`{entry.get('metric_value')}` reasons={', '.join(entry.get('decision_reasons', [])) or 'none'}"
        for entry in recent_entries
    )
    subsystem_section = "\n".join(
        f"### `{subsystem}`\n" + "\n".join(f"- {item}" for item in items[:6])
        for subsystem, items in sorted(subsystem_notes.items())
    ) or "No subsystem-specific learnings recorded yet."
    best_learnings = "\n".join(f"- {item}" for item in best_entry.get("knowledge_items", [])) or "- none recorded"
    text = (
        "# Experiment Knowledge Base\n\n"
        "## Best Known Approach\n\n"
        f"- Experiment: `{best_entry['experiment_id']}`\n"
        f"- Label: `{best_entry.get('label', '')}`\n"
        f"- Commit: `{best_entry.get('git_commit', '')}`\n"
        f"- Metric: `{best_entry.get('metric_name', '')}` = `{best_entry.get('metric_value', '')}`\n"
        f"- Focus subsystems: {', '.join(best_entry.get('focus_subsystems', [])) or 'none'}\n"
        f"- Change summary: {best_entry.get('change_summary', '').strip() or 'Not recorded.'}\n\n"
        "### Best Learnings\n"
        f"{best_learnings}\n\n"
        "## Recent Experiment Trail\n\n"
        f"{recent_lines}\n\n"
        "## Subsystem Learnings\n\n"
        f"{subsystem_section}\n"
    )
    path.write_text(text, encoding="utf-8")


def build_experiment_context(task_root: Path, limit: int = 8) -> str:
    entries = load_experiment_history(task_root)
    if not entries:
        return "No experiments recorded yet."
    relevant = entries[-limit:]
    lines = []
    for entry in relevant:
        lines.append(
            f"{entry['experiment_id']} accepted={entry.get('accepted')} metric={entry.get('metric_value')} label={entry.get('label', '')}"
        )
        if entry.get("change_summary"):
            lines.append(f"change: {entry['change_summary']}")
        if entry.get("knowledge_items"):
            lines.append(f"learned: {'; '.join(entry['knowledge_items'][:3])}")
        if entry.get("decision_reasons"):
            lines.append(f"reasons: {', '.join(entry['decision_reasons'])}")
    best_entry = next((entry for entry in reversed(entries) if entry.get("accepted")), entries[-1])
    return (
        f"Best experiment: {best_entry['experiment_id']} metric={best_entry.get('metric_value')} commit={best_entry.get('git_commit')}\n"
        + "\n".join(lines)
    )


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = value.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def _svg_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_progress_curve_svg(
    entries: list[dict[str, Any]],
    *,
    metric_name: str,
    direction: SignalDirection,
    output_path: Path,
) -> None:
    width = 1280
    height = 720
    margin_left = 90
    margin_right = 30
    margin_top = 80
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    metric_values = [float(entry["signals"][metric_name]) for entry in entries if metric_name in entry.get("signals", {})]
    if not metric_values:
        output_path.write_text("", encoding="utf-8")
        return

    metric_min = min(metric_values)
    metric_max = max(metric_values)
    if math.isclose(metric_min, metric_max):
        span = max(abs(metric_min), 1.0) * 0.02 or 0.02
        metric_min -= span
        metric_max += span
    else:
        padding = (metric_max - metric_min) * 0.08
        metric_min -= padding
        metric_max += padding

    def x_for(index: int) -> float:
        if len(entries) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / (len(entries) - 1))

    def y_for(value: float) -> float:
        scaled = (value - metric_min) / (metric_max - metric_min)
        return margin_top + plot_height - (scaled * plot_height)

    running_best_points: list[tuple[float, float]] = []
    best_value: float | None = None
    for index, entry in enumerate(entries):
        if metric_name not in entry.get("signals", {}):
            continue
        value = float(entry["signals"][metric_name])
        if best_value is None or _metric_improved(value, best_value, direction):
            best_value = value
            running_best_points.append((x_for(index), y_for(value)))

    y_ticks = []
    for tick_index in range(5):
        ratio = tick_index / 4
        value = metric_min + (metric_max - metric_min) * (1 - ratio)
        y_ticks.append((y_for(value), value))

    lower_is_better = direction == SignalDirection.MINIMIZE
    kept_count = sum(1 for entry in entries if entry.get("accepted"))
    title = f"auto-anything progress: {len(entries)} experiments, {kept_count} kept improvements"
    y_axis_label = f"{metric_name} ({'lower' if lower_is_better else 'higher'} is better)"
    best_path = " ".join(
        ("M" if idx == 0 else "L") + f" {point[0]:.2f} {point[1]:.2f}"
        for idx, point in enumerate(running_best_points)
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="42" font-size="18" font-family="Arial, sans-serif" fill="#111827">{_svg_escape(title)}</text>',
        f'<text x="{width / 2:.2f}" y="58" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#374151">{_svg_escape(metric_name)}</text>',
    ]

    for y_value, label_value in y_ticks:
        parts.append(f'<line x1="{margin_left}" y1="{y_value:.2f}" x2="{width - margin_right}" y2="{y_value:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_value + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#6b7280">{label_value:.3f}</text>'
        )

    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.5"/>')
    parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.5"/>')
    parts.append(
        f'<text x="{width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151">Experiment #</text>'
    )
    parts.append(
        f'<text transform="translate(20 {height / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#374151">{_svg_escape(y_axis_label)}</text>'
    )

    if best_path:
        parts.append(f'<path d="{best_path}" fill="none" stroke="#22c55e" stroke-width="2.5"/>')

    for index, entry in enumerate(entries):
        if metric_name not in entry.get("signals", {}):
            continue
        value = float(entry["signals"][metric_name])
        x = x_for(index)
        y = y_for(value)
        accepted = entry.get("accepted")
        is_running_best = any(abs(px - x) < 0.01 and abs(py - y) < 0.01 for px, py in running_best_points)
        fill = "#22c55e" if accepted else "#d1d5db"
        stroke = "#15803d" if is_running_best else "#9ca3af"
        radius = 4.5 if accepted else 3.2
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 18}" text-anchor="middle" font-size="10" font-family="Arial, sans-serif" fill="#6b7280">{index}</text>'
        )
        label = str(entry.get("label", "")).strip()
        if is_running_best and label:
            parts.append(
                f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-size="10" font-family="Arial, sans-serif" fill="#15803d">{_svg_escape(label)}</text>'
            )

    legend_x = width - 180
    legend_y = margin_top + 10
    parts.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="150" height="72" fill="#ffffff" stroke="#d1d5db"/>',
            f'<circle cx="{legend_x + 16}" cy="{legend_y + 18}" r="3.2" fill="#d1d5db" stroke="#9ca3af" stroke-width="1.2"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 22}" font-size="11" font-family="Arial, sans-serif" fill="#374151">Discarded / not kept</text>',
            f'<circle cx="{legend_x + 16}" cy="{legend_y + 40}" r="4.5" fill="#22c55e" stroke="#15803d" stroke-width="1.2"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 44}" font-size="11" font-family="Arial, sans-serif" fill="#374151">Kept improvement</text>',
            f'<line x1="{legend_x + 8}" y1="{legend_y + 60}" x2="{legend_x + 24}" y2="{legend_y + 60}" stroke="#22c55e" stroke-width="2.5"/>',
            f'<text x="{legend_x + 30}" y="{legend_y + 64}" font-size="11" font-family="Arial, sans-serif" fill="#374151">Running best</text>',
        ]
    )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def record_experiment_result(
    *,
    task_root: Path,
    summary: dict[str, Any],
    metric_name: str,
    direction: SignalDirection,
    label: str = "",
    accepted: bool | None = None,
    focus_subsystems: tuple[str, ...] = (),
    hypothesis: str = "",
    change_summary: str = "",
    knowledge_items: tuple[str, ...] = (),
    decision_reasons: tuple[str, ...] = (),
    blocking_signals: tuple[str, ...] = (),
    blocking_findings: tuple[str, ...] = (),
    touched_paths: tuple[str, ...] = (),
    executed_commands: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    task_root = task_root.expanduser().resolve()
    entries = load_experiment_history(task_root)
    experiment_index = len(entries)
    experiment_id = f"aa-exp-{experiment_index:04d}"
    metric_value = float(summary["signals"][metric_name])

    if accepted is None:
        prior_values = [float(entry["signals"][metric_name]) for entry in entries if entry.get("accepted") and metric_name in entry.get("signals", {})]
        if not prior_values:
            accepted = True
        else:
            baseline_value = prior_values[-1]
            accepted = _metric_improved(metric_value, baseline_value, direction)

    previous_commit = _previous_commit(task_root)
    commit_sha = snapshot_experiment_workspace(
        task_root,
        experiment_id=experiment_id,
        label=label or summary.get("candidate_id", "experiment"),
    )
    diff_info = _collect_diff_summary(task_root, previous_commit, commit_sha)
    entry = {
        "experiment_id": experiment_id,
        "experiment_index": experiment_index,
        "candidate_id": summary.get("candidate_id", f"candidate-{experiment_index:04d}"),
        "label": label or summary.get("candidate_id", f"experiment-{experiment_index:04d}"),
        "accepted": bool(accepted),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_direction": direction.value,
        "signals": dict(summary.get("signals", {})),
        "doc_count": int(summary.get("doc_count", 0)),
        "focus_subsystems": list(focus_subsystems),
        "hypothesis": hypothesis,
        "change_summary": change_summary,
        "knowledge_items": list(knowledge_items),
        "decision_reasons": list(decision_reasons),
        "blocking_signals": list(blocking_signals),
        "blocking_findings": list(blocking_findings),
        "touched_paths": list(touched_paths),
        "executed_commands": list(executed_commands),
        "notes": list(notes),
        "git_commit": commit_sha,
        "previous_commit": diff_info["previous_commit"],
        "diff_stat": diff_info["diff_stat"],
        "changed_files": list(diff_info["changed_files"]),
    }
    entries.append(entry)
    _write_history_views(task_root, entries)
    _write_experiment_report(task_root, entry)
    _write_knowledge_base(task_root, entries)
    _, _, svg_path, _, _ = _history_paths(task_root)
    render_progress_curve_svg(entries, metric_name=metric_name, direction=direction, output_path=svg_path)
    return entry


def record_experiment_record(
    *,
    task_root: Path,
    charter: TaskCharter,
    experiment: ExperimentRecord,
    label: str = "",
    hypothesis: str = "",
    change_summary: str = "",
    knowledge_items: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    metric_name, direction = choose_primary_signal_from_charter(charter)
    candidate_signal_map = experiment.candidate_report.signal_map()
    summary = {
        "candidate_id": experiment.candidate_report.candidate_id,
        "signals": {
            name: signal.value
            for name, signal in candidate_signal_map.items()
        },
        "doc_count": 0,
    }
    touched_paths = tuple(
        path
        for step in experiment.iteration_steps
        for path in step.touched_paths
    )
    executed_commands = tuple(
        command
        for step in experiment.iteration_steps
        for command in step.executed_commands
    )
    return record_experiment_result(
        task_root=task_root,
        summary=summary,
        metric_name=metric_name,
        direction=direction,
        label=label or experiment.candidate_id,
        accepted=experiment.decision.accepted,
        focus_subsystems=experiment.focus_subsystems or experiment.decision.focus_subsystems,
        hypothesis=hypothesis,
        change_summary=change_summary,
        knowledge_items=knowledge_items,
        decision_reasons=experiment.decision.reasons,
        blocking_signals=experiment.decision.blocking_signals,
        blocking_findings=experiment.decision.blocking_findings,
        touched_paths=_dedupe_preserve_order(touched_paths),
        executed_commands=_dedupe_preserve_order(executed_commands),
        notes=notes or experiment.notes,
    )
