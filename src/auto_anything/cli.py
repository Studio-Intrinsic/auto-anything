"""auto-anything CLI.

Usage:
    auto-anything bootstrap --objective "..." [--path ...] [options]
    auto-anything iterate   --task-root DIR --hypothesis "..." --change-summary "..." [options]
    auto-anything baseline  --task-root DIR [options]
    auto-anything status    --task-root DIR [options]
    auto-anything history   --task-root DIR [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_env() -> None:
    """Load .env.local from cwd or auto-anything package root."""
    for candidate in (Path.cwd() / ".env.local", Path(__file__).resolve().parents[2] / ".env.local"):
        if candidate.is_file():
            from .invoice_bootstrap import load_env_file

            load_env_file(candidate)
            return


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _add_bootstrap_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("bootstrap", help="Create a task workspace from a plain-English objective.")
    p.add_argument("--objective", required=True, help="Plain-English task objective.")
    p.add_argument("--path", action="append", default=[], help="Referenced file or directory (repeatable).")
    p.add_argument("--task-root", default="", help="Destination workspace path.")
    p.add_argument("--title", default="", help="Short title for the task.")
    p.add_argument("--anti-goal", action="append", default=[], help="Anti-goal (repeatable).")
    p.add_argument("--constraint", action="append", default=[], help="Constraint (repeatable).")
    p.add_argument("--focus-subsystem", action="append", default=[], help="Subsystem to focus on first (repeatable).")
    p.add_argument("--model", default="", help="Preferred default model.")
    p.add_argument("--skip-eval", action="store_true", help="Only scaffold; do not run baseline eval.")
    p.set_defaults(func=_cmd_bootstrap)


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    from .invoice_bootstrap import DEFAULT_MODEL
    from .request_bootstrap import (
        PlainTextTaskRequest,
        bootstrap_task_from_request,
        build_bootstrap_plan_from_request,
    )
    from .task_iteration import run_task_baseline

    model = args.model or DEFAULT_MODEL
    paths = tuple(args.path) if args.path else ()

    request = PlainTextTaskRequest(
        objective_statement=args.objective,
        referenced_paths=paths,
        task_root=args.task_root,
        title=args.title,
        anti_goals=tuple(args.anti_goal),
        constraints=tuple(args.constraint),
        allowed_models=(model,),
        focus_subsystems=tuple(args.focus_subsystem),
    )

    plan = build_bootstrap_plan_from_request(request)
    task_root = bootstrap_task_from_request(request)

    print(f"Workspace:  {task_root}")
    print(f"Charter:    {task_root / 'task_charter.json'}")
    print(f"AGENTS.md:  {task_root / 'AGENTS.md'}")
    print(f"CLAUDE.md:  {task_root / 'CLAUDE.md'}")
    print(f"Family:     {plan.family_id}")
    print(f"Eval mode:  {plan.evaluation_mode.value}")
    for line in plan.rationale:
        print(f"  - {line}")

    if args.skip_eval:
        print("\nSkipped baseline eval. Run: auto-anything baseline --task-root", task_root)
    else:
        print("\nRunning baseline evaluation...")
        baseline = run_task_baseline(task_root=task_root)
        _print_eval_summary(task_root, baseline["summary"])

    print(f"\ncd {task_root}")
    return 0


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def _add_baseline_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("baseline", help="Run the baseline evaluation on a bootstrapped workspace.")
    p.add_argument("--task-root", required=True, help="Bootstrapped task workspace.")
    p.add_argument("--command-name", default="evaluate", help="Run command name from charter.")
    p.add_argument("--label", default="", help="Label for this baseline run.")
    p.set_defaults(func=_cmd_baseline)


def _cmd_baseline(args: argparse.Namespace) -> int:
    from .task_iteration import run_task_baseline

    task_root = Path(args.task_root).expanduser().resolve()
    if not (task_root / "task_charter.json").is_file():
        print(f"No task_charter.json in {task_root}. Run 'auto-anything bootstrap' first.", file=sys.stderr)
        return 1

    result = run_task_baseline(
        task_root=task_root,
        command_name=args.command_name,
        label=args.label,
    )
    _print_eval_summary(task_root, result["summary"])
    return 0


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------


def _add_iterate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("iterate", help="Evaluate changes, run self-critic, and record an experiment.")
    p.add_argument("--task-root", required=True, help="Bootstrapped task workspace.")
    p.add_argument("--hypothesis", required=True, help="What you expect this change to improve.")
    p.add_argument("--change-summary", required=True, help="Short description of what changed.")
    p.add_argument("--label", default="", help="Optional experiment label.")
    p.add_argument("--command-name", default="evaluate", help="Run command name from charter.")
    p.add_argument("--focus-subsystem", action="append", default=[], help="Subsystem focus (repeatable).")
    p.add_argument("--note", action="append", default=[], help="Note to attach (repeatable).")
    p.set_defaults(func=_cmd_iterate)


def _cmd_iterate(args: argparse.Namespace) -> int:
    from .task_iteration import run_task_iteration

    task_root = Path(args.task_root).expanduser().resolve()
    if not (task_root / "task_charter.json").is_file():
        print(f"No task_charter.json in {task_root}. Run 'auto-anything bootstrap' first.", file=sys.stderr)
        return 1

    result = run_task_iteration(
        task_root=task_root,
        hypothesis=args.hypothesis,
        change_summary=args.change_summary,
        label=args.label,
        focus_subsystems=tuple(args.focus_subsystem),
        notes=tuple(args.note),
        command_name=args.command_name,
    )

    decision = result["decision"]
    accepted = decision["accepted"]
    verdict = "ACCEPTED" if accepted else "REJECTED"
    print(f"\n{'=' * 60}")
    print(f"  {verdict}")
    print(f"{'=' * 60}")
    print(f"  Utility gain: {decision['utility_gain']:.4f}")
    if decision["reasons"]:
        print(f"  Reasons: {', '.join(decision['reasons'])}")
    if decision["blocking_signals"]:
        print(f"  Blocking signals: {', '.join(decision['blocking_signals'])}")
    if decision["blocking_findings"]:
        print(f"  Blocking findings: {', '.join(decision['blocking_findings'])}")

    # Per-signal breakdown so the agent can see what drove the decision
    baseline_signals = result.get("baseline_signals", {})
    candidate_signals = result.get("candidate_signals", {})
    if baseline_signals and candidate_signals:
        print(f"\n  Signal breakdown:")
        for name in sorted(set(baseline_signals) | set(candidate_signals)):
            bv = baseline_signals.get(name, 0.0)
            cv = candidate_signals.get(name, 0.0)
            delta = cv - bv
            direction = "+" if delta >= 0 else ""
            print(f"    {name}: {bv} -> {cv} ({direction}{delta:.4f})")
    print()

    if result.get("context"):
        print("Experiment context:")
        print(result["context"])
        print()

    print(f"History:  {task_root / 'artifacts' / 'experiment_history.json'}")
    print(f"Knowledge: {task_root / 'artifacts' / 'knowledge_base.md'}")
    print(f"Curve:    {task_root / 'artifacts' / 'progress_curve.svg'}")
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _add_status_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("status", help="Show current experiment state, best result, and what to try next.")
    p.add_argument("--task-root", required=True, help="Bootstrapped task workspace.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON.")
    p.set_defaults(func=_cmd_status)


def _cmd_status(args: argparse.Namespace) -> int:
    from .history import build_experiment_context, load_experiment_history
    from .serialization import load_task_charter

    task_root = Path(args.task_root).expanduser().resolve()
    if not (task_root / "task_charter.json").is_file():
        print(f"No task_charter.json in {task_root}.", file=sys.stderr)
        return 1

    charter = load_task_charter(task_root / "task_charter.json")
    entries = load_experiment_history(task_root)

    if args.as_json:
        kept = [e for e in entries if e.get("accepted")]
        best = kept[-1] if kept else (entries[-1] if entries else None)
        print(json.dumps({
            "task_root": str(task_root),
            "objective": charter.objective_statement,
            "total_experiments": len(entries),
            "accepted_experiments": len(kept),
            "best_experiment": best,
            "recent_experiments": entries[-5:],
            "signals": [s.name for s in charter.evaluation_plan.signals],
            "subsystems": [s.subsystem_id for s in charter.search_surface.subsystems],
            "focus_subsystems": list(charter.focus_subsystems),
        }, indent=2, sort_keys=True))
        return 0

    print(f"Task:       {charter.title}")
    print(f"Objective:  {charter.objective_statement}")
    print(f"Workspace:  {task_root}")
    print()

    if not entries:
        print("No experiments recorded yet.")
        print(f"\nRun: auto-anything baseline --task-root {task_root}")
        return 0

    kept = [e for e in entries if e.get("accepted")]
    rejected = [e for e in entries if not e.get("accepted")]
    print(f"Experiments: {len(entries)} total, {len(kept)} accepted, {len(rejected)} rejected")

    best = kept[-1] if kept else entries[-1]
    print(f"\nBest experiment: {best['experiment_id']}")
    print(f"  Metric: {best.get('metric_name', '?')} = {best.get('metric_value', '?')}")
    print(f"  Commit: {best.get('git_commit', '?')[:12]}")
    if best.get("change_summary"):
        print(f"  Change: {best['change_summary']}")

    # Signal overview from best
    signals = best.get("signals", {})
    if signals:
        print(f"\nSignals (best):")
        for name, value in sorted(signals.items()):
            print(f"  {name}: {value}")

    # Subsystem headroom
    subsystems = charter.search_surface.subsystems
    if subsystems:
        print(f"\nSubsystems:")
        subsystem_stats: dict[str, dict] = {}
        for sub in subsystems:
            sub_entries = [e for e in entries if sub.subsystem_id in e.get("focus_subsystems", [])]
            subsystem_stats[sub.subsystem_id] = {
                "total": len(sub_entries),
                "accepted": sum(1 for e in sub_entries if e.get("accepted")),
                "summary": sub.summary,
            }
        for sub_id, stats in sorted(subsystem_stats.items()):
            marker = "*" if sub_id in charter.focus_subsystems else " "
            print(f"  {marker} {sub_id}: {stats['total']} experiments ({stats['accepted']} accepted) - {stats['summary']}")

    # Recent trail
    recent = entries[-5:]
    print(f"\nRecent experiments:")
    for entry in recent:
        status = "+" if entry.get("accepted") else "-"
        metric = entry.get("metric_value", "?")
        change = entry.get("change_summary", "")[:60]
        print(f"  {status} {entry['experiment_id']} metric={metric} {change}")

    # Suggestions
    print(f"\nContext:")
    print(build_experiment_context(task_root))
    return 0


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def _add_history_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("history", help="Show full experiment history.")
    p.add_argument("--task-root", required=True, help="Bootstrapped task workspace.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON.")
    p.add_argument("--limit", type=int, default=0, help="Show only the last N experiments.")
    p.set_defaults(func=_cmd_history)


def _cmd_history(args: argparse.Namespace) -> int:
    from .history import load_experiment_history

    task_root = Path(args.task_root).expanduser().resolve()
    entries = load_experiment_history(task_root)

    if args.limit > 0:
        entries = entries[-args.limit:]

    if args.as_json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return 0

    if not entries:
        print("No experiments recorded.")
        return 0

    for entry in entries:
        status = "ACCEPTED" if entry.get("accepted") else "REJECTED"
        experiment_id = entry.get("experiment_id", "?")
        metric_name = entry.get("metric_name", "?")
        metric_value = entry.get("metric_value", "?")
        label = entry.get("label", "")
        hypothesis = entry.get("hypothesis", "")
        change = entry.get("change_summary", "")
        reasons = ", ".join(entry.get("decision_reasons", []))
        focus = ", ".join(entry.get("focus_subsystems", []))
        commit = entry.get("git_commit", "")[:12]

        print(f"{experiment_id}  {status}  {metric_name}={metric_value}  commit={commit}")
        if label:
            print(f"  label: {label}")
        if focus:
            print(f"  focus: {focus}")
        if hypothesis:
            print(f"  hypothesis: {hypothesis}")
        if change:
            print(f"  change: {change}")
        if reasons:
            print(f"  reasons: {reasons}")

        learnings = entry.get("knowledge_items", [])
        for item in learnings[:2]:
            print(f"  learned: {item}")
        print()

    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _print_eval_summary(task_root: Path, summary: dict) -> None:
    signals = summary.get("signals", {})
    print(f"\nEval summary ({summary.get('candidate_id', '?')}):")
    for name, value in sorted(signals.items()):
        print(f"  {name}: {value}")
    print(f"\nArtifacts:")
    print(f"  {task_root / 'artifacts' / 'eval_summary.json'}")
    print(f"  {task_root / 'artifacts' / 'experiment_history.json'}")
    print(f"  {task_root / 'artifacts' / 'knowledge_base.md'}")
    print(f"  {task_root / 'artifacts' / 'progress_curve.svg'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-anything",
        description="Turn high-level objectives into agent-optimizable experiment loops.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_bootstrap_parser(subparsers)
    _add_baseline_parser(subparsers)
    _add_iterate_parser(subparsers)
    _add_status_parser(subparsers)
    _add_history_parser(subparsers)
    return parser


def main() -> int:
    _load_env()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
