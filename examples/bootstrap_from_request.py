from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_SAMPLE_PATH = ROOT / "examples" / "sample_data" / "sample_invoice.pdf"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything import (
    DEFAULT_MODEL,
    PlainTextTaskRequest,
    bootstrap_task_from_request,
    build_bootstrap_plan_from_request,
    infer_task_root,
    load_env_file,
    run_task_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap a task workspace from plain-English intent plus referenced files."
    )
    parser.add_argument("--objective", required=True, help="Plain-English task objective.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Referenced file or directory. May be passed multiple times.",
    )
    parser.add_argument("--task-root", default="", help="Destination task workspace.")
    parser.add_argument("--title", default="", help="Optional shorter title for the task.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Preferred default model.")
    parser.add_argument("--anti-goal", action="append", default=[], help="Optional anti-goal. May be passed multiple times.")
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Optional constraint. May be passed multiple times.",
    )
    parser.add_argument(
        "--focus-subsystem",
        action="append",
        default=[],
        help="Optional subsystem to optimize first. May be passed multiple times.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Only bootstrap the workspace.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(ROOT / ".env.local")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set. Continuing; some pipelines may fall back to non-LLM behavior.", file=sys.stderr)

    request = PlainTextTaskRequest(
        objective_statement=args.objective,
        referenced_paths=tuple(args.path or [str(DEFAULT_SAMPLE_PATH)]),
        task_root=args.task_root,
        title=args.title,
        anti_goals=tuple(args.anti_goal),
        constraints=tuple(args.constraint),
        allowed_models=(args.model,),
        focus_subsystems=tuple(args.focus_subsystem),
    )
    plan = build_bootstrap_plan_from_request(request)
    task_root = bootstrap_task_from_request(request)

    print(f"Bootstrap mode: {plan.family_id}")
    print(f"Inferred evaluation mode: {plan.evaluation_mode.value}")
    print(f"Workspace: {task_root}")
    print(f"Task charter: {task_root / 'task_charter.json'}")
    print(f"Agent handoff: {task_root / 'AGENTS.md'}")
    print(f"Suggested default task root: {infer_task_root(request)}")
    for rationale in plan.rationale:
        print(f"- {rationale}")

    if args.skip_eval:
        return 0

    baseline = run_task_baseline(task_root=task_root)
    print(baseline["execution"].stdout)
    print(f"Eval summary: {task_root / 'artifacts' / 'eval_summary.json'}")
    print(f"Experiment history: {task_root / 'artifacts' / 'experiment_history.json'}")
    print(f"Knowledge base: {task_root / 'artifacts' / 'knowledge_base.md'}")
    print(f"Progress curve: {task_root / 'artifacts' / 'progress_curve.svg'}")
    print("Next step: open AGENTS.md in the workspace and complete the evaluator/pipeline setup it calls out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
