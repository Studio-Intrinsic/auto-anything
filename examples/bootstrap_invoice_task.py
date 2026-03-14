from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_DATA_DIR = ROOT / "examples" / "sample_data"
DEFAULT_TASK_ROOT = ROOT / "work" / "invoice_extraction_demo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.invoice_bootstrap import (
    DEFAULT_MODEL,
    bootstrap_invoice_task,
    load_env_file,
    run_bootstrapped_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap a starter invoice extraction workspace from plain-English intent.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing invoice PDFs and optional gold JSON.")
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT), help="Directory to create the starter task workspace in.")
    parser.add_argument("--objective", required=True, help="Plain-English goal for the invoice extraction task.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default OpenRouter model for the bootstrapped task.")
    parser.add_argument(
        "--focus-subsystem",
        action="append",
        default=[],
        help="Optional subsystem id to optimize first. May be passed multiple times.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Only scaffold the task workspace; do not run the baseline eval.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(ROOT / ".env.local")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    task_root = bootstrap_invoice_task(
        task_root=Path(args.task_root),
        data_dir=Path(args.data_dir),
        objective=args.objective,
        model=args.model,
        focus_subsystems=tuple(args.focus_subsystem),
    )
    print(f"Bootstrapped task at {task_root}")
    print(f"Task charter: {task_root / 'task_charter.json'}")
    print(f"Editable pipeline: {task_root / 'src' / 'invoice_pipeline' / 'extract.py'}")

    if args.skip_eval:
        return 0

    completed = run_bootstrapped_eval(task_root)
    print(completed.stdout)
    print(f"Eval summary: {task_root / 'artifacts' / 'eval_summary.json'}")
    print(f"Experiment history: {task_root / 'artifacts' / 'experiment_history.json'}")
    print(f"Knowledge base: {task_root / 'artifacts' / 'knowledge_base.md'}")
    print(f"Progress curve: {task_root / 'artifacts' / 'progress_curve.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
