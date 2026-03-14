from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_TASK_ROOT = ROOT / "work" / "invoice_extraction_demo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.invoice_bootstrap import load_env_file
from auto_anything.invoice_iteration import run_invoice_iteration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record an authoritative invoice experiment iteration.")
    parser.add_argument("--task-root", default=str(DEFAULT_TASK_ROOT), help="Bootstrapped invoice task workspace.")
    parser.add_argument("--hypothesis", required=True, help="What you expect this iteration to improve.")
    parser.add_argument("--change-summary", required=True, help="Short summary of the changes made before evaluation.")
    parser.add_argument("--label", default="", help="Optional experiment label.")
    parser.add_argument(
        "--focus-subsystem",
        action="append",
        default=[],
        help="Optional subsystem id to evaluate this iteration against. May be passed multiple times.",
    )
    parser.add_argument("--note", action="append", default=[], help="Optional note to attach to the experiment memory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(ROOT / ".env.local")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    result = run_invoice_iteration(
        task_root=Path(args.task_root),
        hypothesis=args.hypothesis,
        change_summary=args.change_summary,
        label=args.label,
        focus_subsystems=tuple(args.focus_subsystem),
        notes=tuple(args.note),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"Experiment record: {Path(args.task_root) / 'artifacts' / 'experiment_history.json'}")
    print(f"Knowledge base: {Path(args.task_root) / 'artifacts' / 'knowledge_base.md'}")
    print(f"Progress curve: {Path(args.task_root) / 'artifacts' / 'progress_curve.svg'}")
    print("Context:")
    print(result["context"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
