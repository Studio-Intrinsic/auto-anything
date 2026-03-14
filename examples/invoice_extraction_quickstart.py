from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_DATA_DIR = ROOT / "examples" / "sample_data"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.invoice_bootstrap import (
    DEFAULT_MODEL,
    InvoiceExtractionSkill,
    build_invoice_objective_brief,
    load_env_file,
)
from auto_anything.compiler import DefaultObjectiveCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a plain-English invoice extraction task into a self-play charter.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing invoice documents and labels. Defaults to the bundled sample invoice directory.",
    )
    parser.add_argument(
        "--objective",
        required=True,
        help="Plain-English objective, e.g. 'Extract invoice fields accurately while staying scalable and cheap.'",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Primary OpenRouter model to use when the agent or pipeline needs model calls.",
    )
    parser.add_argument(
        "--charter-json",
        default="",
        help="Optional path to write the compiled charter JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(ROOT / ".env.local")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    brief = build_invoice_objective_brief(
        data_dir=Path(args.data_dir),
        objective=args.objective,
        model=args.model,
    )

    charter = DefaultObjectiveCompiler().compile(brief, skills=(InvoiceExtractionSkill(),))
    payload = asdict(charter)
    rendered = json.dumps(payload, indent=2, default=str)

    if args.charter_json:
        output_path = Path(args.charter_json).expanduser().resolve()
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote charter to {output_path}")
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
