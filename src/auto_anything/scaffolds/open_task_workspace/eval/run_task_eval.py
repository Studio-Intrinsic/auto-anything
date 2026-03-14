from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LIB_SRC = Path("__AUTO_ANYTHING_LIBRARY_SRC__")
SRC = ROOT / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.history import choose_primary_signal_from_charter, record_experiment_result
from task_pipeline.pipeline import run_candidate
from task_pipeline.postprocess import postprocess_candidate_output


def main() -> int:
    charter = json.loads((ROOT / "task_charter.json").read_text(encoding="utf-8"))
    candidate_output = postprocess_candidate_output(run_candidate(ROOT))
    signals = {signal["name"]: 0.0 for signal in charter["evaluation_plan"]["signals"]}
    summary = {
        "candidate_id": "baseline",
        "signals": signals,
        "notes": [
            "Placeholder evaluator. Replace eval/run_task_eval.py with a task-specific harness before trusting metrics.",
        ],
        "candidate_output_preview": candidate_output,
    }
    summary_path = ROOT / "artifacts" / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if os.getenv("AUTO_ANYTHING_SKIP_AUTO_RECORD", "").strip() not in {"1", "true", "TRUE"}:
        metric_name, direction = choose_primary_signal_from_charter(charter)
        record_experiment_result(
            task_root=ROOT,
            summary=summary,
            metric_name=metric_name,
            direction=direction,
            label=summary["candidate_id"],
            focus_subsystems=tuple(charter.get("focus_subsystems", [])),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
