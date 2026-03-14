from __future__ import annotations

import json
import os
import time
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
from invoice_pipeline.extract import extract_invoice_with_meta


def _normalize(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def main() -> int:
    fixtures_dir = ROOT / "fixtures"
    goldens_dir = ROOT / "goldens"
    pdf_paths = sorted(fixtures_dir.glob("*.pdf"))

    total_expected = 0
    total_correct = 0
    passed_docs = 0
    schema_valid = 1.0
    latencies = []
    token_totals = []

    predictions_dir = ROOT / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        gold_path = goldens_dir / f"{pdf_path.stem}.expected.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.is_file() else {}
        t0 = time.perf_counter()
        prediction, meta = extract_invoice_with_meta(pdf_path)
        latencies.append(time.perf_counter() - t0)
        token_totals.append(int(meta.get("usage", {}).get("total_tokens", 0) or 0))
        (predictions_dir / f"{pdf_path.stem}.predicted.json").write_text(
            json.dumps(prediction, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        doc_expected = 0
        doc_correct = 0
        for key, expected_value in gold.items():
            doc_expected += 1
            total_expected += 1
            if _normalize(prediction.get(key, "")) == _normalize(expected_value):
                doc_correct += 1
                total_correct += 1

        if doc_expected > 0 and doc_correct == doc_expected:
            passed_docs += 1

        if not isinstance(prediction, dict):
            schema_valid = 0.0

    doc_count = len(pdf_paths)
    field_accuracy = (total_correct / total_expected) if total_expected else 0.0
    document_pass_rate = (passed_docs / doc_count) if doc_count else 0.0
    latency_seconds = (sum(latencies) / len(latencies)) if latencies else 0.0
    token_cost = (sum(token_totals) / len(token_totals)) if token_totals else 0.0

    summary = {
        "candidate_id": "baseline",
        "signals": {
            "field_accuracy": field_accuracy,
            "document_pass_rate": document_pass_rate,
            "schema_valid": schema_valid,
            "latency_seconds": latency_seconds,
            "token_cost": token_cost,
        },
        "doc_count": doc_count,
        "total_expected_fields": total_expected,
        "total_correct_fields": total_correct,
    }
    summary_path = ROOT / "artifacts" / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if os.getenv("AUTO_ANYTHING_SKIP_AUTO_RECORD", "").strip() not in {"1", "true", "TRUE"}:
        charter = json.loads((ROOT / "task_charter.json").read_text(encoding="utf-8"))
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
