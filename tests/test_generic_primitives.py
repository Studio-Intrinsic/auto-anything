from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.data_slicing import (
    load_json_manifest,
    sample_rows,
    slice_rows,
    stratified_sample_rows,
    write_json_manifest,
)
from auto_anything.invoice_bootstrap import bootstrap_invoice_task, run_bootstrapped_eval
from auto_anything.model_probe import ProbeExample, ProbeExecutionMeta, probe_candidates, write_probe_report
from auto_anything.serialization import load_task_charter
from auto_anything.task_iteration import run_self_critic


class DataSlicingTests(unittest.TestCase):
    def test_manifest_roundtrip_and_sampling(self) -> None:
        rows = [
            {"id": "a", "bucket": "x"},
            {"id": "b", "bucket": "x"},
            {"id": "c", "bucket": "y"},
            {"id": "d", "bucket": "y"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            write_json_manifest(path, rows)
            loaded = load_json_manifest(path)
            self.assertEqual(loaded, rows)
            self.assertEqual([row["id"] for row in slice_rows(loaded, offset=1, limit=2)], ["b", "c"])
            self.assertEqual(len(sample_rows(loaded, count=2, seed=7)), 2)
            stratified = stratified_sample_rows(loaded, count=2, key_fn=lambda row: row["bucket"], seed=7)
            self.assertEqual(len({row["bucket"] for row in stratified}), 2)


class ModelProbeTests(unittest.TestCase):
    def test_probe_candidates_ranks_by_score_then_cost(self) -> None:
        examples = (
            ProbeExample(example_id="one", input_data=1, expected_output=2),
            ProbeExample(example_id="two", input_data=2, expected_output=4),
        )

        def runner(candidate_id: str, example: ProbeExample):
            multiplier = {"fast-cheap": 2, "slow-better": 2}.get(candidate_id, 1)
            cost = {"fast-cheap": 0.1, "slow-better": 0.2}.get(candidate_id, 0.3)
            latency = {"fast-cheap": 1.0, "slow-better": 2.0}.get(candidate_id, 3.0)
            return example.input_data * multiplier, ProbeExecutionMeta(latency_seconds=latency, cost_usd=cost)

        def scorer(output: object, example: ProbeExample) -> float:
            return 1.0 if output == example.expected_output else 0.0

        summaries = probe_candidates(
            candidate_ids=("bad", "slow-better", "fast-cheap"),
            examples=examples,
            runner=runner,
            scorer=scorer,
        )
        self.assertEqual([summary.candidate_id for summary in summaries], ["fast-cheap", "slow-better", "bad"])
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = write_probe_report(Path(tmpdir) / "probe.json", summaries)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["candidate_id"], "fast-cheap")


class IterationBookkeepingTests(unittest.TestCase):
    def test_self_critic_does_not_penalize_broad_cross_subsystem_work_as_focus_drift(self) -> None:
        sample_data = ROOT / "examples" / "sample_data"
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir) / "invoice_task"
            bootstrap_invoice_task(
                task_root=task_root,
                data_dir=sample_data,
                objective="Extract invoice fields accurately while staying scalable and cheap.",
            )
            run_bootstrapped_eval(task_root)
            charter = load_task_charter(task_root / "task_charter.json")
            report = run_self_critic(
                task_root=task_root,
                charter=charter,
                touched_paths=(
                    "src/invoice_pipeline/document_io.py",
                    "src/invoice_pipeline/field_extractors.py",
                ),
                focus_subsystems=("normalization",),
                change_summary="Broader cleanup across ingestion and extraction.",
            )
            self.assertFalse(any(finding.finding_id == "focus-drift" for finding in report.findings))
            self.assertTrue(any("broader-scope work" in note for note in report.notes))


if __name__ == "__main__":
    unittest.main()
