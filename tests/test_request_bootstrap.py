from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.request_bootstrap import (
    PlainTextTaskRequest,
    bootstrap_task_from_request,
    build_bootstrap_plan_from_request,
    build_brief_from_request,
    infer_task_root,
)
from auto_anything.task_iteration import run_task_baseline


class RequestBootstrapTests(unittest.TestCase):
    def test_build_brief_from_plain_text_request_infers_invoice_asset(self) -> None:
        sample_pdf = ROOT / "examples" / "sample_data" / "sample_invoice.pdf"
        request = PlainTextTaskRequest(
            objective_statement="Extract invoice fields accurately while staying cheap and fast.",
            referenced_paths=(str(sample_pdf),),
        )

        brief = build_brief_from_request(request)
        plan = build_bootstrap_plan_from_request(request)

        self.assertEqual(brief.data_assets[0].kind, "pdf_document")
        self.assertEqual(brief.allowed_models, ("x-ai/grok-4.1-fast",))
        self.assertEqual(plan.family_id, "open-ended-task")

    def test_infer_task_root_uses_objective_slug(self) -> None:
        request = PlainTextTaskRequest(objective_statement="Optimize invoice extraction for speed and accuracy.")

        task_root = infer_task_root(request)

        self.assertEqual(task_root.name, "optimize-invoice-extraction-for-speed-and-accuracy")

    def test_build_brief_from_request_rejects_missing_paths(self) -> None:
        request = PlainTextTaskRequest(
            objective_statement="Extract invoice fields.",
            referenced_paths=("/tmp/does-not-exist-auto-anything.pdf",),
        )

        with self.assertRaises(FileNotFoundError):
            build_brief_from_request(request)

    def test_bootstrap_task_from_request_creates_runnable_workspace(self) -> None:
        sample_dir = ROOT / "examples" / "sample_data"
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir) / "request_task"
            request = PlainTextTaskRequest(
                objective_statement="Extract invoice fields accurately while staying scalable and cheap.",
                referenced_paths=(str(sample_dir),),
                task_root=str(task_root),
            )

            bootstrapped = bootstrap_task_from_request(request)
            baseline = run_task_baseline(task_root=bootstrapped)

            summary = json.loads((bootstrapped / "artifacts" / "eval_summary.json").read_text(encoding="utf-8"))
            history = json.loads((bootstrapped / "artifacts" / "experiment_history.json").read_text(encoding="utf-8"))

            self.assertEqual(bootstrapped, task_root.resolve())
            self.assertEqual(baseline["execution"].returncode, 0)
            self.assertTrue((bootstrapped / "AGENTS.md").is_file())
            self.assertTrue((bootstrapped / "src" / "task_pipeline" / "pipeline.py").is_file())
            self.assertTrue((bootstrapped / "eval" / "run_task_eval.py").is_file())
            self.assertEqual(summary["signals"]["task_quality"], 0.0)
            self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
