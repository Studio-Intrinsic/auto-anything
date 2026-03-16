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

from auto_anything.invoice_bootstrap import bootstrap_invoice_task, run_bootstrapped_eval


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_creates_runnable_invoice_workspace(self) -> None:
        sample_data = ROOT / "examples" / "sample_data"
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir) / "invoice_task"
            bootstrapped = bootstrap_invoice_task(
                task_root=task_root,
                data_dir=sample_data,
                objective="Extract invoice fields accurately while staying scalable and cheap.",
            )

            self.assertEqual(bootstrapped, task_root.resolve())
            self.assertTrue((task_root / "src" / "invoice_pipeline" / "extract.py").is_file())
            self.assertTrue((task_root / "src" / "invoice_pipeline" / "document_io.py").is_file())
            self.assertTrue((task_root / "src" / "invoice_pipeline" / "field_extractors.py").is_file())
            self.assertTrue((task_root / "src" / "invoice_pipeline" / "normalization.py").is_file())
            self.assertTrue((task_root / "src" / "invoice_pipeline" / "openrouter_client.py").is_file())
            self.assertTrue((task_root / "src" / "invoice_pipeline" / "schema.py").is_file())
            self.assertTrue((task_root / "eval" / "run_invoice_eval.py").is_file())
            self.assertTrue((task_root / "AGENTS.md").is_file())
            self.assertTrue((task_root / "fixtures" / "sample_invoice.pdf").is_file())
            self.assertTrue((task_root / "goldens" / "sample_invoice.expected.json").is_file())
            agents_text = (task_root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("## Mutable Surface", agents_text)
            self.assertIn("## Optimizable Artifacts", agents_text)
            self.assertIn("src/invoice_pipeline", agents_text)
            self.assertIn("auto-anything iterate --task-root", agents_text)
            self.assertIn("list_openrouter_models()", agents_text)
            self.assertIn("list_artificial_analysis_llms()", agents_text)

            completed = run_bootstrapped_eval(task_root)
            summary = json.loads((task_root / "artifacts" / "eval_summary.json").read_text(encoding="utf-8"))
            history = json.loads((task_root / "artifacts" / "experiment_history.json").read_text(encoding="utf-8"))

            self.assertEqual(completed.returncode, 0)
            self.assertGreaterEqual(summary["signals"]["field_accuracy"], 0.99)
            self.assertEqual(summary["signals"]["schema_valid"], 1.0)
            self.assertTrue((task_root / ".git").is_dir())
            self.assertTrue((task_root / "artifacts" / "progress_curve.svg").is_file())
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["metric_name"], "field_accuracy")


if __name__ == "__main__":
    unittest.main()
