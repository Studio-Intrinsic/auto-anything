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

from auto_anything.history import load_experiment_history
from auto_anything.invoice_bootstrap import bootstrap_invoice_task, run_bootstrapped_eval
from auto_anything.invoice_iteration import run_invoice_iteration


class InvoiceIterationTests(unittest.TestCase):
    def test_run_invoice_iteration_records_authoritative_experiment(self) -> None:
        sample_data = ROOT / "examples" / "sample_data"
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir) / "invoice_task"
            bootstrap_invoice_task(
                task_root=task_root,
                data_dir=sample_data,
                objective="Extract invoice fields accurately while staying scalable and cheap.",
            )
            run_bootstrapped_eval(task_root)
            normalization_path = task_root / "src" / "invoice_pipeline" / "normalization.py"
            normalization_path.write_text(
                normalization_path.read_text(encoding="utf-8") + "\n# iteration touch\n",
                encoding="utf-8",
            )

            result = run_invoice_iteration(
                task_root=task_root,
                hypothesis="A small normalization adjustment may improve stability.",
                change_summary="Touched normalization to prepare for future parsing cleanup.",
                label="iteration-1",
                focus_subsystems=("normalization",),
                notes=("Testing authoritative iteration recording.",),
            )

            history = load_experiment_history(task_root)
            latest = history[-1]

            self.assertEqual(len(history), 2)
            self.assertFalse(result["decision"]["accepted"])
            self.assertIn("no_positive_utility_gain", result["decision"]["reasons"])
            self.assertEqual(latest["label"], "iteration-1")
            self.assertEqual(
                latest["hypothesis"],
                "A small normalization adjustment may improve stability.",
            )
            self.assertIn("normalization", latest["focus_subsystems"])
            self.assertIn("src/invoice_pipeline/normalization.py", latest["touched_paths"])
            self.assertTrue((task_root / "artifacts" / "experiments" / f"{latest['experiment_id']}.md").is_file())
            self.assertTrue((task_root / "artifacts" / "knowledge_base.md").is_file())
            best = json.loads((task_root / "artifacts" / "best_experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(best["experiment_id"], "aa-exp-0000")


if __name__ == "__main__":
    unittest.main()
