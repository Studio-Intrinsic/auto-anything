from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.family_registry import get_default_task_family_registry
from auto_anything.models import DataAsset, EvaluationMode, ObjectiveBrief, ObjectiveSignal, SignalDirection, SignalKind
from auto_anything.task_family import build_bootstrap_plan, infer_evaluation_mode


class TaskFamilyTests(unittest.TestCase):
    def test_infer_evaluation_mode_prefers_explicit_signals(self) -> None:
        brief = ObjectiveBrief(
            title="Explicit benchmark",
            objective_statement="Optimize with explicit accuracy and latency metrics.",
            explicit_signals=(
                ObjectiveSignal(
                    name="accuracy",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MAXIMIZE,
                    description="Accuracy over the benchmark.",
                ),
            ),
        )

        self.assertEqual(infer_evaluation_mode(brief), EvaluationMode.EXPLICIT_BENCHMARK)

    def test_infer_evaluation_mode_detects_partial_labels(self) -> None:
        brief = ObjectiveBrief(
            title="Partially labeled corpus",
            objective_statement="Optimize invoice extraction.",
            data_assets=(
                DataAsset(
                    name="invoices",
                    kind="invoice_corpus",
                    location="/tmp/invoices",
                    role="golden",
                ),
            ),
        )

        self.assertEqual(infer_evaluation_mode(brief), EvaluationMode.PARTIAL_LABELS)

    def test_build_bootstrap_plan_selects_invoice_family(self) -> None:
        registry = get_default_task_family_registry()
        brief = ObjectiveBrief(
            title="Invoice extraction",
            objective_statement="Extract invoice fields from PDFs with low latency.",
            data_assets=(
                DataAsset(
                    name="invoice_corpus",
                    kind="invoice_corpus",
                    location="/tmp/invoices",
                    role="train_eval_corpus",
                ),
            ),
        )

        plan = build_bootstrap_plan(registry, brief)

        self.assertEqual(plan.family_id, "invoice-document-extraction")
        self.assertEqual(plan.scaffold_id, "invoice-vision-workspace")

    def test_default_registry_stays_generic(self) -> None:
        registry = get_default_task_family_registry()
        family_ids = {family.family_id for family in registry.all()}

        self.assertEqual(family_ids, {"invoice-document-extraction"})


if __name__ == "__main__":
    unittest.main()
