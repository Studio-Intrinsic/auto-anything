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

from auto_anything.history import (
    build_experiment_context,
    choose_primary_signal_from_charter,
    load_experiment_history,
    record_experiment_record,
    record_experiment_result,
)
from auto_anything.models import (
    AcceptanceDecision,
    EvaluationPlan,
    EvaluationReport,
    ExperimentRecord,
    IterationStep,
    ObjectiveSignal,
    SearchSurface,
    SignalDirection,
    SignalKind,
    SignalResult,
    TaskCharter,
    WorkspaceLayout,
)


class HistoryTests(unittest.TestCase):
    def test_records_git_backed_experiment_history_and_progress_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir)
            (task_root / "task_charter.json").write_text(
                json.dumps(
                    {
                        "evaluation_plan": {
                            "signals": [
                                {
                                    "name": "quality",
                                    "direction": "maximize",
                                    "weight": 1.0,
                                    "hard_gate": False,
                                },
                                {
                                    "name": "schema_valid",
                                    "direction": "satisfy",
                                    "weight": 1.0,
                                    "hard_gate": True,
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (task_root / "pipeline.py").write_text("print('baseline')\n", encoding="utf-8")

            metric_name, direction = choose_primary_signal_from_charter(
                json.loads((task_root / "task_charter.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(metric_name, "quality")

            record_experiment_result(
                task_root=task_root,
                summary={"candidate_id": "baseline", "signals": {"quality": 0.70, "schema_valid": 1.0}},
                metric_name=metric_name,
                direction=direction,
                label="baseline",
                hypothesis="Establish a baseline extraction pipeline.",
                change_summary="Initial pipeline snapshot.",
                knowledge_items=("Baseline quality is below target.",),
            )
            (task_root / "pipeline.py").write_text("print('improved')\n", encoding="utf-8")
            record_experiment_result(
                task_root=task_root,
                summary={"candidate_id": "candidate", "signals": {"quality": 0.82, "schema_valid": 1.0}},
                metric_name=metric_name,
                direction=SignalDirection.MAXIMIZE,
                label="candidate",
                hypothesis="Improve quality with a narrower field parser.",
                change_summary="Refined the extraction logic.",
                knowledge_items=("Narrower parsing improved quality on the current corpus.",),
                touched_paths=("pipeline.py",),
            )

            history = load_experiment_history(task_root)
            context = build_experiment_context(task_root)

            self.assertEqual(len(history), 2)
            self.assertTrue((task_root / ".git").is_dir())
            self.assertTrue((task_root / "artifacts" / "progress_curve.svg").is_file())
            self.assertTrue((task_root / "artifacts" / "knowledge_base.md").is_file())
            self.assertTrue((task_root / "artifacts" / "experiments" / "aa-exp-0001.json").is_file())
            self.assertTrue((task_root / "artifacts" / "experiments" / "aa-exp-0001.md").is_file())
            self.assertEqual(history[0]["accepted"], True)
            self.assertEqual(history[1]["accepted"], True)
            self.assertEqual(history[1]["experiment_id"], "aa-exp-0001")
            self.assertIn("Refined the extraction logic.", context)
            self.assertIn("Best experiment: aa-exp-0001", context)

    def test_records_structured_experiment_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir)
            (task_root / "candidate.py").write_text("print('candidate')\n", encoding="utf-8")
            charter = TaskCharter(
                charter_id="demo",
                title="Demo",
                objective_statement="Improve quality while preserving schema validity.",
                data_assets=(),
                hard_constraints=(),
                soft_constraints=(),
                anti_goals=(),
                evaluation_plan=EvaluationPlan(
                    signals=(
                        ObjectiveSignal(
                            name="quality",
                            kind=SignalKind.SCALAR,
                            direction=SignalDirection.MAXIMIZE,
                            description="Quality",
                            weight=1.0,
                        ),
                        ObjectiveSignal(
                            name="schema_valid",
                            kind=SignalKind.BINARY,
                            direction=SignalDirection.SATISFY,
                            description="Schema validity",
                            hard_gate=True,
                            target_value=1.0,
                        ),
                    )
                ),
                search_surface=SearchSurface(),
                workspace_layout=WorkspaceLayout(),
            )
            experiment = ExperimentRecord(
                candidate_id="candidate-v2",
                baseline_candidate_id="baseline",
                decision=AcceptanceDecision(
                    accepted=True,
                    utility_gain=0.12,
                    reasons=("utility_improved",),
                ),
                baseline_report=EvaluationReport(
                    candidate_id="baseline",
                    signals=(
                        SignalResult(name="quality", value=0.70),
                        SignalResult(name="schema_valid", value=1.0),
                    ),
                ),
                candidate_report=EvaluationReport(
                    candidate_id="candidate-v2",
                    signals=(
                        SignalResult(name="quality", value=0.82),
                        SignalResult(name="schema_valid", value=1.0),
                    ),
                ),
                iteration_steps=(
                    IterationStep(
                        summary="Refined parser and validation flow.",
                        touched_paths=("candidate.py",),
                        executed_commands=("python3 eval.py",),
                    ),
                ),
                notes=("Observed better stability on known samples.",),
            )

            entry = record_experiment_record(
                task_root=task_root,
                charter=charter,
                experiment=experiment,
                hypothesis="Better parser routing will improve quality.",
                change_summary="Adjusted parser routing and validation ordering.",
                knowledge_items=("Parser routing was the limiting factor.",),
            )

            self.assertEqual(entry["accepted"], True)
            self.assertEqual(entry["decision_reasons"], ["utility_improved"])
            self.assertEqual(entry["touched_paths"], ["candidate.py"])
            self.assertEqual(entry["executed_commands"], ["python3 eval.py"])
            self.assertTrue((task_root / "artifacts" / "experiments" / "aa-exp-0000.md").is_file())


if __name__ == "__main__":
    unittest.main()
