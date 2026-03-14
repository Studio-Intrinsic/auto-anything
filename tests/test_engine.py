from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.engine import ExperimentEngine
from auto_anything.models import (
    CounterbalanceConfig,
    CounterbalanceMode,
    CounterbalanceReport,
    CritiqueFinding,
    CritiqueSeverity,
    EvaluationPlan,
    EvaluationReport,
    ObjectiveSignal,
    RunCommand,
    SearchSurface,
    SignalDirection,
    SignalKind,
    SignalResult,
    SubsystemSpec,
    TaskCharter,
    WorkspaceLayout,
)


def _charter(*, with_counterbalance: bool = False) -> TaskCharter:
    return TaskCharter(
        charter_id="demo",
        title="Demo",
        objective_statement="Improve quality without breaking constraints.",
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
                    description="Overall quality",
                    weight=1.0,
                ),
                ObjectiveSignal(
                    name="cost",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MINIMIZE,
                    description="Run cost",
                    weight=0.25,
                    max_regression=2.0,
                ),
                ObjectiveSignal(
                    name="compliance",
                    kind=SignalKind.BINARY,
                    direction=SignalDirection.SATISFY,
                    description="Must pass validation",
                    hard_gate=True,
                    target_value=1.0,
                ),
            )
        ),
        search_surface=SearchSurface(
            subsystems=(
                SubsystemSpec(
                    subsystem_id="quality-core",
                    summary="Improve core extraction quality while watching run cost.",
                    owned_paths=("src/invoice_pipeline",),
                    primary_signals=("quality",),
                    guardrail_signals=("cost",),
                ),
            )
        ),
        workspace_layout=WorkspaceLayout(),
        counterbalance=(
            CounterbalanceConfig()
            if with_counterbalance
            else CounterbalanceConfig(mode=CounterbalanceMode.NONE, required=False)
        ),
        run_commands=(
            RunCommand(name="eval", command=("python", "eval.py")),
        ),
    )


class EngineTests(unittest.TestCase):
    def test_accepts_candidate_that_improves_weighted_utility(self) -> None:
        engine = ExperimentEngine()
        baseline = EvaluationReport(
            candidate_id="baseline",
            signals=(
                SignalResult(name="quality", value=0.70),
                SignalResult(name="cost", value=10.0),
                SignalResult(name="compliance", value=1.0),
            ),
        )
        candidate = EvaluationReport(
            candidate_id="candidate",
            signals=(
                SignalResult(name="quality", value=0.82),
                SignalResult(name="cost", value=10.2),
                SignalResult(name="compliance", value=1.0),
            ),
        )

        decision = engine.decide(charter=_charter(), baseline_report=baseline, candidate_report=candidate)

        self.assertTrue(decision.accepted)
        self.assertGreater(decision.utility_gain, 0.0)

    def test_rejects_candidate_that_fails_hard_gate(self) -> None:
        engine = ExperimentEngine()
        baseline = EvaluationReport(
            candidate_id="baseline",
            signals=(
                SignalResult(name="quality", value=0.70),
                SignalResult(name="cost", value=10.0),
                SignalResult(name="compliance", value=1.0),
            ),
        )
        candidate = EvaluationReport(
            candidate_id="candidate",
            signals=(
                SignalResult(name="quality", value=0.95),
                SignalResult(name="cost", value=8.0),
                SignalResult(name="compliance", value=0.0),
            ),
        )

        decision = engine.decide(charter=_charter(), baseline_report=baseline, candidate_report=candidate)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.blocking_signals, ("compliance",))

    def test_rejects_candidate_when_self_critic_finds_blocking_issue(self) -> None:
        engine = ExperimentEngine()
        baseline = EvaluationReport(
            candidate_id="baseline",
            signals=(
                SignalResult(name="quality", value=0.70),
                SignalResult(name="cost", value=10.0),
                SignalResult(name="compliance", value=1.0),
            ),
        )
        candidate = EvaluationReport(
            candidate_id="candidate",
            signals=(
                SignalResult(name="quality", value=0.90),
                SignalResult(name="cost", value=9.0),
                SignalResult(name="compliance", value=1.0),
            ),
        )
        critique = CounterbalanceReport(
            mode="self_critic",
            findings=(
                CritiqueFinding(
                    finding_id="gaming-risk",
                    summary="Pipeline appears to overfit the known invoice template.",
                    severity=CritiqueSeverity.HIGH,
                ),
            ),
        )

        decision = engine.decide(
            charter=_charter(with_counterbalance=True),
            baseline_report=baseline,
            candidate_report=candidate,
            counterbalance_report=critique,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.blocking_findings, ("gaming-risk",))

    def test_focuses_utility_on_selected_subsystem_signals(self) -> None:
        engine = ExperimentEngine()
        baseline = EvaluationReport(
            candidate_id="baseline",
            signals=(
                SignalResult(name="quality", value=0.70),
                SignalResult(name="cost", value=10.0),
                SignalResult(name="compliance", value=1.0),
            ),
        )
        candidate = EvaluationReport(
            candidate_id="candidate",
            signals=(
                SignalResult(name="quality", value=0.80),
                SignalResult(name="cost", value=10.2),
                SignalResult(name="compliance", value=1.0),
            ),
        )

        decision = engine.decide(
            charter=_charter(),
            baseline_report=baseline,
            candidate_report=candidate,
            focus_subsystems=("quality-core",),
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.focus_subsystems, ("quality-core",))

    def test_rejects_unknown_focus_subsystem(self) -> None:
        engine = ExperimentEngine()
        baseline = EvaluationReport(
            candidate_id="baseline",
            signals=(
                SignalResult(name="quality", value=0.70),
                SignalResult(name="cost", value=10.0),
                SignalResult(name="compliance", value=1.0),
            ),
        )
        candidate = EvaluationReport(
            candidate_id="candidate",
            signals=(
                SignalResult(name="quality", value=0.80),
                SignalResult(name="cost", value=10.2),
                SignalResult(name="compliance", value=1.0),
            ),
        )

        with self.assertRaises(ValueError):
            engine.decide(
                charter=_charter(),
                baseline_report=baseline,
                candidate_report=candidate,
                focus_subsystems=("unknown-subsystem",),
            )


if __name__ == "__main__":
    unittest.main()
