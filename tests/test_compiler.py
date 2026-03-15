from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.compiler import DefaultObjectiveCompiler
from auto_anything.models import (
    AgentLoopSpec,
    AgentRole,
    AgentRuntimeConfig,
    Constraint,
    ConstraintLevel,
    DataAsset,
    ExecutionBackendConfig,
    ExecutionBackendKind,
    OptimizationMode,
    ObjectiveBrief,
    ObjectiveSignal,
    OptimizableArtifactKind,
    RolePass,
    RunCommand,
    SearchStrategyKind,
    SearchStrategySpec,
    SignalDirection,
    SignalKind,
    SkillContribution,
    SubsystemSpec,
    WorkspaceLayout,
)


class PdfSkill:
    skill_id = "invoice-extraction"
    description = "Invoice extraction defaults"

    def contribute(self, brief: ObjectiveBrief) -> SkillContribution:
        del brief
        return SkillContribution(
            skill_id=self.skill_id,
            suggested_signals=(
                ObjectiveSignal(
                    name="field_accuracy",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MAXIMIZE,
                    description="Correctly extracted invoice fields",
                    weight=1.0,
                    source="skill",
                ),
                ObjectiveSignal(
                    name="schema_valid",
                    kind=SignalKind.BINARY,
                    direction=SignalDirection.SATISFY,
                    description="Must emit schema-valid extraction output",
                    hard_gate=True,
                    target_value=1.0,
                    source="skill",
                ),
            ),
            suggested_constraints=(
                Constraint(
                    statement="Use only approved LLM models.",
                    level=ConstraintLevel.HARD,
                    rationale="skill_default",
                ),
            ),
            suggested_mutable_paths=("src/invoice_pipeline",),
            suggested_protected_paths=("eval",),
            suggested_subsystems=(
                SubsystemSpec(
                    subsystem_id="field-extraction",
                    summary="Extract invoice fields from invoice text.",
                    owned_paths=("src/invoice_pipeline/extract.py",),
                    primary_signals=("field_accuracy",),
                    guardrail_signals=("schema_valid",),
                ),
            ),
            decomposition_hints=("ocr-cleanup", "line-items", "vendor-normalization", "confidence-calibration"),
            evaluation_notes=("Prefer deterministic validators for hard gates.",),
        )


class CompilerTests(unittest.TestCase):
    def test_compiles_plain_language_brief_with_skill_contributions(self) -> None:
        compiler = DefaultObjectiveCompiler()
        brief = ObjectiveBrief(
            title="Invoice extraction",
            objective_statement="Extract invoice data accurately over the supplied invoice corpus.",
            data_assets=(
                DataAsset(
                    name="invoice_corpus",
                    kind="invoice_corpus",
                    location="/tmp/invoices",
                    role="training_and_eval",
                ),
            ),
            anti_goals=("Do not use unapproved models.",),
            constraints=("Keep runtime scalable.",),
            allowed_models=("x-ai/grok-4.1-fast",),
            focus_subsystems=("field-extraction",),
            run_commands=(
                RunCommand(name="eval", command=("python", "scripts/eval.py")),
            ),
            workspace_layout=WorkspaceLayout(candidate_dir="pipeline"),
            execution_backend=ExecutionBackendConfig(
                kind=ExecutionBackendKind.ISOLATED_WORKSPACE,
                sync_back_paths=("artifacts", "replay"),
            ),
            optimization_mode=OptimizationMode.MULTI_TASK,
            search_strategy=SearchStrategySpec(kind=SearchStrategyKind.PROBE_AND_COMMIT, beam_width=3),
            agent_runtime=AgentRuntimeConfig(
                provider="openrouter",
                api_key_env="OPENROUTER_API_KEY",
                default_model="x-ai/grok-4.1-fast",
            ),
            agent_loop=AgentLoopSpec(
                passes=(
                    RolePass(role=AgentRole.BUILDER, goal="Build the invoice extraction pipeline."),
                    RolePass(role=AgentRole.CRITIC, goal="Attack the pipeline for gaming and brittle assumptions."),
                    RolePass(role=AgentRole.JUDGE, goal="Decide whether the pipeline improved holistically."),
                )
            ),
        )

        charter = compiler.compile(brief, skills=(PdfSkill(),))

        self.assertEqual(charter.charter_id, "invoice-extraction")
        self.assertEqual(charter.applied_skills, ("invoice-extraction",))
        self.assertEqual(
            charter.search_surface.mutable_paths,
            ("src/invoice_pipeline", "src/invoice_pipeline/extract.py"),
        )
        self.assertEqual(charter.workspace_layout.candidate_dir, "pipeline")
        self.assertEqual(charter.execution_backend.kind, ExecutionBackendKind.ISOLATED_WORKSPACE)
        self.assertEqual(charter.execution_backend.sync_back_paths, ("artifacts", "replay"))
        self.assertEqual(charter.optimization_mode, OptimizationMode.MULTI_TASK)
        self.assertEqual(charter.search_strategy.kind, SearchStrategyKind.PROBE_AND_COMMIT)
        self.assertEqual(charter.search_strategy.beam_width, 3)
        self.assertEqual(charter.run_commands[0].name, "eval")
        self.assertEqual(charter.agent_runtime.provider, "openrouter")
        self.assertEqual(charter.agent_runtime.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(tuple(role.role.value for role in charter.agent_loop.passes), ("builder", "critic", "judge"))
        self.assertIn("line-items", charter.decomposition_hints)
        self.assertEqual(charter.focus_subsystems, ("field-extraction",))
        self.assertEqual(charter.search_surface.subsystems[0].subsystem_id, "field-extraction")
        self.assertIn("src/invoice_pipeline/extract.py", charter.search_surface.mutable_paths)
        self.assertEqual(
            tuple(signal.name for signal in charter.evaluation_plan.signals),
            ("field_accuracy", "schema_valid"),
        )
        self.assertEqual(charter.optimizable_artifacts[0].artifact_id, "candidate-surface")
        self.assertEqual(charter.optimizable_artifacts[0].kind, OptimizableArtifactKind.WORKSPACE_SLICE)
        self.assertIn("src/invoice_pipeline/extract.py", charter.optimizable_artifacts[0].mutable_paths)
        self.assertEqual(len(charter.hard_constraints), 3)

    def test_derives_default_rubric_signal_when_no_metric_is_supplied(self) -> None:
        compiler = DefaultObjectiveCompiler()
        brief = ObjectiveBrief(
            title="General task",
            objective_statement="Be useful and accurate.",
        )

        charter = compiler.compile(brief)

        self.assertEqual(len(charter.evaluation_plan.signals), 1)
        self.assertEqual(charter.evaluation_plan.signals[0].name, "goal_alignment")
        self.assertEqual(charter.evaluation_plan.signals[0].kind, SignalKind.RUBRIC)


if __name__ == "__main__":
    unittest.main()
