from __future__ import annotations

import json
import os
import shutil
import textwrap
from dataclasses import asdict
from pathlib import Path

from .compiler import DefaultObjectiveCompiler
from .execution import run_task_command
from .history import choose_primary_signal_from_charter, record_experiment_result
from .models import (
    AgentLoopSpec,
    AgentRole,
    AgentRuntimeConfig,
    Constraint,
    ConstraintLevel,
    DataAsset,
    ExecutionBackendConfig,
    ExecutionBackendKind,
    ObjectiveBrief,
    ObjectiveSignal,
    RolePass,
    RunCommand,
    SignalDirection,
    SignalKind,
    SkillContribution,
    SubsystemSpec,
    WorkspaceLayout,
)
from .scaffold import materialize_scaffold
from .serialization import load_task_charter
from .workspace import resolve_workspace_paths


DEFAULT_MODEL = "x-ai/grok-4.1-fast"
LIBRARY_SRC = Path(__file__).resolve().parents[1]
INVOICE_SCAFFOLD_DIR = Path(__file__).resolve().parent / "scaffolds" / "invoice_workspace"


class InvoiceExtractionSkill:
    skill_id = "invoice-extraction"
    description = "Defaults for building and stress-testing invoice extraction pipelines."

    def contribute(self, brief: ObjectiveBrief) -> SkillContribution:
        del brief
        return SkillContribution(
            skill_id=self.skill_id,
            summary="Invoice extraction starter pack.",
            suggested_signals=(
                ObjectiveSignal(
                    name="field_accuracy",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MAXIMIZE,
                    description="Correctly extracted invoice fields across the eval corpus.",
                    weight=1.0,
                    source="skill",
                ),
                ObjectiveSignal(
                    name="document_pass_rate",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MAXIMIZE,
                    description="Fraction of invoices with all required fields extracted correctly.",
                    weight=0.75,
                    source="skill",
                ),
                ObjectiveSignal(
                    name="schema_valid",
                    kind=SignalKind.BINARY,
                    direction=SignalDirection.SATISFY,
                    description="Candidate output must be schema-valid.",
                    hard_gate=True,
                    target_value=1.0,
                    source="skill",
                ),
                ObjectiveSignal(
                    name="latency_seconds",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MINIMIZE,
                    description="Average end-to-end per-document latency.",
                    weight=0.15,
                    max_regression=2.0,
                    source="skill",
                ),
                ObjectiveSignal(
                    name="token_cost",
                    kind=SignalKind.SCALAR,
                    direction=SignalDirection.MINIMIZE,
                    description="Average token spend per document.",
                    weight=0.10,
                    max_regression=500.0,
                    source="skill",
                ),
            ),
            suggested_mutable_paths=("src/invoice_pipeline",),
            suggested_protected_paths=("eval", "fixtures", "goldens"),
            suggested_constraints=(
                Constraint(
                    statement="Keep the invoice pipeline modular, with explicit subsystem boundaries instead of one growing extractor file.",
                    level=ConstraintLevel.SOFT,
                    rationale="maintainability",
                ),
            ),
            suggested_subsystems=(
                SubsystemSpec(
                    subsystem_id="document-ingestion",
                    summary="Turn invoice PDFs into machine-usable page text or image inputs.",
                    owned_paths=("src/invoice_pipeline/document_io.py",),
                    primary_signals=("field_accuracy", "document_pass_rate"),
                    guardrail_signals=("latency_seconds",),
                    decomposition_hints=("pdf-to-text", "ocr-fallback", "page-ordering"),
                ),
                SubsystemSpec(
                    subsystem_id="field-extraction",
                    summary="Extract invoice fields from the document representation.",
                    owned_paths=(
                        "src/invoice_pipeline/field_extractors.py",
                        "src/invoice_pipeline/openrouter_client.py",
                        "src/invoice_pipeline/extract.py",
                    ),
                    primary_signals=("field_accuracy", "document_pass_rate"),
                    guardrail_signals=("token_cost", "latency_seconds"),
                    decomposition_hints=("vendor-header-detection", "date-parsing", "amount-parsing"),
                ),
                SubsystemSpec(
                    subsystem_id="normalization",
                    summary="Normalize raw extracted values into stable invoice schema fields.",
                    owned_paths=("src/invoice_pipeline/normalization.py",),
                    primary_signals=("field_accuracy", "schema_valid"),
                    guardrail_signals=("document_pass_rate",),
                    decomposition_hints=("currency-normalization", "string-cleanup", "date-standardization"),
                ),
                SubsystemSpec(
                    subsystem_id="schema-validation",
                    summary="Keep output shape stable and make low-confidence failures explicit.",
                    owned_paths=("src/invoice_pipeline/schema.py",),
                    primary_signals=("schema_valid", "document_pass_rate"),
                    guardrail_signals=("field_accuracy",),
                    decomposition_hints=("required-fields", "empty-value-policy", "confidence-notes"),
                ),
            ),
            decomposition_hints=(
                "ocr-cleanup",
                "vendor-normalization",
                "line-items",
                "schema-validation",
                "confidence-calibration",
            ),
            evaluation_notes=(
                "Use the builder role to improve extraction quality.",
                "Use the critic role to probe template overfitting, OCR brittleness, and missing-field handling.",
                "Use the critic role to flag architectural sprawl, hidden coupling, and avoidable monolith growth.",
                "Use the judge role to weigh holistic improvement against latency and token cost.",
            ),
        )


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_invoice_objective_brief(
    *,
    data_dir: Path,
    objective: str,
    model: str = DEFAULT_MODEL,
    focus_subsystems: tuple[str, ...] = (),
) -> ObjectiveBrief:
    return ObjectiveBrief(
        title="Invoice extraction",
        objective_statement=objective,
        data_assets=(
            DataAsset(
                name="invoice_corpus",
                kind="invoice_corpus",
                location=str(data_dir.expanduser().resolve()),
                role="train_eval_corpus",
            ),
        ),
        constraints=(
            "Only use models available through OpenRouter.",
            "Keep the extraction pipeline scalable and operationally simple.",
            "Keep the pipeline modular so subsystems can be improved independently.",
        ),
        anti_goals=(
            "Do not game the eval set through invoice-template memorization.",
            "Do not hide low-confidence cases behind silent fallback behavior.",
        ),
        focus_subsystems=focus_subsystems,
        allowed_models=(model,),
        run_commands=(
            RunCommand(
                name="evaluate",
                command=("python3", "eval/run_invoice_eval.py"),
                working_dir=".",
                timeout_seconds=600,
            ),
        ),
        workspace_layout=WorkspaceLayout(
            root_dir=".",
            candidate_dir="src/invoice_pipeline",
            evaluator_dir="eval",
            artifacts_dir="artifacts",
            replay_dir="replay",
        ),
        execution_backend=ExecutionBackendConfig(
            kind=ExecutionBackendKind.ISOLATED_WORKSPACE,
            sync_back_paths=("artifacts",),
            notes=(
                "Run candidate and eval commands inside a copied workspace, then sync artifacts back.",
            ),
        ),
        agent_runtime=AgentRuntimeConfig(
            provider="openrouter",
            api_key_env="OPENROUTER_API_KEY",
            default_model=model,
            allowed_models=(model,),
            notes=("The same CLI agent should build, critique, and judge the candidate in separate passes.",),
        ),
        agent_loop=AgentLoopSpec(
            passes=(
                RolePass(role=AgentRole.BUILDER, goal="Build or improve the invoice extraction pipeline."),
                RolePass(
                    role=AgentRole.CRITIC,
                    goal="Attack the pipeline for benchmark gaming, OCR brittleness, schema edge cases, and template overfitting.",
                ),
                RolePass(role=AgentRole.JUDGE, goal="Decide whether the candidate improved holistically."),
            )
        ),
        notes=("This task was bootstrapped from plain-English objective text plus a PDF directory.",),
    )


def compile_invoice_charter(
    *,
    data_dir: Path,
    objective: str,
    model: str = DEFAULT_MODEL,
    focus_subsystems: tuple[str, ...] = (),
):
    brief = build_invoice_objective_brief(
        data_dir=data_dir,
        objective=objective,
        model=model,
        focus_subsystems=focus_subsystems,
    )
    return DefaultObjectiveCompiler().compile(brief, skills=(InvoiceExtractionSkill(),))


def _task_readme_source(task_name: str) -> str:
    return textwrap.dedent(
        f"""
        # {task_name}

        This workspace was bootstrapped by `auto-anything`.

        The intended loop is:

        1. inspect `task_charter.json` for subsystem ids and owned paths
        2. improve one subsystem at a time inside `src/invoice_pipeline/`
        3. keep `eval/`, `fixtures/`, and `goldens/` stable
        4. run `python3 eval/run_invoice_eval.py`
        5. inspect `artifacts/eval_summary.json`, `artifacts/experiment_history.json`, `artifacts/knowledge_base.md`, and `artifacts/progress_curve.svg`
        6. inspect `artifacts/experiments/` for per-experiment markdown and JSON reports
        7. use local git history to diff current work against previous experiments
        8. switch into a critic stance and look for brittle assumptions or gaming
        9. keep iterating

        Starter subsystem ownership:

        - `document-ingestion`: `src/invoice_pipeline/document_io.py`
        - `field-extraction`: `src/invoice_pipeline/field_extractors.py`, `src/invoice_pipeline/openrouter_client.py`, `src/invoice_pipeline/extract.py`
        - `normalization`: `src/invoice_pipeline/normalization.py`
        - `schema-validation`: `src/invoice_pipeline/schema.py`

        This split exists so the same agent can focus work on one subsystem without losing
        whole-pipeline eval guardrails.

        Each eval run also snapshots the workspace into local git, writes a per-experiment report,
        and updates a rolling knowledge base so future experiments can inspect, diff, and refine
        earlier attempts instead of working blind.
        """
    ).strip() + "\n"


def _copy_dataset(data_dir: Path, fixtures_dir: Path, goldens_dir: Path) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    goldens_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        shutil.copy2(pdf_path, fixtures_dir / pdf_path.name)
        stem = pdf_path.stem
        for suffix in (".expected.json", ".json"):
            gold_path = data_dir / f"{stem}{suffix}"
            if gold_path.is_file():
                shutil.copy2(gold_path, goldens_dir / f"{stem}.expected.json")
                break
        else:
            (goldens_dir / f"{stem}.expected.json").write_text("{}\n", encoding="utf-8")


def bootstrap_invoice_task(
    *,
    task_root: Path,
    data_dir: Path,
    objective: str,
    model: str = DEFAULT_MODEL,
    focus_subsystems: tuple[str, ...] = (),
) -> Path:
    charter = compile_invoice_charter(
        data_dir=data_dir,
        objective=objective,
        model=model,
        focus_subsystems=focus_subsystems,
    )
    task_root = task_root.expanduser().resolve()
    paths = resolve_workspace_paths(task_root, charter.workspace_layout)
    paths.ensure_exists()
    (task_root / "fixtures").mkdir(parents=True, exist_ok=True)
    (task_root / "goldens").mkdir(parents=True, exist_ok=True)

    replacements = {
        "__AUTO_ANYTHING_LIBRARY_SRC__": str(LIBRARY_SRC),
        "__AUTO_ANYTHING_DEFAULT_MODEL__": model,
    }
    materialize_scaffold(
        scaffold_dir=INVOICE_SCAFFOLD_DIR,
        destination_dir=task_root,
        replacements=replacements,
    )
    _copy_dataset(data_dir, task_root / "fixtures", task_root / "goldens")
    (task_root / "README.md").write_text(_task_readme_source(task_root.name), encoding="utf-8")
    (task_root / "task_charter.json").write_text(
        json.dumps(asdict(charter), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return task_root


def run_bootstrapped_eval(task_root: Path):
    task_root = task_root.expanduser().resolve()
    charter = load_task_charter(task_root / "task_charter.json")
    result = run_task_command(
        task_root=task_root,
        charter=charter,
        command_name="evaluate",
        extra_env={"AUTO_ANYTHING_SKIP_AUTO_RECORD": "1"},
    )
    result.check_returncode()
    summary = json.loads((task_root / "artifacts" / "eval_summary.json").read_text(encoding="utf-8"))
    metric_name, direction = choose_primary_signal_from_charter(charter)
    record_experiment_result(
        task_root=task_root,
        summary=summary,
        metric_name=metric_name,
        direction=direction,
        label=summary.get("candidate_id", "baseline"),
        focus_subsystems=charter.focus_subsystems,
        executed_commands=(
            " ".join(result.command),
            f"backend={result.backend_kind.value}",
        ),
        notes=tuple(result.notes) + tuple(f"sync_back={path}" for path in result.synced_paths),
    )
    return result
