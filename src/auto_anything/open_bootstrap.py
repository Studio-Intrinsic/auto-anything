from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import textwrap

from .compiler import DefaultObjectiveCompiler
from .invoice_bootstrap import DEFAULT_MODEL
from .models import (
    AgentLoopSpec,
    AgentRole,
    AgentRuntimeConfig,
    ExecutionBackendConfig,
    ExecutionBackendKind,
    ObjectiveBrief,
    ObjectiveSignal,
    RolePass,
    RunCommand,
    SignalDirection,
    SignalKind,
    SubsystemSpec,
    WorkspaceLayout,
)
from .scaffold import materialize_scaffold
from .task_docs import render_task_agents_md
from .workspace import resolve_workspace_paths


OPEN_TASK_SCAFFOLD_DIR = Path(__file__).resolve().parent / "scaffolds" / "open_task_workspace"
LIBRARY_SRC = Path(__file__).resolve().parents[1]


def _normalize_words(value: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if token}


def _derived_signals_for_objective(objective: str) -> tuple[ObjectiveSignal, ...]:
    terms = _normalize_words(objective)
    signals: list[ObjectiveSignal] = []
    if terms & {"accuracy", "accurate", "accurately", "quality", "correct", "correctness", "f1", "precision", "recall"}:
        signals.append(
            ObjectiveSignal(
                name="task_quality",
                kind=SignalKind.RUBRIC,
                direction=SignalDirection.MAXIMIZE,
                description="Overall task quality against the user's objective.",
                weight=1.0,
                source="derived",
            )
        )
    if terms & {"token", "tokens", "cost", "cheap", "budget"}:
        signals.append(
            ObjectiveSignal(
                name="token_cost",
                kind=SignalKind.SCALAR,
                direction=SignalDirection.MINIMIZE,
                description="Average token usage or equivalent model cost per task run.",
                weight=0.2,
                source="derived",
            )
        )
    if terms & {"latency", "time", "fast", "runtime", "speed"}:
        signals.append(
            ObjectiveSignal(
                name="latency_seconds",
                kind=SignalKind.SCALAR,
                direction=SignalDirection.MINIMIZE,
                description="Average end-to-end runtime in seconds.",
                weight=0.2,
                source="derived",
            )
        )
    return tuple(signals)


def _default_subsystems(focus_subsystems: tuple[str, ...]) -> tuple[SubsystemSpec, ...]:
    defaults = [
        SubsystemSpec(
            subsystem_id="data-ingestion",
            summary="Load and normalize referenced input data into a workable internal representation.",
            owned_paths=("src/task_pipeline/data_ingestion.py",),
            decomposition_hints=("format-detection", "dataset-slicing", "input-normalization"),
        ),
        SubsystemSpec(
            subsystem_id="pipeline-logic",
            summary="Core candidate logic for solving the task.",
            owned_paths=("src/task_pipeline/pipeline.py", "src/task_pipeline/postprocess.py"),
            decomposition_hints=("candidate-architecture", "model-calls", "postprocessing"),
        ),
        SubsystemSpec(
            subsystem_id="evaluation-harness",
            summary="Evaluator and metrics harness for the current task.",
            owned_paths=("eval/run_task_eval.py",),
            decomposition_hints=("metric-definition", "golden-loading", "failure-analysis"),
        ),
    ]
    seen = {spec.subsystem_id for spec in defaults}
    for subsystem_id in focus_subsystems:
        if subsystem_id in seen:
            continue
        defaults.append(
            SubsystemSpec(
                subsystem_id=subsystem_id,
                summary=f"User-requested focus subsystem: {subsystem_id}.",
                owned_paths=("src/task_pipeline",),
            )
        )
    return tuple(defaults)


def build_open_objective_brief(
    *,
    title: str,
    objective: str,
    data_assets: tuple,
    anti_goals: tuple[str, ...] = (),
    constraints: tuple[str, ...] = (),
    allowed_models: tuple[str, ...] = (),
    focus_subsystems: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> ObjectiveBrief:
    model = allowed_models[0] if allowed_models else DEFAULT_MODEL
    return ObjectiveBrief(
        title=title,
        objective_statement=objective,
        data_assets=data_assets,
        explicit_signals=_derived_signals_for_objective(objective),
        constraints=constraints
        + (
            "Keep the candidate pipeline modular and easy to iterate on.",
            "Make the evaluation harness explicit and inspectable.",
        ),
        anti_goals=anti_goals,
        mutable_paths=("src/task_pipeline", "eval"),
        protected_paths=("artifacts",),
        entrypoints=("src/task_pipeline/pipeline.py", "eval/run_task_eval.py"),
        subsystems=_default_subsystems(focus_subsystems),
        focus_subsystems=focus_subsystems,
        allowed_models=allowed_models or (model,),
        run_commands=(
            RunCommand(
                name="evaluate",
                command=("python3", "eval/run_task_eval.py"),
                working_dir=".",
                timeout_seconds=600,
            ),
        ),
        workspace_layout=WorkspaceLayout(
            root_dir=".",
            candidate_dir="src/task_pipeline",
            evaluator_dir="eval",
            artifacts_dir="artifacts",
            replay_dir="replay",
        ),
        execution_backend=ExecutionBackendConfig(
            kind=ExecutionBackendKind.ISOLATED_WORKSPACE,
            sync_back_paths=("artifacts",),
            notes=("Run the candidate and eval harness inside a copied workspace, then sync artifacts back.",),
        ),
        agent_runtime=AgentRuntimeConfig(
            provider="openrouter",
            api_key_env="OPENROUTER_API_KEY",
            default_model=model,
            allowed_models=allowed_models or (model,),
            notes=("The same CLI agent should construct, critique, and judge the task world.",),
        ),
        agent_loop=AgentLoopSpec(
            passes=(
                RolePass(role=AgentRole.BUILDER, goal="Build or improve the task pipeline and evaluator."),
                RolePass(
                    role=AgentRole.CRITIC,
                    goal="Attack the current setup for gaming, weak evaluation, brittle assumptions, or architecture drift.",
                ),
                RolePass(role=AgentRole.JUDGE, goal="Decide whether the task world improved holistically."),
            )
        ),
        notes=notes
        + (
            "This workspace was synthesized from an open-ended plain-text request.",
            "The initial evaluator is a placeholder until the agent implements task-specific metrics.",
        ),
    )


def compile_open_task_charter(
    *,
    title: str,
    objective: str,
    data_assets: tuple,
    anti_goals: tuple[str, ...] = (),
    constraints: tuple[str, ...] = (),
    allowed_models: tuple[str, ...] = (),
    focus_subsystems: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
):
    brief = build_open_objective_brief(
        title=title,
        objective=objective,
        data_assets=data_assets,
        anti_goals=anti_goals,
        constraints=constraints,
        allowed_models=allowed_models,
        focus_subsystems=focus_subsystems,
        notes=notes,
    )
    return DefaultObjectiveCompiler().compile(brief)


def _task_readme_source(*, task_name: str, objective: str, data_assets: tuple) -> str:
    asset_lines = "\n".join(f"- `{asset.name}`: `{asset.location}`" for asset in data_assets) or "- none"
    return textwrap.dedent(
        f"""
        # {task_name}

        This workspace was synthesized by `auto-anything` from an open-ended request.

        Objective:

        {objective}

        Referenced data:

        {asset_lines}

        The initial scaffold is intentionally generic. The first real task is usually:

        1. inspect the data and decide what the candidate pipeline should do
        2. replace the placeholder evaluator in `eval/run_task_eval.py` with a real task-specific harness
        3. shape `src/task_pipeline/` into a real candidate implementation
        4. run the baseline eval
        5. iterate using the experiment history and critic loop
        """
    ).strip() + "\n"


def bootstrap_open_task(
    *,
    task_root: Path,
    title: str,
    objective: str,
    data_assets: tuple,
    anti_goals: tuple[str, ...] = (),
    constraints: tuple[str, ...] = (),
    allowed_models: tuple[str, ...] = (),
    focus_subsystems: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> Path:
    charter = compile_open_task_charter(
        title=title,
        objective=objective,
        data_assets=data_assets,
        anti_goals=anti_goals,
        constraints=constraints,
        allowed_models=allowed_models,
        focus_subsystems=focus_subsystems,
        notes=notes,
    )
    task_root = task_root.expanduser().resolve()
    paths = resolve_workspace_paths(task_root, charter.workspace_layout)
    paths.ensure_exists()
    replacements = {
        "__AUTO_ANYTHING_LIBRARY_SRC__": str(LIBRARY_SRC),
    }
    materialize_scaffold(
        scaffold_dir=OPEN_TASK_SCAFFOLD_DIR,
        destination_dir=task_root,
        replacements=replacements,
    )
    iteration_script = (LIBRARY_SRC.parent / "examples" / "run_task_iteration.py").resolve()
    iteration_command = f"python3 {iteration_script} --task-root {task_root} --hypothesis ... --change-summary ..."
    (task_root / "README.md").write_text(
        _task_readme_source(task_name=task_root.name, objective=objective, data_assets=data_assets),
        encoding="utf-8",
    )
    agents_content = render_task_agents_md(
        charter=charter,
        task_name=task_root.name,
        iteration_command=iteration_command,
    )
    (task_root / "AGENTS.md").write_text(agents_content, encoding="utf-8")
    (task_root / "CLAUDE.md").write_text(agents_content, encoding="utf-8")
    (task_root / "task_charter.json").write_text(
        json.dumps(asdict(charter), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return task_root


__all__ = [
    "build_open_objective_brief",
    "compile_open_task_charter",
    "bootstrap_open_task",
]
