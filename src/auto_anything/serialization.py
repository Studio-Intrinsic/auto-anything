from __future__ import annotations

import json
from pathlib import Path

from .models import (
    AgentLoopSpec,
    AgentRole,
    AgentRuntimeConfig,
    Constraint,
    ConstraintLevel,
    CounterbalanceConfig,
    CounterbalanceMode,
    CritiqueSeverity,
    DataAsset,
    ExecutionBackendConfig,
    ExecutionBackendKind,
    EvaluationPlan,
    EvaluatorDiagnostic,
    EvaluationReport,
    OptimizationMode,
    OptimizableArtifact,
    OptimizableArtifactKind,
    ObjectiveSignal,
    RolePass,
    RunCommand,
    SearchStrategyKind,
    SearchStrategySpec,
    SearchSurface,
    SignalDirection,
    SignalKind,
    SignalResult,
    SubsystemSpec,
    TaskCharter,
    WorkspaceLayout,
)


def _constraint_from_dict(payload: dict) -> Constraint:
    return Constraint(
        statement=payload["statement"],
        level=ConstraintLevel(payload.get("level", ConstraintLevel.HARD.value)),
        rationale=payload.get("rationale", ""),
    )


def _objective_signal_from_dict(payload: dict) -> ObjectiveSignal:
    return ObjectiveSignal(
        name=payload["name"],
        kind=SignalKind(payload["kind"]),
        direction=SignalDirection(payload["direction"]),
        description=payload["description"],
        weight=float(payload.get("weight", 1.0)),
        hard_gate=bool(payload.get("hard_gate", False)),
        target_value=payload.get("target_value"),
        max_regression=payload.get("max_regression"),
        source=payload.get("source", "user"),
    )


def _subsystem_from_dict(payload: dict) -> SubsystemSpec:
    return SubsystemSpec(
        subsystem_id=payload["subsystem_id"],
        summary=payload["summary"],
        owned_paths=tuple(payload.get("owned_paths", [])),
        primary_signals=tuple(payload.get("primary_signals", [])),
        guardrail_signals=tuple(payload.get("guardrail_signals", [])),
        decomposition_hints=tuple(payload.get("decomposition_hints", [])),
        notes=tuple(payload.get("notes", [])),
    )


def _run_command_from_dict(payload: dict) -> RunCommand:
    return RunCommand(
        name=payload["name"],
        command=tuple(payload["command"]),
        working_dir=payload.get("working_dir", "."),
        timeout_seconds=payload.get("timeout_seconds"),
        notes=tuple(payload.get("notes", [])),
    )


def _role_pass_from_dict(payload: dict) -> RolePass:
    return RolePass(
        role=AgentRole(payload["role"]),
        goal=payload["goal"],
        notes=tuple(payload.get("notes", [])),
    )


def _optimizable_artifact_from_dict(payload: dict) -> OptimizableArtifact:
    return OptimizableArtifact(
        artifact_id=payload["artifact_id"],
        kind=OptimizableArtifactKind(payload.get("kind", OptimizableArtifactKind.WORKSPACE_SLICE.value)),
        location=payload.get("location", ""),
        mutable_paths=tuple(payload.get("mutable_paths", [])),
        description=payload.get("description", ""),
        serialization_hint=payload.get("serialization_hint", ""),
        notes=tuple(payload.get("notes", [])),
    )


def task_charter_from_dict(payload: dict) -> TaskCharter:
    return TaskCharter(
        charter_id=payload["charter_id"],
        title=payload["title"],
        objective_statement=payload["objective_statement"],
        data_assets=tuple(
            DataAsset(
                name=item["name"],
                kind=item["kind"],
                location=item["location"],
                role=item.get("role", ""),
                notes=tuple(item.get("notes", [])),
            )
            for item in payload.get("data_assets", [])
        ),
        hard_constraints=tuple(_constraint_from_dict(item) for item in payload.get("hard_constraints", [])),
        soft_constraints=tuple(_constraint_from_dict(item) for item in payload.get("soft_constraints", [])),
        anti_goals=tuple(payload.get("anti_goals", [])),
        evaluation_plan=EvaluationPlan(
            signals=tuple(
                _objective_signal_from_dict(item)
                for item in payload["evaluation_plan"]["signals"]
            ),
            notes=tuple(payload["evaluation_plan"].get("notes", [])),
        ),
        optimizable_artifacts=tuple(
            _optimizable_artifact_from_dict(item)
            for item in payload.get("optimizable_artifacts", [])
        ),
        optimization_mode=OptimizationMode(payload.get("optimization_mode", OptimizationMode.SINGLE_TASK.value)),
        search_strategy=SearchStrategySpec(
            kind=SearchStrategyKind(
                payload.get("search_strategy", {}).get("kind", SearchStrategyKind.PROBE_AND_COMMIT.value)
            ),
            beam_width=int(payload.get("search_strategy", {}).get("beam_width", 1)),
            notes=tuple(payload.get("search_strategy", {}).get("notes", [])),
        ),
        search_surface=SearchSurface(
            mutable_paths=tuple(payload["search_surface"].get("mutable_paths", [])),
            protected_paths=tuple(payload["search_surface"].get("protected_paths", [])),
            entrypoints=tuple(payload["search_surface"].get("entrypoints", [])),
            subsystems=tuple(
                _subsystem_from_dict(item)
                for item in payload["search_surface"].get("subsystems", [])
            ),
            notes=tuple(payload["search_surface"].get("notes", [])),
        ),
        workspace_layout=WorkspaceLayout(
            root_dir=payload["workspace_layout"].get("root_dir", "."),
            candidate_dir=payload["workspace_layout"].get("candidate_dir", "candidate"),
            evaluator_dir=payload["workspace_layout"].get("evaluator_dir", "eval"),
            artifacts_dir=payload["workspace_layout"].get("artifacts_dir", "artifacts"),
            replay_dir=payload["workspace_layout"].get("replay_dir", "replay"),
            notes=tuple(payload["workspace_layout"].get("notes", [])),
        ),
        execution_backend=ExecutionBackendConfig(
            kind=ExecutionBackendKind(
                payload.get("execution_backend", {}).get("kind", ExecutionBackendKind.DIRECT_SUBPROCESS.value)
            ),
            env_allowlist=tuple(
                payload.get("execution_backend", {}).get(
                    "env_allowlist", ExecutionBackendConfig().env_allowlist
                )
            ),
            inherit_env=bool(payload.get("execution_backend", {}).get("inherit_env", True)),
            sync_back_paths=tuple(
                payload.get("execution_backend", {}).get(
                    "sync_back_paths", ExecutionBackendConfig().sync_back_paths
                )
            ),
            notes=tuple(payload.get("execution_backend", {}).get("notes", [])),
        ),
        focus_subsystems=tuple(payload.get("focus_subsystems", [])),
        agent_runtime=AgentRuntimeConfig(
            provider=payload.get("agent_runtime", {}).get("provider", "openrouter"),
            api_key_env=payload.get("agent_runtime", {}).get("api_key_env", "OPENROUTER_API_KEY"),
            default_model=payload.get("agent_runtime", {}).get("default_model", ""),
            allowed_models=tuple(payload.get("agent_runtime", {}).get("allowed_models", [])),
            notes=tuple(payload.get("agent_runtime", {}).get("notes", [])),
        ),
        agent_loop=AgentLoopSpec(
            passes=tuple(
                _role_pass_from_dict(item)
                for item in payload.get("agent_loop", {}).get("passes", [])
            ) or AgentLoopSpec().passes,
            notes=tuple(payload.get("agent_loop", {}).get("notes", [])),
        ),
        counterbalance=CounterbalanceConfig(
            mode=CounterbalanceMode(payload.get("counterbalance", {}).get("mode", CounterbalanceMode.SELF_CRITIC.value)),
            required=bool(payload.get("counterbalance", {}).get("required", True)),
            block_on_severity=CritiqueSeverity(
                payload.get("counterbalance", {}).get("block_on_severity", CritiqueSeverity.HIGH.value)
            ),
            penalty_per_finding=float(payload.get("counterbalance", {}).get("penalty_per_finding", 0.1)),
            notes=tuple(payload.get("counterbalance", {}).get("notes", [])),
        ),
        run_commands=tuple(_run_command_from_dict(item) for item in payload.get("run_commands", [])),
        applied_skills=tuple(payload.get("applied_skills", [])),
        decomposition_hints=tuple(payload.get("decomposition_hints", [])),
        allowed_models=tuple(payload.get("allowed_models", [])),
        disallowed_models=tuple(payload.get("disallowed_models", [])),
        allowed_tools=tuple(payload.get("allowed_tools", [])),
        notes=tuple(payload.get("notes", [])),
    )


def load_task_charter(path: Path) -> TaskCharter:
    return task_charter_from_dict(json.loads(path.read_text(encoding="utf-8")))


def evaluation_report_from_summary(summary: dict) -> EvaluationReport:
    return EvaluationReport(
        candidate_id=summary["candidate_id"],
        signals=tuple(
            SignalResult(name=name, value=float(value))
            for name, value in summary.get("signals", {}).items()
        ),
        diagnostics=tuple(
            EvaluatorDiagnostic(
                diagnostic_id=item.get("diagnostic_id", ""),
                summary=item["summary"],
                severity=CritiqueSeverity(item.get("severity", CritiqueSeverity.MEDIUM.value)),
                related_signals=tuple(item.get("related_signals", [])),
                notes=tuple(item.get("notes", [])),
            )
            for item in summary.get("diagnostics", [])
        ),
    )
