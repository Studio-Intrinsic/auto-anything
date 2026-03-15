from __future__ import annotations

import re

from .interfaces import ObjectiveCompiler, SkillPack
from .models import (
    Constraint,
    ConstraintLevel,
    EvaluationPlan,
    ObjectiveBrief,
    ObjectiveSignal,
    OptimizableArtifact,
    OptimizableArtifactKind,
    SearchSurface,
    SignalDirection,
    SignalKind,
    SkillContribution,
    SubsystemSpec,
    TaskCharter,
)


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "task"


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = value.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def _merge_constraints(
    *,
    brief: ObjectiveBrief,
    contributions: tuple[SkillContribution, ...],
) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    constraints.extend(
        Constraint(statement=item.strip(), level=ConstraintLevel.HARD, rationale="user_constraint")
        for item in brief.constraints
        if item.strip()
    )
    constraints.extend(
        Constraint(statement=item.strip(), level=ConstraintLevel.HARD, rationale="anti_goal")
        for item in brief.anti_goals
        if item.strip()
    )
    for contribution in contributions:
        constraints.extend(contribution.suggested_constraints)
    seen: set[tuple[str, ConstraintLevel]] = set()
    merged: list[Constraint] = []
    for constraint in constraints:
        key = (constraint.statement, constraint.level)
        if key in seen:
            continue
        seen.add(key)
        merged.append(constraint)
    return tuple(merged)


def _default_signal_for_brief(brief: ObjectiveBrief) -> ObjectiveSignal:
    return ObjectiveSignal(
        name="goal_alignment",
        kind=SignalKind.RUBRIC,
        direction=SignalDirection.MAXIMIZE,
        description=(
            "Model-judged alignment with the user objective: "
            f"{brief.objective_statement.strip()}"
        ),
        weight=1.0,
        source="derived",
    )


def _merge_signals(
    *,
    brief: ObjectiveBrief,
    contributions: tuple[SkillContribution, ...],
) -> tuple[ObjectiveSignal, ...]:
    signal_map: dict[str, ObjectiveSignal] = {}
    for signal in brief.explicit_signals:
        signal_map[signal.name] = signal
    for contribution in contributions:
        for signal in contribution.suggested_signals:
            signal_map.setdefault(signal.name, signal)
    if not signal_map:
        derived = _default_signal_for_brief(brief)
        signal_map[derived.name] = derived
    return tuple(signal_map.values())


def _merge_subsystems(
    *,
    brief: ObjectiveBrief,
    contributions: tuple[SkillContribution, ...],
) -> tuple[SubsystemSpec, ...]:
    merged: dict[str, SubsystemSpec] = {}
    ordered_specs = tuple(brief.subsystems) + tuple(
        subsystem
        for contribution in contributions
        for subsystem in contribution.suggested_subsystems
    )
    for subsystem in ordered_specs:
        existing = merged.get(subsystem.subsystem_id)
        if existing is None:
            merged[subsystem.subsystem_id] = subsystem
            continue
        merged[subsystem.subsystem_id] = SubsystemSpec(
            subsystem_id=subsystem.subsystem_id,
            summary=existing.summary or subsystem.summary,
            owned_paths=_dedupe_preserve_order(existing.owned_paths + subsystem.owned_paths),
            primary_signals=_dedupe_preserve_order(existing.primary_signals + subsystem.primary_signals),
            guardrail_signals=_dedupe_preserve_order(existing.guardrail_signals + subsystem.guardrail_signals),
            decomposition_hints=_dedupe_preserve_order(
                existing.decomposition_hints + subsystem.decomposition_hints
            ),
            notes=_dedupe_preserve_order(existing.notes + subsystem.notes),
        )
    return tuple(merged.values())


def _default_optimizable_artifacts(
    brief: ObjectiveBrief,
    *,
    mutable_paths: tuple[str, ...],
    entrypoints: tuple[str, ...],
) -> tuple[OptimizableArtifact, ...]:
    if brief.optimizable_artifacts:
        return brief.optimizable_artifacts
    location = brief.workspace_layout.candidate_dir or "."
    serialization_hint = "python_module" if any(path.endswith(".py") for path in (*mutable_paths, *entrypoints)) else ""
    description = (
        "Primary mutable candidate surface materialized in the local workspace."
        if mutable_paths or entrypoints
        else "Primary mutable candidate surface for this task."
    )
    return (
        OptimizableArtifact(
            artifact_id="candidate-surface",
            kind=OptimizableArtifactKind.WORKSPACE_SLICE,
            location=location,
            mutable_paths=_dedupe_preserve_order(mutable_paths + entrypoints),
            description=description,
            serialization_hint=serialization_hint,
        ),
    )


class DefaultObjectiveCompiler(ObjectiveCompiler):
    def compile(self, brief: ObjectiveBrief, skills: tuple[SkillPack, ...] = ()) -> TaskCharter:
        contributions = tuple(skill.contribute(brief) for skill in skills)
        signals = _merge_signals(brief=brief, contributions=contributions)
        subsystems = _merge_subsystems(brief=brief, contributions=contributions)
        constraints = _merge_constraints(brief=brief, contributions=contributions)
        hard_constraints = tuple(item for item in constraints if item.level == ConstraintLevel.HARD)
        soft_constraints = tuple(item for item in constraints if item.level == ConstraintLevel.SOFT)
        decomposition_hints = _dedupe_preserve_order(
            tuple(
                hint
                for contribution in contributions
                for hint in contribution.decomposition_hints
            )
        )
        notes = _dedupe_preserve_order(
            tuple(brief.notes)
            + tuple(
                note
                for contribution in contributions
                for note in (*contribution.notes, *contribution.evaluation_notes)
            )
        )
        mutable_paths = _dedupe_preserve_order(
            tuple(brief.mutable_paths)
            + tuple(path for contribution in contributions for path in contribution.suggested_mutable_paths)
            + tuple(path for subsystem in subsystems for path in subsystem.owned_paths)
        )
        protected_paths = _dedupe_preserve_order(
            tuple(brief.protected_paths)
            + tuple(path for contribution in contributions for path in contribution.suggested_protected_paths)
        )
        focus_subsystems = _dedupe_preserve_order(brief.focus_subsystems)
        unknown_focus = tuple(subsystem_id for subsystem_id in focus_subsystems if subsystem_id not in {item.subsystem_id for item in subsystems})
        if unknown_focus:
            raise ValueError(
                "ObjectiveBrief.focus_subsystems referenced unknown subsystem ids: "
                + ", ".join(unknown_focus)
            )
        optimizable_artifacts = _default_optimizable_artifacts(
            brief,
            mutable_paths=mutable_paths,
            entrypoints=_dedupe_preserve_order(brief.entrypoints),
        )
        charter_id = _slugify(brief.title)
        return TaskCharter(
            charter_id=charter_id,
            title=brief.title.strip(),
            objective_statement=brief.objective_statement.strip(),
            data_assets=brief.data_assets,
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            anti_goals=_dedupe_preserve_order(brief.anti_goals),
            evaluation_plan=EvaluationPlan(
                signals=signals,
                notes=_dedupe_preserve_order(
                    tuple(
                        note
                        for contribution in contributions
                        for note in contribution.evaluation_notes
                    )
                ),
            ),
            optimizable_artifacts=optimizable_artifacts,
            optimization_mode=brief.optimization_mode,
            search_strategy=brief.search_strategy,
            search_surface=SearchSurface(
                mutable_paths=mutable_paths,
                protected_paths=protected_paths,
                entrypoints=_dedupe_preserve_order(brief.entrypoints),
                subsystems=subsystems,
                notes=decomposition_hints,
            ),
            workspace_layout=brief.workspace_layout,
            execution_backend=brief.execution_backend,
            focus_subsystems=focus_subsystems,
            agent_runtime=brief.agent_runtime,
            agent_loop=brief.agent_loop,
            counterbalance=brief.counterbalance,
            run_commands=brief.run_commands,
            applied_skills=tuple(skill.skill_id for skill in skills),
            decomposition_hints=decomposition_hints,
            allowed_models=_dedupe_preserve_order(brief.allowed_models),
            disallowed_models=_dedupe_preserve_order(brief.disallowed_models),
            allowed_tools=_dedupe_preserve_order(brief.allowed_tools),
            notes=notes,
        )
