from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import BootstrapPlan, EvaluationMode, ObjectiveBrief, ScaffoldPackSpec, TaskFamilySpec


def _normalize_words(value: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if token}


@dataclass
class TaskFamilyRegistry:
    families: dict[str, TaskFamilySpec]

    def __init__(self, families: tuple[TaskFamilySpec, ...] = ()) -> None:
        self.families = {family.family_id: family for family in families}

    def register(self, family: TaskFamilySpec) -> None:
        self.families[family.family_id] = family

    def get(self, family_id: str) -> TaskFamilySpec:
        return self.families[family_id]

    def all(self) -> tuple[TaskFamilySpec, ...]:
        return tuple(self.families.values())


def infer_evaluation_mode(brief: ObjectiveBrief) -> EvaluationMode:
    if brief.explicit_signals:
        return EvaluationMode.EXPLICIT_BENCHMARK
    labeled_roles = {"golden", "labels", "ground_truth", "eval_corpus", "train_eval_corpus"}
    if any(asset.role in labeled_roles for asset in brief.data_assets):
        return EvaluationMode.PARTIAL_LABELS
    return EvaluationMode.WEAK_SUPERVISION


def score_task_family_match(family: TaskFamilySpec, brief: ObjectiveBrief) -> tuple[int, int, int]:
    objective_terms = _normalize_words(brief.objective_statement)
    keyword_hits = len(objective_terms & set(family.objective_keywords))
    data_hits = sum(1 for asset in brief.data_assets if asset.kind in family.data_kinds)
    family_hint = 1 if family.family_id in _normalize_words(" ".join(brief.notes)) else 0
    return (data_hits, keyword_hits, family_hint)


def infer_task_family(registry: TaskFamilyRegistry, brief: ObjectiveBrief) -> TaskFamilySpec:
    if not registry.families:
        raise ValueError("TaskFamilyRegistry is empty.")
    ranked = sorted(
        ((score_task_family_match(family, brief), family) for family in registry.all()),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_family = ranked[0]
    if best_score == (0, 0, 0):
        return best_family
    return best_family


def build_bootstrap_plan(registry: TaskFamilyRegistry, brief: ObjectiveBrief) -> BootstrapPlan:
    family = infer_task_family(registry, brief)
    evaluation_mode = infer_evaluation_mode(brief)
    if evaluation_mode not in family.evaluation_modes:
        evaluation_mode = family.default_evaluation_mode
    rationale = [
        f"Selected task family '{family.family_id}' because it matches the available data and objective.",
        f"Using evaluation mode '{evaluation_mode.value}'.",
        f"Using scaffold pack '{family.scaffold.scaffold_id}'.",
    ]
    signal_names = tuple(signal.name for signal in brief.explicit_signals)
    return BootstrapPlan(
        family_id=family.family_id,
        evaluation_mode=evaluation_mode,
        scaffold_id=family.scaffold.scaffold_id,
        rationale=tuple(rationale),
        inferred_objective_signals=signal_names,
        notes=family.notes,
    )


def scaffold_pack_path(base_dir: Path, scaffold: ScaffoldPackSpec) -> Path:
    return (base_dir / scaffold.scaffold_dir).resolve()
