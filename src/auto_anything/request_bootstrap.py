from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

from .invoice_bootstrap import DEFAULT_MODEL
from .models import BootstrapPlan, DataAsset, ObjectiveBrief
from .open_bootstrap import bootstrap_open_task, build_open_objective_brief
from .task_family import infer_evaluation_mode


@dataclass(frozen=True)
class PlainTextTaskRequest:
    objective_statement: str
    referenced_paths: tuple[str, ...] = ()
    task_root: str = ""
    title: str = ""
    anti_goals: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    focus_subsystems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _normalize_words(value: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if token}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "task"


def resolve_referenced_paths(request: PlainTextTaskRequest) -> tuple[Path, ...]:
    return tuple(Path(raw).expanduser().resolve() for raw in request.referenced_paths)


def _asset_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if path.is_dir():
        suffixes = [child.suffix.lower() for child in path.iterdir() if child.is_file()]
        if not suffixes:
            return "directory"
        top_suffixes = {item for item, _ in Counter(suffixes).most_common(2)}
        if top_suffixes <= {".pdf"}:
            return "pdf_corpus"
        if top_suffixes <= {".png", ".jpg", ".jpeg", ".webp"}:
            return "image_corpus"
        if top_suffixes <= {".json"}:
            return "json_corpus"
        if {".png", ".jpg", ".jpeg", ".webp"} & top_suffixes and ".json" in top_suffixes:
            return "image_with_labels_corpus"
        if ".pdf" in top_suffixes and ".json" in top_suffixes:
            return "pdf_with_labels_corpus"
        return "mixed_corpus"
    if suffix == ".pdf":
        return "pdf_document"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image_document"
    if suffix == ".json":
        return "json_document"
    return "file"


def infer_task_root(request: PlainTextTaskRequest) -> Path:
    if request.task_root:
        return Path(request.task_root).expanduser().resolve()
    title = request.title.strip() or request.objective_statement.strip().split(".")[0][:80] or "Task request"
    return (Path.cwd() / "work" / _slugify(title)).resolve()


def build_brief_from_request(request: PlainTextTaskRequest) -> ObjectiveBrief:
    referenced_paths = resolve_referenced_paths(request)
    missing_paths = tuple(str(path) for path in referenced_paths if not path.exists())
    if missing_paths:
        raise FileNotFoundError(f"Referenced paths do not exist: {', '.join(missing_paths)}")
    data_assets = tuple(
        DataAsset(
            name=path.name,
            kind=_asset_kind_for_path(path),
            location=str(path),
            role="train_eval_corpus",
        )
        for path in referenced_paths
    )
    title = request.title.strip() or request.objective_statement.strip().split(".")[0][:80] or "Task request"
    return build_open_objective_brief(
        title=title,
        data_assets=data_assets,
        objective=request.objective_statement,
        anti_goals=request.anti_goals,
        constraints=request.constraints,
        allowed_models=request.allowed_models or (DEFAULT_MODEL,),
        focus_subsystems=request.focus_subsystems,
        notes=request.notes + ("Built from plain-text request plus referenced paths.",),
    )


def build_bootstrap_plan_from_request(
    request: PlainTextTaskRequest,
) -> BootstrapPlan:
    brief = build_brief_from_request(request)
    evaluation_mode = infer_evaluation_mode(brief)
    return BootstrapPlan(
        family_id="open-ended-task",
        evaluation_mode=evaluation_mode,
        scaffold_id="open-task-workspace",
        rationale=(
            "Synthesizing a generic task workspace from the plain-text request and referenced paths.",
            "No predefined task family is required for bootstrap.",
            "The calling agent is expected to finish shaping the pipeline and evaluator inside the workspace.",
        ),
        inferred_objective_signals=tuple(signal.name for signal in brief.explicit_signals),
        notes=(
            "Bootstrap creates a useful starting world, not a finished task solution.",
        ),
    )


def bootstrap_task_from_request(
    request: PlainTextTaskRequest,
) -> Path:
    brief = build_brief_from_request(request)
    task_root = infer_task_root(request)
    return bootstrap_open_task(
        task_root=task_root,
        title=brief.title,
        objective=brief.objective_statement,
        data_assets=brief.data_assets,
        anti_goals=brief.anti_goals,
        constraints=brief.constraints,
        allowed_models=brief.allowed_models or (DEFAULT_MODEL,),
        focus_subsystems=brief.focus_subsystems,
        notes=brief.notes,
    )
