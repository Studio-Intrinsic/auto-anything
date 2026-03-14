from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .family_registry import get_default_task_family_registry
from .invoice_bootstrap import DEFAULT_MODEL, bootstrap_invoice_task
from .models import BootstrapPlan, DataAsset, ObjectiveBrief
from .task_family import TaskFamilyRegistry, build_bootstrap_plan, infer_task_family


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


def _asset_kind_for_path(path: Path, objective_statement: str) -> str:
    objective_terms = _normalize_words(objective_statement)
    if path.is_dir():
        has_pdfs = any(child.suffix.lower() == ".pdf" for child in path.iterdir() if child.is_file())
        if has_pdfs:
            return "invoice_corpus" if "invoice" in objective_terms else "pdf_corpus"
        return "document_corpus"
    if path.suffix.lower() == ".pdf":
        return "invoice_corpus" if "invoice" in objective_terms else "pdf_corpus"
    return "document_corpus"


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
            kind=_asset_kind_for_path(path, request.objective_statement),
            location=str(path),
            role="train_eval_corpus",
        )
        for path in referenced_paths
    )
    title = request.title.strip() or request.objective_statement.strip().split(".")[0][:80] or "Task request"
    return ObjectiveBrief(
        title=title,
        objective_statement=request.objective_statement,
        data_assets=data_assets,
        anti_goals=request.anti_goals,
        constraints=request.constraints,
        allowed_models=request.allowed_models or (DEFAULT_MODEL,),
        focus_subsystems=request.focus_subsystems,
        notes=request.notes + ("Built from plain-text request plus referenced paths.",),
    )


def build_bootstrap_plan_from_request(
    request: PlainTextTaskRequest,
    *,
    registry: TaskFamilyRegistry | None = None,
) -> BootstrapPlan:
    registry = registry or get_default_task_family_registry()
    brief = build_brief_from_request(request)
    return build_bootstrap_plan(registry, brief)


def bootstrap_task_from_request(
    request: PlainTextTaskRequest,
    *,
    registry: TaskFamilyRegistry | None = None,
) -> Path:
    registry = registry or get_default_task_family_registry()
    brief = build_brief_from_request(request)
    family = infer_task_family(registry, brief)
    referenced_paths = resolve_referenced_paths(request)

    if family.family_id == "invoice-document-extraction":
        if not referenced_paths:
            raise ValueError("Invoice bootstrap requires at least one referenced path.")
        data_roots = [path for path in referenced_paths if path.is_dir()]
        if data_roots:
            data_dir = data_roots[0]
        else:
            first_file = referenced_paths[0]
            data_dir = first_file.parent
        task_root = infer_task_root(request)
        model = request.allowed_models[0] if request.allowed_models else DEFAULT_MODEL
        return bootstrap_invoice_task(
            task_root=task_root,
            data_dir=data_dir,
            objective=request.objective_statement,
            model=model,
            focus_subsystems=request.focus_subsystems,
        )

    raise NotImplementedError(f"No bootstrap dispatcher is registered for task family '{family.family_id}'.")
