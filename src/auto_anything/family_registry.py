from __future__ import annotations

from .models import EvaluationMode, ScaffoldPackSpec, TaskFamilySpec
from .task_family import TaskFamilyRegistry


INVOICE_WORKSPACE_SCAFFOLD = ScaffoldPackSpec(
    scaffold_id="invoice-vision-workspace",
    scaffold_dir="scaffolds/invoice_workspace",
    description="Vision-first invoice extraction workspace scaffold.",
    placeholder_keys=("__AUTO_ANYTHING_LIBRARY_SRC__", "__AUTO_ANYTHING_DEFAULT_MODEL__"),
)


def get_default_task_family_registry() -> TaskFamilyRegistry:
    registry = TaskFamilyRegistry()
    registry.register(
        TaskFamilySpec(
            family_id="invoice-document-extraction",
            summary="Extract structured invoice fields from PDFs using a modular pipeline.",
            evaluation_modes=(
                EvaluationMode.EXPLICIT_BENCHMARK,
                EvaluationMode.PARTIAL_LABELS,
                EvaluationMode.WEAK_SUPERVISION,
            ),
            default_evaluation_mode=EvaluationMode.PARTIAL_LABELS,
            scaffold=INVOICE_WORKSPACE_SCAFFOLD,
            data_kinds=("invoice_corpus", "pdf_corpus", "document_corpus"),
            objective_keywords=("invoice", "pdf", "extract", "extraction", "fields", "ocr", "vision"),
        )
    )
    return registry
