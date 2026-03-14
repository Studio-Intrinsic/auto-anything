from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from dataclasses import asdict
from pathlib import Path

from .compiler import DefaultObjectiveCompiler
from .models import (
    AgentLoopSpec,
    AgentRole,
    AgentRuntimeConfig,
    Constraint,
    ConstraintLevel,
    DataAsset,
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
from .workspace import resolve_workspace_paths


DEFAULT_MODEL = "x-ai/grok-4.1-fast"


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
                    owned_paths=("src/invoice_pipeline/field_extractors.py", "src/invoice_pipeline/extract.py"),
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


def _extractor_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path

        from invoice_pipeline.document_io import extract_text
        from invoice_pipeline.field_extractors import extract_fields
        from invoice_pipeline.normalization import normalize_invoice
        from invoice_pipeline.schema import validate_invoice


        def extract_invoice(pdf_path: str | Path) -> dict[str, str]:
            path = Path(pdf_path)
            document_text = extract_text(path)
            raw_fields = extract_fields(document_text)
            normalized = normalize_invoice(raw_fields)
            return validate_invoice(normalized)


        if __name__ == "__main__":
            import sys

            payload = extract_invoice(sys.argv[1])
            print(json.dumps(payload, indent=2, sort_keys=True))
        """
    ).strip() + "\n"


def _document_io_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import subprocess
        from pathlib import Path


        def extract_text(pdf_path: Path) -> str:
            completed = subprocess.run(
                ["pdftotext", str(pdf_path), "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout
        """
    ).strip() + "\n"


def _field_extractors_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import re


        def _search(pattern: str, text: str) -> str:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else ""


        def extract_fields(text: str) -> dict[str, str]:
            return {
                "invoice_number": _search(r"Invoice Number\\s+([A-Z0-9-]+)", text),
                "order_number": _search(r"Order Number\\s+([A-Z0-9-]+)", text),
                "invoice_date": _search(r"Invoice Date\\s+([A-Za-z]+\\s+\\d{1,2},\\s+\\d{4})", text),
                "due_date": _search(r"Due Date\\s+([A-Za-z]+\\s+\\d{1,2},\\s+\\d{4})", text),
                "total_due": _search(r"Total Due\\s+(\\$[0-9.,]+)", text),
                "vendor_name": _search(r"From:\\s+(.+)", text),
                "customer_name": _search(r"To:\\s+(.+)", text),
            }
        """
    ).strip() + "\n"


def _normalization_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations


        def _normalize_space(value: str) -> str:
            return " ".join(str(value or "").split())


        def normalize_invoice(fields: dict[str, str]) -> dict[str, str]:
            return {
                key: _normalize_space(value)
                for key, value in fields.items()
            }
        """
    ).strip() + "\n"


def _schema_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations


        REQUIRED_KEYS = (
            "invoice_number",
            "order_number",
            "invoice_date",
            "due_date",
            "total_due",
            "vendor_name",
            "customer_name",
        )


        def validate_invoice(payload: dict[str, str]) -> dict[str, str]:
            validated: dict[str, str] = {}
            for key in REQUIRED_KEYS:
                value = payload.get(key, "")
                validated[key] = str(value or "").strip()
            return validated
        """
    ).strip() + "\n"


def _eval_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import time
        from pathlib import Path
        import sys

        ROOT = Path(__file__).resolve().parents[1]
        SRC = ROOT / "src"
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))

        from invoice_pipeline.extract import extract_invoice


        def _normalize(value: str) -> str:
            return " ".join(str(value or "").strip().lower().split())


        def main() -> int:
            fixtures_dir = ROOT / "fixtures"
            goldens_dir = ROOT / "goldens"
            pdf_paths = sorted(fixtures_dir.glob("*.pdf"))

            total_expected = 0
            total_correct = 0
            passed_docs = 0
            schema_valid = 1.0
            latencies = []

            predictions_dir = ROOT / "artifacts" / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)

            for pdf_path in pdf_paths:
                gold_path = goldens_dir / f"{pdf_path.stem}.expected.json"
                gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.is_file() else {}
                t0 = time.perf_counter()
                prediction = extract_invoice(pdf_path)
                latencies.append(time.perf_counter() - t0)
                (predictions_dir / f"{pdf_path.stem}.predicted.json").write_text(
                    json.dumps(prediction, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

                doc_expected = 0
                doc_correct = 0
                for key, expected_value in gold.items():
                    doc_expected += 1
                    total_expected += 1
                    if _normalize(prediction.get(key, "")) == _normalize(expected_value):
                        doc_correct += 1
                        total_correct += 1

                if doc_expected > 0 and doc_correct == doc_expected:
                    passed_docs += 1

                if not isinstance(prediction, dict):
                    schema_valid = 0.0

            doc_count = len(pdf_paths)
            field_accuracy = (total_correct / total_expected) if total_expected else 0.0
            document_pass_rate = (passed_docs / doc_count) if doc_count else 0.0
            latency_seconds = (sum(latencies) / len(latencies)) if latencies else 0.0
            token_cost = 0.0

            summary = {
                "candidate_id": "baseline",
                "signals": {
                    "field_accuracy": field_accuracy,
                    "document_pass_rate": document_pass_rate,
                    "schema_valid": schema_valid,
                    "latency_seconds": latency_seconds,
                    "token_cost": token_cost,
                },
                "doc_count": doc_count,
                "total_expected_fields": total_expected,
                "total_correct_fields": total_correct,
            }
            summary_path = ROOT / "artifacts" / "eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip() + "\n"


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
        5. inspect `artifacts/eval_summary.json`
        6. switch into a critic stance and look for brittle assumptions or gaming
        7. keep iterating

        Starter subsystem ownership:

        - `document-ingestion`: `src/invoice_pipeline/document_io.py`
        - `field-extraction`: `src/invoice_pipeline/field_extractors.py`, `src/invoice_pipeline/extract.py`
        - `normalization`: `src/invoice_pipeline/normalization.py`
        - `schema-validation`: `src/invoice_pipeline/schema.py`

        This split exists so the same agent can focus work on one subsystem without losing
        whole-pipeline eval guardrails.
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

    _copy_dataset(data_dir, task_root / "fixtures", task_root / "goldens")
    (paths.candidate_dir / "__init__.py").write_text("", encoding="utf-8")
    (paths.candidate_dir / "extract.py").write_text(_extractor_source(), encoding="utf-8")
    (paths.candidate_dir / "document_io.py").write_text(_document_io_source(), encoding="utf-8")
    (paths.candidate_dir / "field_extractors.py").write_text(_field_extractors_source(), encoding="utf-8")
    (paths.candidate_dir / "normalization.py").write_text(_normalization_source(), encoding="utf-8")
    (paths.candidate_dir / "schema.py").write_text(_schema_source(), encoding="utf-8")
    (paths.evaluator_dir / "run_invoice_eval.py").write_text(_eval_source(), encoding="utf-8")
    (task_root / "README.md").write_text(_task_readme_source(task_root.name), encoding="utf-8")
    (task_root / "task_charter.json").write_text(
        json.dumps(asdict(charter), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return task_root


def run_bootstrapped_eval(task_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "eval/run_invoice_eval.py"],
        cwd=str(task_root),
        check=True,
        capture_output=True,
        text=True,
    )
