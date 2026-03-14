from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from invoice_pipeline.document_io import extract_text, render_pdf_pages
from invoice_pipeline.field_extractors import extract_fields_from_text
from invoice_pipeline.normalization import normalize_invoice
from invoice_pipeline.openrouter_client import extract_invoice_from_images
from invoice_pipeline.schema import validate_invoice


def extract_invoice_with_meta(pdf_path: str | Path) -> tuple[dict[str, str], dict[str, object]]:
    path = Path(pdf_path)
    document_text = extract_text(path)
    fallback_meta: dict[str, object] = {
        "used_model": "",
        "used_vision": False,
        "usage": {"total_tokens": 0},
        "fallback_reason": "",
    }

    if os.getenv("OPENROUTER_API_KEY"):
        try:
            with tempfile.TemporaryDirectory(prefix="invoice-pages-") as tmpdir:
                image_paths = render_pdf_pages(path, Path(tmpdir))
                raw_fields, remote_meta = extract_invoice_from_images(
                    image_paths=image_paths,
                    text_hint=document_text,
                )
            normalized = normalize_invoice(raw_fields)
            return validate_invoice(normalized), remote_meta
        except Exception as exc:
            fallback_meta["fallback_reason"] = str(exc)

    raw_fields = extract_fields_from_text(document_text)
    normalized = normalize_invoice(raw_fields)
    return validate_invoice(normalized), fallback_meta


def extract_invoice(pdf_path: str | Path) -> dict[str, str]:
    payload, _ = extract_invoice_with_meta(pdf_path)
    return payload


if __name__ == "__main__":
    import sys

    payload, meta = extract_invoice_with_meta(sys.argv[1])
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))
