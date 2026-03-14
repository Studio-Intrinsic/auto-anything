from __future__ import annotations

import re


def _search(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_fields_from_text(text: str) -> dict[str, str]:
    return {
        "invoice_number": _search(r"Invoice Number\s+([A-Z0-9-]+)", text),
        "order_number": _search(r"Order Number\s+([A-Z0-9-]+)", text),
        "invoice_date": _search(r"Invoice Date\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text),
        "due_date": _search(r"Due Date\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text),
        "total_due": _search(r"Total Due\s+(\$[0-9.,]+)", text),
        "vendor_name": _search(r"From:\s+(.+)", text),
        "customer_name": _search(r"To:\s+(.+)", text),
    }


def coerce_model_output(payload: dict[str, object]) -> dict[str, str]:
    return {
        "invoice_number": str(payload.get("invoice_number", "") or "").strip(),
        "order_number": str(payload.get("order_number", "") or "").strip(),
        "invoice_date": str(payload.get("invoice_date", "") or "").strip(),
        "due_date": str(payload.get("due_date", "") or "").strip(),
        "total_due": str(payload.get("total_due", "") or "").strip(),
        "vendor_name": str(payload.get("vendor_name", "") or "").strip(),
        "customer_name": str(payload.get("customer_name", "") or "").strip(),
    }
