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
