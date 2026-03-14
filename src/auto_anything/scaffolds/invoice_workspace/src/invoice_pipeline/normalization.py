from __future__ import annotations


def _normalize_space(value: str) -> str:
    return " ".join(str(value or "").split())


def normalize_invoice(fields: dict[str, str]) -> dict[str, str]:
    return {
        key: _normalize_space(value)
        for key, value in fields.items()
    }
