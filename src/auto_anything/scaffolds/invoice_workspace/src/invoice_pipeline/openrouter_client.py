from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from invoice_pipeline.document_io import image_to_data_url
from invoice_pipeline.field_extractors import coerce_model_output


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "__AUTO_ANYTHING_DEFAULT_MODEL__"


def _response_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    raise ValueError("Unsupported response content shape from OpenRouter.")


def _extract_json_block(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model response did not contain JSON: {text}")
    return json.loads(stripped[start:end + 1])


def extract_invoice_from_images(
    *,
    image_paths: list[Path],
    text_hint: str,
    model: str | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    chosen_model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Extract invoice fields from these invoice page images. "
                "Return JSON only with the keys: invoice_number, order_number, "
                "invoice_date, due_date, total_due, vendor_name, customer_name. "
                "Prefer exact strings from the invoice. Use empty strings when unknown. "
                f"Helpful OCR/text hint: {text_hint[:4000]}"
            ),
        }
    ]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image_path),
                },
            }
        )

    payload = {
        "model": chosen_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a careful invoice extraction system. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Studio-Intrinsic/auto-anything",
            "X-Title": "auto-anything invoice extraction",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = json.loads(response.read().decode("utf-8"))

    message = raw["choices"][0]["message"]
    parsed = _extract_json_block(_response_text(message.get("content", "")))
    usage = raw.get("usage", {})
    meta = {
        "used_model": chosen_model,
        "used_vision": True,
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
        "fallback_reason": "",
    }
    return coerce_model_output(parsed), meta
