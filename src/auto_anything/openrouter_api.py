from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def _coerce_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _coerce_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def _read_openrouter_json(
    *,
    path: str,
    api_key: str = "",
    timeout_seconds: int = 30,
    method: str = "GET",
    body: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Studio-Intrinsic/auto-anything",
        "X-Title": "auto-anything",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        urllib.parse.urljoin(f"{OPENROUTER_API_BASE}/", path.lstrip("/")),
        data=payload,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class OpenRouterPricing:
    prompt: float = 0.0
    completion: float = 0.0
    image: float = 0.0
    request: float = 0.0
    web_search: float = 0.0
    internal_reasoning: float = 0.0
    input_cache_read: float = 0.0
    input_cache_write: float = 0.0


@dataclass(frozen=True)
class OpenRouterModel:
    model_id: str
    canonical_slug: str = ""
    name: str = ""
    description: str = ""
    created: int = 0
    context_length: int = 0
    architecture_modality: str = ""
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    supported_parameters: tuple[str, ...] = ()
    pricing: OpenRouterPricing = field(default_factory=OpenRouterPricing)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenRouterUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    image_tokens: int = 0
    audio_tokens: int = 0
    video_tokens: int = 0
    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    used_exact_cost: bool = False
    generation_id: str = ""
    model_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenRouterGeneration:
    generation_id: str
    model_id: str
    total_cost_usd: float
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    native_prompt_tokens: int
    native_completion_tokens: int
    native_reasoning_tokens: int
    native_cached_tokens: int
    provider_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def parse_openrouter_model(payload: Mapping[str, object]) -> OpenRouterModel:
    architecture = payload.get("architecture", {}) if isinstance(payload.get("architecture"), Mapping) else {}
    pricing_payload = payload.get("pricing", {}) if isinstance(payload.get("pricing"), Mapping) else {}
    return OpenRouterModel(
        model_id=str(payload.get("id", "")),
        canonical_slug=str(payload.get("canonical_slug", "")),
        name=str(payload.get("name", "")),
        description=str(payload.get("description", "")),
        created=_coerce_int(payload.get("created", 0)),
        context_length=_coerce_int(payload.get("context_length", 0)),
        architecture_modality=str(architecture.get("modality", "")),
        input_modalities=tuple(str(item) for item in architecture.get("input_modalities", []) if item),
        output_modalities=tuple(str(item) for item in architecture.get("output_modalities", []) if item),
        supported_parameters=tuple(str(item) for item in payload.get("supported_parameters", []) if item),
        pricing=OpenRouterPricing(
            prompt=_coerce_float(pricing_payload.get("prompt")),
            completion=_coerce_float(pricing_payload.get("completion")),
            image=_coerce_float(pricing_payload.get("image")),
            request=_coerce_float(pricing_payload.get("request")),
            web_search=_coerce_float(pricing_payload.get("web_search")),
            internal_reasoning=_coerce_float(pricing_payload.get("internal_reasoning")),
            input_cache_read=_coerce_float(pricing_payload.get("input_cache_read")),
            input_cache_write=_coerce_float(pricing_payload.get("input_cache_write")),
        ),
        raw=dict(payload),
    )


def list_openrouter_models(
    *,
    api_key_env: str = "OPENROUTER_API_KEY",
    available_only: bool = False,
    timeout_seconds: int = 30,
) -> tuple[OpenRouterModel, ...]:
    api_key = os.getenv(api_key_env, "").strip()
    path = "models/user" if available_only and api_key else "models"
    raw = _read_openrouter_json(path=path, api_key=api_key, timeout_seconds=timeout_seconds)
    return tuple(parse_openrouter_model(item) for item in raw.get("data", []) if isinstance(item, Mapping))


def get_openrouter_model(
    model_id: str,
    *,
    api_key_env: str = "OPENROUTER_API_KEY",
    available_only: bool = False,
    timeout_seconds: int = 30,
) -> OpenRouterModel | None:
    for model in list_openrouter_models(
        api_key_env=api_key_env,
        available_only=available_only,
        timeout_seconds=timeout_seconds,
    ):
        if model.model_id == model_id or model.canonical_slug == model_id:
            return model
    return None


def estimate_openrouter_cost(*, usage: OpenRouterUsage, pricing: OpenRouterPricing) -> float:
    return (
        (usage.prompt_tokens * pricing.prompt)
        + (usage.completion_tokens * pricing.completion)
        + (usage.reasoning_tokens * pricing.internal_reasoning)
        + (usage.cached_tokens * pricing.input_cache_read)
        + (usage.cache_write_tokens * pricing.input_cache_write)
        + (usage.image_tokens * pricing.image)
        + pricing.request
    )


def extract_openrouter_usage(
    payload: Mapping[str, object],
    *,
    model: OpenRouterModel | None = None,
) -> OpenRouterUsage:
    usage_payload = payload.get("usage", payload) if isinstance(payload.get("usage"), Mapping) else payload
    prompt_details = (
        usage_payload.get("prompt_tokens_details", {})
        if isinstance(usage_payload.get("prompt_tokens_details"), Mapping)
        else {}
    )
    completion_details = (
        usage_payload.get("completion_tokens_details", {})
        if isinstance(usage_payload.get("completion_tokens_details"), Mapping)
        else {}
    )
    exact_cost = _coerce_float(usage_payload.get("cost") or payload.get("total_cost") or usage_payload.get("total_cost"))
    usage = OpenRouterUsage(
        prompt_tokens=_coerce_int(usage_payload.get("prompt_tokens")),
        completion_tokens=_coerce_int(usage_payload.get("completion_tokens")),
        total_tokens=_coerce_int(usage_payload.get("total_tokens")),
        reasoning_tokens=_coerce_int(completion_details.get("reasoning_tokens")),
        cached_tokens=_coerce_int(prompt_details.get("cached_tokens")),
        cache_write_tokens=_coerce_int(prompt_details.get("cache_write_tokens")),
        image_tokens=_coerce_int(completion_details.get("image_tokens")),
        audio_tokens=_coerce_int(completion_details.get("audio_tokens")) + _coerce_int(prompt_details.get("audio_tokens")),
        video_tokens=_coerce_int(prompt_details.get("video_tokens")),
        cost_usd=exact_cost,
        used_exact_cost=exact_cost > 0,
        generation_id=str(payload.get("id", "")),
        model_id=str(payload.get("model", "")),
        raw=dict(usage_payload if isinstance(usage_payload, Mapping) else {}),
    )
    estimated_cost = usage.cost_usd
    if usage.cost_usd == 0.0 and model is not None:
        estimated_cost = estimate_openrouter_cost(usage=usage, pricing=model.pricing)
    return OpenRouterUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cached_tokens=usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        image_tokens=usage.image_tokens,
        audio_tokens=usage.audio_tokens,
        video_tokens=usage.video_tokens,
        cost_usd=usage.cost_usd,
        estimated_cost_usd=estimated_cost,
        used_exact_cost=usage.used_exact_cost,
        generation_id=usage.generation_id,
        model_id=usage.model_id,
        raw=usage.raw,
    )


def fetch_openrouter_generation(
    generation_id: str,
    *,
    api_key_env: str = "OPENROUTER_API_KEY",
    timeout_seconds: int = 30,
) -> OpenRouterGeneration:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")
    raw = _read_openrouter_json(
        path=f"generation?id={urllib.parse.quote(generation_id, safe='')}",
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    payload = raw.get("data", {}) if isinstance(raw.get("data"), Mapping) else {}
    return OpenRouterGeneration(
        generation_id=str(payload.get("id", generation_id)),
        model_id=str(payload.get("model", "")),
        total_cost_usd=_coerce_float(payload.get("total_cost") or payload.get("usage")),
        latency_ms=_coerce_int(payload.get("latency")),
        prompt_tokens=_coerce_int(payload.get("tokens_prompt")),
        completion_tokens=_coerce_int(payload.get("tokens_completion")),
        native_prompt_tokens=_coerce_int(payload.get("native_tokens_prompt")),
        native_completion_tokens=_coerce_int(payload.get("native_tokens_completion")),
        native_reasoning_tokens=_coerce_int(payload.get("native_tokens_reasoning")),
        native_cached_tokens=_coerce_int(payload.get("native_tokens_cached")),
        provider_name=str(payload.get("provider_name", "")),
        raw=dict(payload),
    )


__all__ = [
    "OPENROUTER_API_BASE",
    "OpenRouterGeneration",
    "OpenRouterModel",
    "OpenRouterPricing",
    "OpenRouterUsage",
    "estimate_openrouter_cost",
    "extract_openrouter_usage",
    "fetch_openrouter_generation",
    "get_openrouter_model",
    "list_openrouter_models",
    "parse_openrouter_model",
]
