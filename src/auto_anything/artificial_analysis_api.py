from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping


ARTIFICIAL_ANALYSIS_API_BASE = "https://artificialanalysis.ai/api/v2"


def _coerce_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _read_artificial_analysis_json(
    *,
    path: str,
    api_key_env: str = "ARTIFICIAL_ANALYSIS_API_KEY",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")
    request = urllib.request.Request(
        urllib.parse.urljoin(f"{ARTIFICIAL_ANALYSIS_API_BASE}/", path.lstrip("/")),
        headers={"x-api-key": api_key},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class ArtificialAnalysisPricing:
    price_1m_blended_3_to_1: float | None = None
    price_1m_input_tokens: float | None = None
    price_1m_output_tokens: float | None = None


@dataclass(frozen=True)
class ArtificialAnalysisLLM:
    model_id: str
    name: str
    slug: str
    creator_id: str = ""
    creator_name: str = ""
    creator_slug: str = ""
    evaluations: dict[str, float | None] = field(default_factory=dict)
    pricing: ArtificialAnalysisPricing = field(default_factory=ArtificialAnalysisPricing)
    median_output_tokens_per_second: float | None = None
    median_time_to_first_token_seconds: float | None = None
    median_time_to_first_answer_token: float | None = None
    release_date: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def parse_artificial_analysis_llm(payload: Mapping[str, object]) -> ArtificialAnalysisLLM:
    creator = payload.get("model_creator", {}) if isinstance(payload.get("model_creator"), Mapping) else {}
    evaluations = payload.get("evaluations", {}) if isinstance(payload.get("evaluations"), Mapping) else {}
    pricing_payload = payload.get("pricing", {}) if isinstance(payload.get("pricing"), Mapping) else {}
    return ArtificialAnalysisLLM(
        model_id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        slug=str(payload.get("slug", "")),
        creator_id=str(creator.get("id", "")),
        creator_name=str(creator.get("name", "")),
        creator_slug=str(creator.get("slug", "")),
        evaluations={str(key): _coerce_float(value) for key, value in evaluations.items()},
        pricing=ArtificialAnalysisPricing(
            price_1m_blended_3_to_1=_coerce_float(pricing_payload.get("price_1m_blended_3_to_1")),
            price_1m_input_tokens=_coerce_float(pricing_payload.get("price_1m_input_tokens")),
            price_1m_output_tokens=_coerce_float(pricing_payload.get("price_1m_output_tokens")),
        ),
        median_output_tokens_per_second=_coerce_float(payload.get("median_output_tokens_per_second")),
        median_time_to_first_token_seconds=_coerce_float(payload.get("median_time_to_first_token_seconds")),
        median_time_to_first_answer_token=_coerce_float(payload.get("median_time_to_first_answer_token")),
        release_date=str(payload.get("release_date", "")),
        raw=dict(payload),
    )


def list_artificial_analysis_llms(
    *,
    api_key_env: str = "ARTIFICIAL_ANALYSIS_API_KEY",
    prompt_length: str = "medium",
    parallel_queries: int = 1,
    timeout_seconds: int = 30,
) -> tuple[ArtificialAnalysisLLM, ...]:
    query = urllib.parse.urlencode(
        {
            "prompt_length": prompt_length,
            "parallel_queries": parallel_queries,
        }
    )
    raw = _read_artificial_analysis_json(
        path=f"data/llms/models?{query}",
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
    )
    return tuple(parse_artificial_analysis_llm(item) for item in raw.get("data", []) if isinstance(item, Mapping))


def shortlist_artificial_analysis_llms(
    models: tuple[ArtificialAnalysisLLM, ...],
    *,
    benchmark_key: str = "artificial_analysis_intelligence_index",
    max_blended_price_1m: float | None = None,
    min_output_tokens_per_second: float | None = None,
    creator_slug: str = "",
    limit: int = 10,
) -> tuple[ArtificialAnalysisLLM, ...]:
    filtered: list[ArtificialAnalysisLLM] = []
    for model in models:
        if creator_slug and model.creator_slug != creator_slug:
            continue
        if max_blended_price_1m is not None:
            price = model.pricing.price_1m_blended_3_to_1
            if price is None or price > max_blended_price_1m:
                continue
        if min_output_tokens_per_second is not None:
            speed = model.median_output_tokens_per_second
            if speed is None or speed < min_output_tokens_per_second:
                continue
        filtered.append(model)
    ranked = sorted(
        filtered,
        key=lambda item: (
            item.evaluations.get(benchmark_key) is None,
            -(item.evaluations.get(benchmark_key) or 0.0),
            item.pricing.price_1m_blended_3_to_1 if item.pricing.price_1m_blended_3_to_1 is not None else float("inf"),
            -(item.median_output_tokens_per_second or 0.0),
        ),
    )
    return tuple(ranked[:limit])


__all__ = [
    "ARTIFICIAL_ANALYSIS_API_BASE",
    "ArtificialAnalysisLLM",
    "ArtificialAnalysisPricing",
    "list_artificial_analysis_llms",
    "parse_artificial_analysis_llm",
    "shortlist_artificial_analysis_llms",
]
