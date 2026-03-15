from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from .artificial_analysis_api import ArtificialAnalysisLLM, list_artificial_analysis_llms
from .openrouter_api import OpenRouterModel, list_openrouter_models


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_TOKENS = {
    "preview",
    "chat",
    "image",
    "mini",
    "reasoning",
    "non",
    "low",
    "medium",
    "high",
    "pro",
    "codex",
}


@dataclass(frozen=True)
class ModelSelectionCandidate:
    openrouter_model_id: str
    creator_slug: str
    modalities: tuple[str, ...]
    openrouter_prompt_price_1m: float
    openrouter_completion_price_1m: float
    artificial_analysis_slug: str = ""
    artificial_analysis_name: str = ""
    benchmark_score: float | None = None
    coding_score: float | None = None
    blended_price_1m: float | None = None
    output_tokens_per_second: float | None = None
    release_date: str = ""
    match_confidence: float = 0.0
    composite_score: float = 0.0
    notes: tuple[str, ...] = ()
    raw_tokens: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelCatalogMatch:
    openrouter_model: OpenRouterModel
    artificial_analysis_model: ArtificialAnalysisLLM | None
    creator_slug: str
    match_confidence: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSelectionWeights:
    benchmark: float = 1.0
    speed: float = 20.0 / 300.0
    blended_price_penalty: float = 3.0
    openrouter_price_penalty: float = 1.5
    recency: float = 1.0
    match_confidence: float = 10.0


def _creator_slug_from_openrouter(model_id: str) -> str:
    prefix = model_id.split("/", 1)[0].strip().lower()
    if prefix == "x-ai":
        return "xai"
    return prefix


def _tokens(value: str) -> set[str]:
    parts = {token for token in _TOKEN_RE.findall(value.lower()) if token and token not in _STOP_TOKENS}
    return parts


def _parse_release_date(value: str) -> date | None:
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%b %Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m":
                return date(parsed.year, parsed.month, 1)
            if fmt == "%b %Y":
                return date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def _recency_bonus(release_date: str) -> float:
    parsed = _parse_release_date(release_date)
    if parsed is None:
        return 0.0
    age_days = max((date.today() - parsed).days, 0)
    if age_days <= 90:
        return 8.0
    if age_days <= 180:
        return 5.0
    if age_days <= 365:
        return 2.0
    return 0.0


def _match_quality(openrouter_model: OpenRouterModel, aa_model: ArtificialAnalysisLLM) -> float:
    openrouter_tokens = _tokens(openrouter_model.model_id.split("/", 1)[1])
    aa_tokens = _tokens(aa_model.slug)
    if not openrouter_tokens or not aa_tokens:
        return 0.0
    overlap = len(openrouter_tokens & aa_tokens)
    union = len(openrouter_tokens | aa_tokens)
    jaccard = overlap / union if union else 0.0
    contains_bonus = 0.15 if aa_model.slug.replace("-", "") in openrouter_model.model_id.replace("-", "") else 0.0
    family_bonus = 0.1 if any(token in aa_tokens for token in openrouter_tokens) else 0.0
    return min(jaccard + contains_bonus + family_bonus, 1.0)


def _best_artificial_analysis_match(
    openrouter_model: OpenRouterModel,
    aa_models: tuple[ArtificialAnalysisLLM, ...],
) -> tuple[ArtificialAnalysisLLM | None, float]:
    creator = _creator_slug_from_openrouter(openrouter_model.model_id)
    same_creator = [model for model in aa_models if model.creator_slug == creator]
    best_model: ArtificialAnalysisLLM | None = None
    best_score = 0.0
    for aa_model in same_creator:
        score = _match_quality(openrouter_model, aa_model)
        if score > best_score:
            best_model = aa_model
            best_score = score
    return best_model, best_score


def filter_openrouter_models(
    models: tuple[OpenRouterModel, ...],
    *,
    required_input_modality: str = "",
    creator_allowlist: tuple[str, ...] = (),
    max_prompt_price_1m: float | None = None,
    max_completion_price_1m: float | None = None,
    include_preview: bool = True,
) -> tuple[OpenRouterModel, ...]:
    filtered: list[OpenRouterModel] = []
    for model in models:
        creator_slug = _creator_slug_from_openrouter(model.model_id)
        if creator_allowlist and creator_slug not in creator_allowlist:
            continue
        if required_input_modality and required_input_modality not in model.input_modalities:
            continue
        prompt_price_1m = model.pricing.prompt * 1_000_000
        completion_price_1m = model.pricing.completion * 1_000_000
        if max_prompt_price_1m is not None and prompt_price_1m > max_prompt_price_1m:
            continue
        if max_completion_price_1m is not None and completion_price_1m > max_completion_price_1m:
            continue
        if not include_preview and "preview" in model.model_id:
            continue
        filtered.append(model)
    return tuple(filtered)


def filter_artificial_analysis_models(
    models: tuple[ArtificialAnalysisLLM, ...],
    *,
    creator_allowlist: tuple[str, ...] = (),
    benchmark_key: str = "",
    min_benchmark_score: float | None = None,
    max_blended_price_1m: float | None = None,
    min_output_tokens_per_second: float | None = None,
) -> tuple[ArtificialAnalysisLLM, ...]:
    filtered: list[ArtificialAnalysisLLM] = []
    for model in models:
        if creator_allowlist and model.creator_slug not in creator_allowlist:
            continue
        benchmark_score = model.evaluations.get(benchmark_key) if benchmark_key else None
        if min_benchmark_score is not None and benchmark_score is not None and benchmark_score < min_benchmark_score:
            continue
        if max_blended_price_1m is not None:
            price = model.pricing.price_1m_blended_3_to_1
            if price is not None and price > max_blended_price_1m:
                continue
        if min_output_tokens_per_second is not None:
            speed = model.median_output_tokens_per_second
            if speed is not None and speed < min_output_tokens_per_second:
                continue
        filtered.append(model)
    return tuple(filtered)


def match_openrouter_models_to_artificial_analysis(
    openrouter_models: tuple[OpenRouterModel, ...],
    aa_models: tuple[ArtificialAnalysisLLM, ...],
) -> tuple[ModelCatalogMatch, ...]:
    matches: list[ModelCatalogMatch] = []
    for openrouter_model in openrouter_models:
        creator_slug = _creator_slug_from_openrouter(openrouter_model.model_id)
        aa_model, match_confidence = _best_artificial_analysis_match(openrouter_model, aa_models)
        notes: list[str] = []
        if aa_model is None:
            notes.append("No close Artificial Analysis match; relying on OpenRouter data only.")
        elif match_confidence < 0.45:
            notes.append("Artificial Analysis match is fuzzy; verify with a small empirical probe.")
        if "preview" in openrouter_model.model_id:
            notes.append("Preview model; verify stability against a few real examples before locking it in.")
        matches.append(
            ModelCatalogMatch(
                openrouter_model=openrouter_model,
                artificial_analysis_model=aa_model,
                creator_slug=creator_slug,
                match_confidence=match_confidence,
                notes=tuple(notes),
            )
        )
    return tuple(matches)


def score_model_catalog_match(
    match: ModelCatalogMatch,
    *,
    benchmark_key: str = "artificial_analysis_intelligence_index",
    weights: ModelSelectionWeights | None = None,
) -> ModelSelectionCandidate:
    weights = weights or ModelSelectionWeights()
    aa_model = match.artificial_analysis_model
    benchmark_score = aa_model.evaluations.get(benchmark_key) if aa_model else None
    coding_score = aa_model.evaluations.get("artificial_analysis_coding_index") if aa_model else None
    blended_price = aa_model.pricing.price_1m_blended_3_to_1 if aa_model else None
    speed = aa_model.median_output_tokens_per_second if aa_model else None
    release_date = aa_model.release_date if aa_model else ""
    prompt_price_1m = match.openrouter_model.pricing.prompt * 1_000_000
    completion_price_1m = match.openrouter_model.pricing.completion * 1_000_000
    composite = _composite_score(
        benchmark_score=benchmark_score,
        speed=speed,
        blended_price_1m=blended_price,
        openrouter_prompt_price_1m=prompt_price_1m,
        openrouter_completion_price_1m=completion_price_1m,
        release_date=release_date,
        match_confidence=match.match_confidence,
        weights=weights,
    )
    return ModelSelectionCandidate(
        openrouter_model_id=match.openrouter_model.model_id,
        creator_slug=match.creator_slug,
        modalities=match.openrouter_model.input_modalities,
        openrouter_prompt_price_1m=prompt_price_1m,
        openrouter_completion_price_1m=completion_price_1m,
        artificial_analysis_slug=aa_model.slug if aa_model else "",
        artificial_analysis_name=aa_model.name if aa_model else "",
        benchmark_score=benchmark_score,
        coding_score=coding_score,
        blended_price_1m=blended_price,
        output_tokens_per_second=speed,
        release_date=release_date,
        match_confidence=match.match_confidence,
        composite_score=composite,
        notes=match.notes,
        raw_tokens=tuple(sorted(_tokens(match.openrouter_model.model_id))),
    )


def _composite_score(
    *,
    benchmark_score: float | None,
    speed: float | None,
    blended_price_1m: float | None,
    openrouter_prompt_price_1m: float,
    openrouter_completion_price_1m: float,
    release_date: str,
    match_confidence: float,
    weights: ModelSelectionWeights,
) -> float:
    benchmark_component = (benchmark_score or 0.0) * weights.benchmark
    speed_component = min(speed or 0.0, 300.0) * weights.speed
    blended_price_penalty = min(blended_price_1m or 10.0, 10.0) * weights.blended_price_penalty
    openrouter_price_penalty = (openrouter_prompt_price_1m + openrouter_completion_price_1m) * weights.openrouter_price_penalty
    recency_component = _recency_bonus(release_date) * weights.recency
    confidence_component = match_confidence * weights.match_confidence
    return benchmark_component + speed_component + recency_component + confidence_component - blended_price_penalty - openrouter_price_penalty


def recommend_openrouter_models_for_task(
    *,
    required_input_modality: str = "image",
    benchmark_key: str = "artificial_analysis_intelligence_index",
    max_blended_price_1m: float | None = None,
    min_output_tokens_per_second: float | None = None,
    creator_allowlist: tuple[str, ...] = (),
    available_only: bool = False,
    limit: int = 10,
    weights: ModelSelectionWeights | None = None,
) -> tuple[ModelSelectionCandidate, ...]:
    openrouter_models = filter_openrouter_models(
        list_openrouter_models(available_only=available_only),
        required_input_modality=required_input_modality,
        creator_allowlist=creator_allowlist,
    )
    aa_models = filter_artificial_analysis_models(
        list_artificial_analysis_llms(),
        creator_allowlist=creator_allowlist,
        benchmark_key=benchmark_key,
        max_blended_price_1m=max_blended_price_1m,
        min_output_tokens_per_second=min_output_tokens_per_second,
    )
    candidates = [
        score_model_catalog_match(match, benchmark_key=benchmark_key, weights=weights)
        for match in match_openrouter_models_to_artificial_analysis(openrouter_models, aa_models)
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.composite_score,
            -(item.benchmark_score or 0.0),
            item.blended_price_1m if item.blended_price_1m is not None else math.inf,
            -(item.output_tokens_per_second or 0.0),
        ),
    )
    return tuple(ranked[:limit])


__all__ = [
    "ModelCatalogMatch",
    "ModelSelectionCandidate",
    "ModelSelectionWeights",
    "filter_artificial_analysis_models",
    "filter_openrouter_models",
    "match_openrouter_models_to_artificial_analysis",
    "recommend_openrouter_models_for_task",
    "score_model_catalog_match",
]
