from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.artificial_analysis_api import (
    ArtificialAnalysisLLM,
    ArtificialAnalysisPricing,
    parse_artificial_analysis_llm,
    shortlist_artificial_analysis_llms,
)
from auto_anything.model_selection import (
    ModelSelectionWeights,
    filter_artificial_analysis_models,
    filter_openrouter_models,
    match_openrouter_models_to_artificial_analysis,
    recommend_openrouter_models_for_task,
    score_model_catalog_match,
)
from auto_anything.openrouter_api import (
    OpenRouterModel,
    OpenRouterPricing,
    estimate_openrouter_cost,
    extract_openrouter_usage,
    parse_openrouter_model,
)


class OpenRouterApiTests(unittest.TestCase):
    def test_parse_openrouter_model(self) -> None:
        payload = {
            "id": "x-ai/grok-4.1-fast",
            "canonical_slug": "x-ai/grok-4.1-fast-20250301",
            "name": "xAI: Grok 4.1 Fast",
            "description": "Fast model",
            "created": 1773325367,
            "context_length": 2000000,
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "pricing": {
                "prompt": "0.000002",
                "completion": "0.000006",
                "image": "0.001",
                "request": "0.0001",
                "input_cache_read": "0.0000002",
            },
            "supported_parameters": ["max_tokens", "temperature"],
        }
        model = parse_openrouter_model(payload)
        self.assertEqual(model.model_id, "x-ai/grok-4.1-fast")
        self.assertEqual(model.input_modalities, ("text", "image"))
        self.assertAlmostEqual(model.pricing.prompt, 0.000002)

    def test_extract_openrouter_usage_prefers_exact_cost(self) -> None:
        payload = {
            "id": "gen-123",
            "model": "x-ai/grok-4.1-fast",
            "usage": {
                "prompt_tokens": 163,
                "completion_tokens": 333,
                "total_tokens": 496,
                "cost": 0.0001746855,
                "prompt_tokens_details": {
                    "cached_tokens": 151,
                    "cache_write_tokens": 0,
                    "audio_tokens": 0,
                    "video_tokens": 0,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 332,
                    "image_tokens": 0,
                    "audio_tokens": 0,
                },
            },
        }
        usage = extract_openrouter_usage(payload)
        self.assertEqual(usage.generation_id, "gen-123")
        self.assertEqual(usage.reasoning_tokens, 332)
        self.assertTrue(usage.used_exact_cost)
        self.assertAlmostEqual(usage.cost_usd, 0.0001746855)

    def test_extract_openrouter_usage_estimates_when_cost_missing(self) -> None:
        payload = {
            "model": "x-ai/grok-4.1-fast",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 7, "image_tokens": 2},
            },
        }
        model = OpenRouterModel(
            model_id="x-ai/grok-4.1-fast",
            pricing=OpenRouterPricing(
                prompt=0.000002,
                completion=0.000006,
                image=0.001,
                request=0.0001,
                internal_reasoning=0.000001,
                input_cache_read=0.0000002,
                input_cache_write=0.0000004,
            ),
        )
        usage = extract_openrouter_usage(payload, model=model)
        expected = estimate_openrouter_cost(usage=usage, pricing=model.pricing)
        self.assertFalse(usage.used_exact_cost)
        self.assertAlmostEqual(usage.estimated_cost_usd, expected)


class ArtificialAnalysisApiTests(unittest.TestCase):
    def test_parse_artificial_analysis_llm(self) -> None:
        payload = {
            "id": "2dad8957-4c16-4e74-bf2d-8b21514e0ae9",
            "name": "o3-mini",
            "slug": "o3-mini",
            "model_creator": {
                "id": "openai-id",
                "name": "OpenAI",
                "slug": "openai",
            },
            "evaluations": {
                "artificial_analysis_intelligence_index": 62.9,
                "artificial_analysis_coding_index": 55.8,
            },
            "pricing": {
                "price_1m_blended_3_to_1": 1.925,
                "price_1m_input_tokens": 1.1,
                "price_1m_output_tokens": 4.4,
            },
            "median_output_tokens_per_second": 153.831,
            "median_time_to_first_token_seconds": 14.939,
            "median_time_to_first_answer_token": 14.939,
        }
        model = parse_artificial_analysis_llm(payload)
        self.assertEqual(model.slug, "o3-mini")
        self.assertEqual(model.creator_slug, "openai")
        self.assertAlmostEqual(model.pricing.price_1m_blended_3_to_1 or 0.0, 1.925)

    def test_shortlist_artificial_analysis_llms(self) -> None:
        models = (
            ArtificialAnalysisLLM(
                model_id="a",
                name="A",
                slug="a",
                evaluations={"artificial_analysis_intelligence_index": 50.0},
                pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=3.0),
                median_output_tokens_per_second=100.0,
            ),
            ArtificialAnalysisLLM(
                model_id="b",
                name="B",
                slug="b",
                evaluations={"artificial_analysis_intelligence_index": 55.0},
                pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=2.0),
                median_output_tokens_per_second=120.0,
            ),
        )
        shortlisted = shortlist_artificial_analysis_llms(
            models,
            max_blended_price_1m=2.5,
            min_output_tokens_per_second=110.0,
            limit=5,
        )
        self.assertEqual(tuple(model.slug for model in shortlisted), ("b",))


class ModelSelectionTests(unittest.TestCase):
    def test_filter_openrouter_models(self) -> None:
        models = (
            OpenRouterModel(model_id="google/gemini-2.5-flash-lite", input_modalities=("text", "image"), pricing=OpenRouterPricing(prompt=1e-07, completion=4e-07)),
            OpenRouterModel(model_id="openai/gpt-5-mini", input_modalities=("text",), pricing=OpenRouterPricing(prompt=2.5e-07, completion=2e-06)),
        )
        filtered = filter_openrouter_models(models, required_input_modality="image", creator_allowlist=("google",))
        self.assertEqual(tuple(model.model_id for model in filtered), ("google/gemini-2.5-flash-lite",))

    def test_filter_artificial_analysis_models(self) -> None:
        models = (
            ArtificialAnalysisLLM(
                model_id="g1",
                name="Gemini 3.1 Flash Lite Preview",
                slug="gemini-3-1-flash-lite-preview",
                creator_slug="google",
                evaluations={"artificial_analysis_intelligence_index": 33.5},
                pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=0.563),
                median_output_tokens_per_second=260.334,
            ),
            ArtificialAnalysisLLM(
                model_id="g2",
                name="Gemini 3.1 Pro Preview",
                slug="gemini-3-1-pro-preview",
                creator_slug="google",
                evaluations={"artificial_analysis_intelligence_index": 57.2},
                pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=4.5),
                median_output_tokens_per_second=121.375,
            ),
        )
        filtered = filter_artificial_analysis_models(
            models,
            creator_allowlist=("google",),
            benchmark_key="artificial_analysis_intelligence_index",
            max_blended_price_1m=1.0,
        )
        self.assertEqual(tuple(model.slug for model in filtered), ("gemini-3-1-flash-lite-preview",))

    def test_match_and_score_model_catalogs(self) -> None:
        openrouter_models = (
            OpenRouterModel(
                model_id="google/gemini-2.5-flash-lite",
                input_modalities=("text", "image"),
                pricing=OpenRouterPricing(prompt=1e-07, completion=4e-07),
            ),
        )
        aa_models = (
            ArtificialAnalysisLLM(
                model_id="g1",
                name="Gemini 3.1 Flash Lite Preview",
                slug="gemini-3-1-flash-lite-preview",
                creator_slug="google",
                evaluations={"artificial_analysis_intelligence_index": 33.5, "artificial_analysis_coding_index": 30.0},
                pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=0.563),
                median_output_tokens_per_second=260.334,
                release_date="2025-10",
            ),
        )
        matches = match_openrouter_models_to_artificial_analysis(openrouter_models, aa_models)
        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0].match_confidence, 0.2)
        candidate = score_model_catalog_match(matches[0], weights=ModelSelectionWeights())
        self.assertEqual(candidate.openrouter_model_id, "google/gemini-2.5-flash-lite")
        self.assertEqual(candidate.artificial_analysis_slug, "gemini-3-1-flash-lite-preview")

    def test_recommend_function_is_patchable_and_ranked(self) -> None:
        import auto_anything.model_selection as model_selection

        original_list_openrouter_models = model_selection.list_openrouter_models
        original_list_artificial_analysis_llms = model_selection.list_artificial_analysis_llms
        try:
            model_selection.list_openrouter_models = lambda available_only=False: (
                OpenRouterModel(
                    model_id="google/gemini-2.5-flash-lite",
                    input_modalities=("text", "image"),
                    pricing=OpenRouterPricing(prompt=1e-07, completion=4e-07),
                ),
                OpenRouterModel(
                    model_id="x-ai/grok-4-fast",
                    input_modalities=("text", "image"),
                    pricing=OpenRouterPricing(prompt=2e-07, completion=5e-07),
                ),
            )
            model_selection.list_artificial_analysis_llms = lambda: (
                ArtificialAnalysisLLM(
                    model_id="g1",
                    name="Gemini 3.1 Flash Lite Preview",
                    slug="gemini-3-1-flash-lite-preview",
                    creator_slug="google",
                    evaluations={"artificial_analysis_intelligence_index": 33.5},
                    pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=0.563),
                    median_output_tokens_per_second=260.334,
                    release_date="2025-10",
                ),
                ArtificialAnalysisLLM(
                    model_id="x1",
                    name="Grok 4 Fast",
                    slug="grok-4-fast",
                    creator_slug="xai",
                    evaluations={"artificial_analysis_intelligence_index": 23.1},
                    pricing=ArtificialAnalysisPricing(price_1m_blended_3_to_1=0.275),
                    median_output_tokens_per_second=215.025,
                    release_date="2025-09",
                ),
            )
            ranked = recommend_openrouter_models_for_task(required_input_modality="image", limit=2)
            self.assertEqual(ranked[0].openrouter_model_id, "google/gemini-2.5-flash-lite")
        finally:
            model_selection.list_openrouter_models = original_list_openrouter_models
            model_selection.list_artificial_analysis_llms = original_list_artificial_analysis_llms


if __name__ == "__main__":
    unittest.main()
