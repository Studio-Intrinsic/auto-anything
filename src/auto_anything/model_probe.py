from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ProbeExample:
    example_id: str
    input_data: Any
    expected_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeExecutionMeta:
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeExampleResult:
    example_id: str
    candidate_id: str
    score: float
    latency_seconds: float
    cost_usd: float
    output: Any = None
    error: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateProbeSummary:
    candidate_id: str
    average_score: float
    average_latency_seconds: float
    average_cost_usd: float
    success_rate: float
    example_results: tuple[ProbeExampleResult, ...] = ()
    notes: tuple[str, ...] = ()


ProbeRunner = Callable[[str, ProbeExample], tuple[Any, ProbeExecutionMeta]]
ProbeScorer = Callable[[Any, ProbeExample], float]


def probe_candidates(
    *,
    candidate_ids: tuple[str, ...],
    examples: tuple[ProbeExample, ...],
    runner: ProbeRunner,
    scorer: ProbeScorer,
) -> tuple[CandidateProbeSummary, ...]:
    summaries: list[CandidateProbeSummary] = []
    for candidate_id in candidate_ids:
        example_results: list[ProbeExampleResult] = []
        for example in examples:
            try:
                output, meta = runner(candidate_id, example)
                score = float(scorer(output, example))
                example_results.append(
                    ProbeExampleResult(
                        example_id=example.example_id,
                        candidate_id=candidate_id,
                        score=score,
                        latency_seconds=float(meta.latency_seconds),
                        cost_usd=float(meta.cost_usd),
                        output=output,
                        notes=tuple(meta.notes),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                example_results.append(
                    ProbeExampleResult(
                        example_id=example.example_id,
                        candidate_id=candidate_id,
                        score=0.0,
                        latency_seconds=0.0,
                        cost_usd=0.0,
                        error=str(exc),
                    )
                )
        success_count = sum(1 for result in example_results if not result.error)
        summaries.append(
            CandidateProbeSummary(
                candidate_id=candidate_id,
                average_score=sum(result.score for result in example_results) / len(example_results) if example_results else 0.0,
                average_latency_seconds=(
                    sum(result.latency_seconds for result in example_results) / len(example_results) if example_results else 0.0
                ),
                average_cost_usd=sum(result.cost_usd for result in example_results) / len(example_results) if example_results else 0.0,
                success_rate=(success_count / len(example_results)) if example_results else 0.0,
                example_results=tuple(example_results),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                -item.average_score,
                item.average_cost_usd,
                item.average_latency_seconds,
                -item.success_rate,
            ),
        )
    )


def write_probe_report(path: Path, summaries: tuple[CandidateProbeSummary, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            **asdict(summary),
            "example_results": [asdict(result) for result in summary.example_results],
        }
        for summary in summaries
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "CandidateProbeSummary",
    "ProbeExample",
    "ProbeExampleResult",
    "ProbeExecutionMeta",
    "probe_candidates",
    "write_probe_report",
]
