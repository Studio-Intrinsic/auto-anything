"""Diagnose failure modes and maintain a hypothesis ledger.

After each eval, this module analyzes per-doc results to surface
what's actually going wrong, cross-references with past experiment
hypotheses, and maintains an open/closed hypothesis ledger so the
agent can see what's been tried and what remains untested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .history import load_experiment_history


def _load_eval_summary(task_root: Path) -> dict:
    path = task_root / "artifacts" / "eval_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hypotheses(task_root: Path) -> list[dict]:
    path = task_root / "artifacts" / "hypotheses.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_hypotheses(task_root: Path, hypotheses: list[dict]) -> None:
    path = task_root / "artifacts" / "hypotheses.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hypotheses, indent=2, sort_keys=True), encoding="utf-8")


def _analyze_doc_results(summary: dict, primary_signal: str) -> dict:
    """Analyze per-doc results to find failure patterns."""
    doc_results = summary.get("doc_results", [])
    if not doc_results:
        return {"failure_modes": [], "doc_count": 0, "note": "No per-doc results in eval_summary. Add doc_results to your evaluator for failure analysis."}

    n = len(doc_results)
    # Find the primary signal per doc (look for common names)
    signal_key = primary_signal
    # Try to find matching key in doc results
    sample = doc_results[0]
    if signal_key not in sample:
        # Try common alternatives
        for alt in ("f1", "score", "accuracy", "semantic", "compliance"):
            if alt in sample:
                signal_key = alt
                break

    if signal_key not in doc_results[0]:
        return {"failure_modes": [], "doc_count": n, "note": f"Could not find per-doc signal '{primary_signal}' in doc_results. Keys available: {list(sample.keys())}"}

    # Sort docs by performance
    sorted_docs = sorted(doc_results, key=lambda d: d.get(signal_key, 0))

    # Identify tiers
    failing = [d for d in sorted_docs if d.get(signal_key, 0) < 0.5]
    weak = [d for d in sorted_docs if 0.5 <= d.get(signal_key, 0) < 0.8]
    passing = [d for d in sorted_docs if d.get(signal_key, 0) >= 0.8]
    perfect = [d for d in sorted_docs if d.get(signal_key, 0) >= 0.999]

    # Build failure mode observations from doc-level data
    failure_modes: list[dict] = []

    if failing:
        failure_modes.append({
            "mode": "hard_failures",
            "description": f"{len(failing)}/{n} docs score below 0.5 on {signal_key}",
            "impact": "high",
            "docs": [d.get("doc", "?") for d in failing],
            "detail": {d.get("doc", "?"): {k: v for k, v in d.items() if k != "doc"} for d in failing},
        })

    if weak:
        failure_modes.append({
            "mode": "weak_results",
            "description": f"{len(weak)}/{n} docs score 0.5-0.8 on {signal_key}",
            "impact": "medium",
            "docs": [d.get("doc", "?") for d in weak],
        })

    # Check for precision vs recall imbalance if available
    prec_docs = [d for d in doc_results if "precision" in d and "recall" in d]
    if prec_docs:
        avg_prec = sum(d["precision"] for d in prec_docs) / len(prec_docs)
        avg_rec = sum(d["recall"] for d in prec_docs) / len(prec_docs)
        if avg_prec < avg_rec - 0.1:
            failure_modes.append({
                "mode": "precision_deficit",
                "description": f"Avg precision ({avg_prec:.3f}) much lower than recall ({avg_rec:.3f}) — model is predicting extra/wrong fields",
                "impact": "medium",
                "suggestion": "Tighten the output schema or add post-filtering to remove hallucinated fields",
            })
        elif avg_rec < avg_prec - 0.1:
            failure_modes.append({
                "mode": "recall_deficit",
                "description": f"Avg recall ({avg_rec:.3f}) much lower than precision ({avg_prec:.3f}) — model is missing fields",
                "impact": "medium",
                "suggestion": "Expand the prompt to cover more field types, or try a more capable model",
            })

    # Check for error patterns
    error_docs = [d for d in doc_results if d.get("error")]
    if error_docs:
        failure_modes.append({
            "mode": "runtime_errors",
            "description": f"{len(error_docs)}/{n} docs had runtime errors",
            "impact": "high",
            "docs": [d.get("doc", "?") for d in error_docs],
            "errors": [d.get("error", "") for d in error_docs],
        })

    return {
        "doc_count": n,
        "signal_key": signal_key,
        "tiers": {
            "perfect": len(perfect),
            "passing": len(passing),
            "weak": len(weak),
            "failing": len(failing),
        },
        "worst_docs": [{"doc": d.get("doc", "?"), signal_key: d.get(signal_key, 0)} for d in sorted_docs[:5]],
        "best_docs": [{"doc": d.get("doc", "?"), signal_key: d.get(signal_key, 0)} for d in sorted_docs[-3:]],
        "failure_modes": failure_modes,
    }


def _cross_reference_hypotheses(
    hypotheses: list[dict],
    history: list[dict[str, Any]],
) -> list[dict]:
    """Update hypothesis statuses based on experiment outcomes."""
    for hyp in hypotheses:
        if hyp.get("status") == "closed":
            continue
        # Check if any experiment targeted this hypothesis
        exp_id = hyp.get("tested_by")
        if exp_id:
            matching = [e for e in history if e.get("experiment_id") == exp_id]
            if matching:
                entry = matching[0]
                hyp["tested_accepted"] = entry.get("accepted", False)
                hyp["status"] = "closed"
                hyp["outcome"] = "confirmed" if entry.get("accepted") else "rejected"
    return hypotheses


def diagnose(task_root: Path, primary_signal: str = "") -> dict:
    """Run full diagnosis: failure analysis + hypothesis ledger update."""
    task_root = task_root.expanduser().resolve()
    summary = _load_eval_summary(task_root)
    history = load_experiment_history(task_root)
    hypotheses = _load_hypotheses(task_root)

    # Determine primary signal
    if not primary_signal:
        charter_path = task_root / "task_charter.json"
        if charter_path.is_file():
            charter = json.loads(charter_path.read_text(encoding="utf-8"))
            signals = charter.get("evaluation_plan", {}).get("signals", [])
            if signals:
                # Pick highest-weight signal
                primary_signal = max(signals, key=lambda s: s.get("weight", 0)).get("name", "")

    analysis = _analyze_doc_results(summary, primary_signal)
    hypotheses = _cross_reference_hypotheses(hypotheses, history)

    # Build hypothesis summary
    open_hyps = [h for h in hypotheses if h.get("status") != "closed"]
    closed_hyps = [h for h in hypotheses if h.get("status") == "closed"]

    # Check experiment history for hypothesis patterns
    experiment_patterns: list[str] = []
    if history:
        subsystem_stats: dict[str, dict] = {}
        for entry in history:
            for sub in entry.get("focus_subsystems", []):
                stats = subsystem_stats.setdefault(sub, {"total": 0, "accepted": 0})
                stats["total"] += 1
                if entry.get("accepted"):
                    stats["accepted"] += 1
        for sub, stats in subsystem_stats.items():
            rate = stats["accepted"] / stats["total"] if stats["total"] else 0
            if stats["total"] >= 2 and rate == 0:
                experiment_patterns.append(f"Subsystem '{sub}' has {stats['total']} experiments with 0 acceptances — consider a different approach or different subsystem focus")
            elif rate >= 0.5:
                experiment_patterns.append(f"Subsystem '{sub}' is productive ({stats['accepted']}/{stats['total']} accepted)")

    _save_hypotheses(task_root, hypotheses)

    return {
        "analysis": analysis,
        "hypotheses": {
            "open": open_hyps,
            "closed": closed_hyps,
        },
        "experiment_patterns": experiment_patterns,
    }


def add_hypothesis(
    task_root: Path,
    hypothesis: str,
    targets: str = "",
    failure_mode: str = "",
) -> dict:
    """Add a new hypothesis to the ledger."""
    task_root = task_root.expanduser().resolve()
    hypotheses = _load_hypotheses(task_root)
    hyp_id = f"hyp-{len(hypotheses):04d}"
    entry = {
        "id": hyp_id,
        "hypothesis": hypothesis,
        "targets": targets,
        "failure_mode": failure_mode,
        "status": "open",
        "tested_by": None,
        "outcome": None,
    }
    hypotheses.append(entry)
    _save_hypotheses(task_root, hypotheses)
    return entry


def link_hypothesis_to_experiment(
    task_root: Path,
    hypothesis_id: str,
    experiment_id: str,
) -> None:
    """Mark a hypothesis as being tested by a specific experiment."""
    task_root = task_root.expanduser().resolve()
    hypotheses = _load_hypotheses(task_root)
    for hyp in hypotheses:
        if hyp["id"] == hypothesis_id:
            hyp["tested_by"] = experiment_id
            break
    _save_hypotheses(task_root, hypotheses)
