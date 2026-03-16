from __future__ import annotations

from .models import (
    AcceptanceDecision,
    CounterbalanceMode,
    CounterbalanceReport,
    CritiqueSeverity,
    EvaluationReport,
    ExperimentRecord,
    ObjectiveSignal,
    SignalDirection,
    TaskCharter,
)


def _signal_value(report: EvaluationReport, signal_name: str) -> float:
    try:
        return report.signal_map()[signal_name].value
    except KeyError as exc:
        raise KeyError(f"Report for candidate {report.candidate_id} is missing signal '{signal_name}'.") from exc


def _improvement(signal: ObjectiveSignal, baseline_value: float, candidate_value: float) -> float:
    """Compute normalized improvement as a fraction of the baseline value.

    Raw deltas are scale-dependent: a 0.01 change in a 0-1 signal is huge,
    while a 1-unit change in a 0-500 signal is trivial. Normalizing by the
    baseline magnitude puts all signals on a comparable fractional scale so
    weights work as intended.
    """
    if signal.direction in {SignalDirection.MAXIMIZE, SignalDirection.SATISFY}:
        raw = candidate_value - baseline_value
    else:
        raw = baseline_value - candidate_value
    # Normalize by baseline magnitude (use 1.0 floor to avoid division by zero
    # and to keep signals near zero on an absolute scale)
    scale = max(abs(baseline_value), 1.0)
    return raw / scale


def _passes_hard_gate(signal: ObjectiveSignal, candidate_value: float) -> bool:
    if not signal.hard_gate:
        return True
    assert signal.target_value is not None
    if signal.direction in {SignalDirection.MAXIMIZE, SignalDirection.SATISFY}:
        return candidate_value >= signal.target_value
    return candidate_value <= signal.target_value


_SEVERITY_ORDER = {
    CritiqueSeverity.LOW: 0,
    CritiqueSeverity.MEDIUM: 1,
    CritiqueSeverity.HIGH: 2,
    CritiqueSeverity.CRITICAL: 3,
}


def _is_blocking_finding(severity: CritiqueSeverity, threshold: CritiqueSeverity) -> bool:
    return _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[threshold]


def _utility_signal_names(charter: TaskCharter, focus_subsystems: tuple[str, ...]) -> set[str]:
    if not focus_subsystems:
        return {signal.name for signal in charter.evaluation_plan.signals}
    subsystem_map = {
        subsystem.subsystem_id: subsystem
        for subsystem in charter.search_surface.subsystems
    }
    unknown = [subsystem_id for subsystem_id in focus_subsystems if subsystem_id not in subsystem_map]
    if unknown:
        raise ValueError(f"Unknown subsystem ids for focused iteration: {', '.join(unknown)}")
    selected: set[str] = set()
    for subsystem_id in focus_subsystems:
        subsystem = subsystem_map[subsystem_id]
        selected.update(subsystem.primary_signals)
        selected.update(subsystem.guardrail_signals)
    return selected or {signal.name for signal in charter.evaluation_plan.signals}


class ExperimentEngine:
    def decide(
        self,
        *,
        charter: TaskCharter,
        baseline_report: EvaluationReport,
        candidate_report: EvaluationReport,
        counterbalance_report: CounterbalanceReport | None = None,
        focus_subsystems: tuple[str, ...] = (),
    ) -> AcceptanceDecision:
        blocking_signals: list[str] = []
        blocking_findings: list[str] = []
        reasons: list[str] = []
        utility_gain = 0.0
        resolved_focus = focus_subsystems or charter.focus_subsystems
        utility_signals = _utility_signal_names(charter, resolved_focus)

        for signal in charter.evaluation_plan.signals:
            baseline_value = _signal_value(baseline_report, signal.name)
            candidate_value = _signal_value(candidate_report, signal.name)

            if not _passes_hard_gate(signal, candidate_value):
                blocking_signals.append(signal.name)
                reasons.append(f"hard_gate_failed:{signal.name}")
                continue

            delta = _improvement(signal, baseline_value, candidate_value)
            if signal.max_regression is not None and delta < -signal.max_regression:
                blocking_signals.append(signal.name)
                reasons.append(f"regression_exceeded:{signal.name}")
                continue

            if signal.name in utility_signals:
                utility_gain += signal.weight * delta

        counterbalance = charter.counterbalance
        if counterbalance.mode != CounterbalanceMode.NONE:
            if counterbalance.required and counterbalance_report is None:
                reasons.append("counterbalance_missing")
                return AcceptanceDecision(
                    accepted=False,
                    utility_gain=utility_gain,
                    focus_subsystems=resolved_focus,
                    reasons=tuple(reasons),
                    blocking_signals=tuple(blocking_signals),
                    blocking_findings=("counterbalance_missing",),
                )

            if counterbalance_report is not None:
                for finding in counterbalance_report.findings:
                    if _is_blocking_finding(finding.severity, counterbalance.block_on_severity):
                        blocking_findings.append(finding.finding_id or finding.summary)
                        reasons.append(f"counterbalance_blocked:{finding.finding_id or finding.summary}")
                    else:
                        utility_gain -= counterbalance.penalty_per_finding

        accepted = not blocking_signals and not blocking_findings and utility_gain > 0.0
        if accepted:
            reasons.append("utility_improved")
        elif not blocking_signals and not blocking_findings:
            reasons.append("no_positive_utility_gain")

        return AcceptanceDecision(
            accepted=accepted,
            utility_gain=utility_gain,
            focus_subsystems=resolved_focus,
            reasons=tuple(reasons),
            blocking_signals=tuple(blocking_signals),
            blocking_findings=tuple(blocking_findings),
        )

    def record(
        self,
        *,
        charter: TaskCharter,
        baseline_report: EvaluationReport,
        candidate_report: EvaluationReport,
        counterbalance_report: CounterbalanceReport | None = None,
        focus_subsystems: tuple[str, ...] = (),
        notes: tuple[str, ...] = (),
    ) -> ExperimentRecord:
        decision = self.decide(
            charter=charter,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            counterbalance_report=counterbalance_report,
            focus_subsystems=focus_subsystems,
        )
        return ExperimentRecord(
            candidate_id=candidate_report.candidate_id,
            baseline_candidate_id=baseline_report.candidate_id,
            decision=decision,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            counterbalance_report=counterbalance_report,
            focus_subsystems=decision.focus_subsystems,
            notes=notes,
        )
