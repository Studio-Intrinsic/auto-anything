from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SignalKind(str, Enum):
    SCALAR = "scalar"
    RUBRIC = "rubric"
    BINARY = "binary"


class SignalDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    SATISFY = "satisfy"


class ConstraintLevel(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class AgentRole(str, Enum):
    BUILDER = "builder"
    CRITIC = "critic"
    JUDGE = "judge"


class CounterbalanceMode(str, Enum):
    NONE = "none"
    SELF_CRITIC = "self_critic"


class CritiqueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DataAsset:
    name: str
    kind: str
    location: str
    role: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Constraint:
    statement: str
    level: ConstraintLevel = ConstraintLevel.HARD
    rationale: str = ""


@dataclass(frozen=True)
class ObjectiveSignal:
    name: str
    kind: SignalKind
    direction: SignalDirection
    description: str
    weight: float = 1.0
    hard_gate: bool = False
    target_value: float | None = None
    max_regression: float | None = None
    source: str = "user"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ObjectiveSignal.name must not be empty.")
        if self.weight < 0:
            raise ValueError("ObjectiveSignal.weight must be non-negative.")
        if self.hard_gate and self.target_value is None:
            raise ValueError("Hard-gated signals must define target_value.")
        if self.max_regression is not None and self.max_regression < 0:
            raise ValueError("ObjectiveSignal.max_regression must be non-negative when provided.")


@dataclass(frozen=True)
class SubsystemSpec:
    subsystem_id: str
    summary: str
    owned_paths: tuple[str, ...] = ()
    primary_signals: tuple[str, ...] = ()
    guardrail_signals: tuple[str, ...] = ()
    decomposition_hints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.subsystem_id.strip():
            raise ValueError("SubsystemSpec.subsystem_id must not be empty.")
        if not self.summary.strip():
            raise ValueError("SubsystemSpec.summary must not be empty.")


@dataclass(frozen=True)
class SearchSurface:
    mutable_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    entrypoints: tuple[str, ...] = ()
    subsystems: tuple[SubsystemSpec, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunCommand:
    name: str
    command: tuple[str, ...]
    working_dir: str = "."
    timeout_seconds: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("RunCommand.name must not be empty.")
        if not self.command:
            raise ValueError("RunCommand.command must not be empty.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("RunCommand.timeout_seconds must be positive when provided.")


@dataclass(frozen=True)
class WorkspaceLayout:
    root_dir: str = "."
    candidate_dir: str = "candidate"
    evaluator_dir: str = "eval"
    artifacts_dir: str = "artifacts"
    replay_dir: str = "replay"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RolePass:
    role: AgentRole
    goal: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentLoopSpec:
    passes: tuple[RolePass, ...] = (
        RolePass(role=AgentRole.BUILDER, goal="Improve the candidate pipeline against the charter."),
        RolePass(role=AgentRole.CRITIC, goal="Try to break the candidate, find gaming, and expose weak spots."),
        RolePass(role=AgentRole.JUDGE, goal="Decide whether the candidate improved holistically."),
    )
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("AgentLoopSpec.passes must not be empty.")


@dataclass(frozen=True)
class AgentRuntimeConfig:
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    default_model: str = ""
    allowed_models: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterbalanceConfig:
    mode: CounterbalanceMode = CounterbalanceMode.SELF_CRITIC
    required: bool = True
    block_on_severity: CritiqueSeverity = CritiqueSeverity.HIGH
    penalty_per_finding: float = 0.1
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationPlan:
    signals: tuple[ObjectiveSignal, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.signals:
            raise ValueError("EvaluationPlan.signals must contain at least one signal.")
        seen: set[str] = set()
        for signal in self.signals:
            if signal.name in seen:
                raise ValueError(f"Duplicate signal name: {signal.name}")
            seen.add(signal.name)


@dataclass(frozen=True)
class SkillContribution:
    skill_id: str
    summary: str = ""
    suggested_signals: tuple[ObjectiveSignal, ...] = ()
    suggested_constraints: tuple[Constraint, ...] = ()
    suggested_mutable_paths: tuple[str, ...] = ()
    suggested_protected_paths: tuple[str, ...] = ()
    suggested_subsystems: tuple[SubsystemSpec, ...] = ()
    decomposition_hints: tuple[str, ...] = ()
    evaluation_notes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectiveBrief:
    title: str
    objective_statement: str
    data_assets: tuple[DataAsset, ...] = ()
    anti_goals: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    explicit_signals: tuple[ObjectiveSignal, ...] = ()
    mutable_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    entrypoints: tuple[str, ...] = ()
    subsystems: tuple[SubsystemSpec, ...] = ()
    focus_subsystems: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    disallowed_models: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    run_commands: tuple[RunCommand, ...] = ()
    workspace_layout: WorkspaceLayout = field(default_factory=WorkspaceLayout)
    agent_runtime: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    agent_loop: AgentLoopSpec = field(default_factory=AgentLoopSpec)
    counterbalance: CounterbalanceConfig = field(default_factory=CounterbalanceConfig)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("ObjectiveBrief.title must not be empty.")
        if not self.objective_statement.strip():
            raise ValueError("ObjectiveBrief.objective_statement must not be empty.")


@dataclass(frozen=True)
class TaskCharter:
    charter_id: str
    title: str
    objective_statement: str
    data_assets: tuple[DataAsset, ...]
    hard_constraints: tuple[Constraint, ...]
    soft_constraints: tuple[Constraint, ...]
    anti_goals: tuple[str, ...]
    evaluation_plan: EvaluationPlan
    search_surface: SearchSurface
    workspace_layout: WorkspaceLayout
    focus_subsystems: tuple[str, ...] = ()
    agent_runtime: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    agent_loop: AgentLoopSpec = field(default_factory=AgentLoopSpec)
    counterbalance: CounterbalanceConfig = field(default_factory=CounterbalanceConfig)
    run_commands: tuple[RunCommand, ...] = ()
    applied_skills: tuple[str, ...] = ()
    decomposition_hints: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    disallowed_models: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationArtifact:
    name: str
    path: Path | None = None
    uri: str = ""
    description: str = ""


@dataclass(frozen=True)
class SignalResult:
    name: str
    value: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    candidate_id: str
    signals: tuple[SignalResult, ...]
    artifacts: tuple[EvaluationArtifact, ...] = ()
    notes: tuple[str, ...] = ()

    def signal_map(self) -> dict[str, SignalResult]:
        return {signal.name: signal for signal in self.signals}


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_id: str
    root_dir: Path
    mutable_paths: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class IterationStep:
    summary: str
    role: AgentRole = AgentRole.BUILDER
    touched_paths: tuple[str, ...] = ()
    executed_commands: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CritiqueFinding:
    finding_id: str
    summary: str
    severity: CritiqueSeverity = CritiqueSeverity.MEDIUM
    rationale: str = ""
    suggested_probes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CounterbalanceReport:
    mode: CounterbalanceMode
    findings: tuple[CritiqueFinding, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    utility_gain: float
    focus_subsystems: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    blocking_signals: tuple[str, ...] = ()
    blocking_findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentRecord:
    candidate_id: str
    baseline_candidate_id: str
    decision: AcceptanceDecision
    baseline_report: EvaluationReport
    candidate_report: EvaluationReport
    counterbalance_report: CounterbalanceReport | None = None
    candidate_snapshot: CandidateSnapshot | None = None
    iteration_steps: tuple[IterationStep, ...] = ()
    focus_subsystems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
