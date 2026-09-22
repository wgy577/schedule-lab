"""Solver Teacher Trace V1 -- trace construction and configuration.

A :class:`SolverTeacherTrace` answers a strictly scoped question: *given that
the free-global CP-SAT result beat the frozen-local result of the same seed
proposal, what did the solver additionally do?*  It deliberately separates:

* **Attribution experience** -- ``local_delta_cmax``: what the seed proposal
  itself achieves inside its local closure (the only causal label of the
  proposal, owned by the frozen-local evaluator).
* **Solver teacher experience** -- ``additional_edits``: the extra changes the
  global solver made on top of the local repair (this module's subject).  The
  global delta is **never** attributed to the seed proposal.
* **Delayed / enabling experience** -- only established later, by deterministic
  replay/ablation (:mod:`solver_trace_replay_v1`), never by the raw -49.

All traces default to ``training_eligible=False``; this round extracts and
verifies teacher behaviour only.  ``identified`` stays ``False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..ir import Problem, Schedule
from ..validation import schedule_hash
from .solver_schedule_diff_v1 import (
    TeacherEditSet,
    changed_operation_ids,
)

TEACHER_TRACE_SCHEMA = "solver_teacher_trace_v1"

__all__ = [
    "SolverTeacherTraceConfig",
    "SolverTeacherTrace",
    "load_solver_teacher_trace_config",
    "build_solver_teacher_trace",
    "TEACHER_TRACE_SCHEMA",
]


def _repository_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "hyperparameters.yaml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SolverTeacherTraceConfig:
    """The ``solver_teacher_trace`` hyperparameter section."""

    enabled: bool = True
    llm_provider: str = "deepseek"
    model: str = ""
    max_teacher_edits: int = 12
    max_replay_steps: int = 8
    max_replay_groups: int = 6
    dependency_depth: int = 3
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    enable_llm_analysis: bool = True
    enable_replay: bool = True
    timeout_seconds: float = 120.0
    max_attempts: int = 2
    thinking_mode: str = "disabled"
    solver_time: float = 5.0
    seed: int = 0

    def validate(self) -> None:
        if not self.llm_provider.strip():
            raise ValueError("solver_teacher_trace.llm_provider must be non-empty")
        if self.llm_provider.strip().lower() != "deepseek":
            raise ValueError("solver teacher trace requires llm_provider=deepseek")
        if self.max_teacher_edits < 1:
            raise ValueError("solver_teacher_trace.max_teacher_edits must be >= 1")
        if self.max_replay_steps < 1:
            raise ValueError("solver_teacher_trace.max_replay_steps must be >= 1")
        if self.max_replay_groups < 1:
            raise ValueError("solver_teacher_trace.max_replay_groups must be >= 1")
        if self.dependency_depth < 1:
            raise ValueError("solver_teacher_trace.dependency_depth must be >= 1")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("solver_teacher_trace.llm_temperature must be in [0,2]")
        if self.llm_max_tokens < 128:
            raise ValueError("solver_teacher_trace.llm_max_tokens must be >= 128")
        if self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("invalid solver_teacher_trace provider budget")
        if self.thinking_mode not in {"enabled", "disabled", "adaptive"}:
            raise ValueError("invalid solver_teacher_trace.thinking_mode")
        if self.solver_time <= 0:
            raise ValueError("solver_teacher_trace.solver_time must be > 0")


def load_solver_teacher_trace_config(
    path: str | Path | None = None,
) -> SolverTeacherTraceConfig:
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or (_repository_root() / "configs" / "hyperparameters.yaml")
    ).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    section = payload.get("solver_teacher_trace")
    if not isinstance(section, Mapping):
        raise ValueError("missing solver_teacher_trace hyperparameters")
    config = SolverTeacherTraceConfig(
        enabled=bool(section.get("enabled", True)),
        llm_provider=str(section.get("llm_provider", "deepseek")),
        model=str(section.get("model") or ""),
        max_teacher_edits=int(section.get("max_teacher_edits", 12)),
        max_replay_steps=int(section.get("max_replay_steps", 8)),
        max_replay_groups=int(section.get("max_replay_groups", 6)),
        dependency_depth=int(section.get("dependency_depth", 3)),
        llm_temperature=float(section.get("llm_temperature", 0.2)),
        llm_max_tokens=int(section.get("llm_max_tokens", 4096)),
        enable_llm_analysis=bool(section.get("enable_llm_analysis", True)),
        enable_replay=bool(section.get("enable_replay", True)),
        timeout_seconds=float(section.get("timeout_seconds", 120.0)),
        max_attempts=int(section.get("max_attempts", 2)),
        thinking_mode=str(section.get("thinking_mode", "disabled")),
        solver_time=float(section.get("solver_time", 5.0)),
        seed=int(section.get("seed", 0)),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class SolverTeacherTrace:
    """One seed proposal's solver-teacher trace (local vs global)."""

    trace_id: str
    instance_id: str
    proposal_id: str
    comparison_id: str
    operator_type: str
    problem_hash: str
    before_schedule_hash: str
    local_schedule_hash: str
    global_schedule_hash: str
    seed_proposal: Mapping[str, Any]
    local_delta_cmax: float
    global_delta_cmax: float
    global_extra_gain: float  # local_delta - global_delta (>0 = solver extra)
    teacher_decisions: tuple[Mapping[str, Any], ...]
    induced_consequences: tuple[Mapping[str, Any], ...]
    seed_absorbed_components: tuple[Mapping[str, Any], ...]
    explicit_proposal_operations: tuple[str, ...]
    local_closure_operations: tuple[str, ...]
    global_changed_operations: tuple[str, ...]
    teacher_decision_operations: tuple[str, ...]
    induced_consequence_operations: tuple[str, ...]
    solver_additional_operations: tuple[str, ...]
    affected_operations: tuple[str, ...]
    affected_resources: tuple[str, ...]
    teacher_edit_count: int
    local_solver_status: str = ""
    local_freeze_level: str = ""
    llm_analysis: Mapping[str, Any] | None = None
    replay_analysis: Mapping[str, Any] | None = None
    training_eligible: bool = False
    identified: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": TEACHER_TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "instance_id": self.instance_id,
            "proposal_id": self.proposal_id,
            "comparison_id": self.comparison_id,
            "operator_type": self.operator_type,
            "problem_hash": self.problem_hash,
            "before_schedule_hash": self.before_schedule_hash,
            "local_schedule_hash": self.local_schedule_hash,
            "global_schedule_hash": self.global_schedule_hash,
            "seed_proposal": dict(self.seed_proposal),
            "local_delta_cmax": float(self.local_delta_cmax),
            "global_delta_cmax": float(self.global_delta_cmax),
            "global_extra_gain": float(self.global_extra_gain),
            "teacher_decisions": [dict(e) for e in self.teacher_decisions],
            "induced_consequences": [
                dict(e) for e in self.induced_consequences
            ],
            "seed_absorbed_components": [
                dict(a) for a in self.seed_absorbed_components
            ],
            "teacher_edit_semantics": (
                "component-level subtraction of seed-requested dimensions; "
                "decisions vs induced consequences"
            ),
            "explicit_proposal_operations": list(self.explicit_proposal_operations),
            "local_closure_operations": list(self.local_closure_operations),
            "global_changed_operations": list(self.global_changed_operations),
            "teacher_decision_operations": list(self.teacher_decision_operations),
            "induced_consequence_operations": list(
                self.induced_consequence_operations
            ),
            "solver_additional_operations": list(self.solver_additional_operations),
            "affected_operations": list(self.affected_operations),
            "affected_resources": list(self.affected_resources),
            "teacher_edit_count": int(self.teacher_edit_count),
            "induced_consequence_count": len(self.induced_consequences),
            "local_solver_status": self.local_solver_status,
            "local_freeze_level": self.local_freeze_level,
            "llm_analysis": dict(self.llm_analysis) if self.llm_analysis else None,
            "replay_analysis": (
                dict(self.replay_analysis) if self.replay_analysis else None
            ),
            "training_eligible": False,
            "identified": False,
        }


def _problem_hash(problem: Problem) -> str:
    return hashlib.sha256(problem.model_dump_json().encode("utf-8")).hexdigest()


def build_solver_teacher_trace(
    *,
    problem: Problem,
    before: Schedule,
    local: Schedule,
    global_schedule: Schedule,
    proposal,
    seed_edits: Sequence[Any],
    teacher_edits: TeacherEditSet,
    local_delta_cmax: float,
    global_delta_cmax: float,
    local_solver_status: str = "",
    local_freeze_level: str = "",
    local_closure_operations: Sequence[str] = (),
    comparison_id: str = "",
) -> SolverTeacherTrace:
    """Assemble the trace from the three schedules of one dual evaluation.

    ``global_extra_gain = local_delta - global_delta`` (both signed; a positive
    extra gain means the global solver improved further than the local repair).
    The trace never exposes the global delta as a seed-proposal effect.
    ``teacher_edits`` is the component-level teacher diff (decisions vs
    induced consequences) computed by :func:`extract_teacher_components`.
    """
    explicit = tuple(sorted({e.operation_id for e in seed_edits}))
    global_changed = changed_operation_ids(before, global_schedule)
    decisions = teacher_edits.decisions
    consequences = teacher_edits.consequences
    decision_ops = teacher_edits.decision_operations
    consequence_ops = teacher_edits.consequence_operations
    additional_ops = tuple(sorted(set(decision_ops) | set(consequence_ops)))
    affected_resources = tuple(sorted(
        {e.affected_resource for e in (*decisions, *consequences)}
        | {e.source_resource for e in decisions if e.source_resource}
    ))
    trace_id = (
        f"teacher:{schedule_hash(before)[:12]}:{proposal.proposal_id or 'proposal'}"
    )
    return SolverTeacherTrace(
        trace_id=trace_id,
        instance_id=str(problem.metadata.get("base_instance_id", problem.id)),
        proposal_id=str(proposal.proposal_id),
        comparison_id=str(comparison_id or trace_id),
        operator_type=str(proposal.operator_type),
        problem_hash=_problem_hash(problem),
        before_schedule_hash=schedule_hash(before),
        local_schedule_hash=schedule_hash(local),
        global_schedule_hash=schedule_hash(global_schedule),
        seed_proposal={
            "proposal_id": str(proposal.proposal_id),
            "operator": str(proposal.operator_type),
            "root_decision_id": str(proposal.root_decision_id),
            "causal_chain": list(proposal.causal_chain),
            "actions": [e.edit_id for e in seed_edits],
            "appearance_type": str(proposal.appearance_type),
        },
        local_delta_cmax=float(local_delta_cmax),
        global_delta_cmax=float(global_delta_cmax),
        global_extra_gain=float(local_delta_cmax) - float(global_delta_cmax),
        teacher_decisions=tuple(e.to_json() for e in decisions),
        induced_consequences=tuple(e.to_json() for e in consequences),
        seed_absorbed_components=tuple(a.to_json() for a in teacher_edits.seed_absorbed),
        explicit_proposal_operations=explicit,
        local_closure_operations=tuple(local_closure_operations),
        global_changed_operations=tuple(global_changed),
        teacher_decision_operations=decision_ops,
        induced_consequence_operations=consequence_ops,
        solver_additional_operations=additional_ops,
        affected_operations=additional_ops,
        affected_resources=affected_resources,
        teacher_edit_count=len(decisions),
        local_solver_status=str(local_solver_status),
        local_freeze_level=str(local_freeze_level),
        training_eligible=False,
        identified=False,
    )
