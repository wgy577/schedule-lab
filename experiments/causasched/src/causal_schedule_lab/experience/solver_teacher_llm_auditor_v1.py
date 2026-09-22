"""Solver Teacher Trace V1 -- DeepSeek interpretation of solver teacher edits.

DeepSeek's role here is strictly **hypothesis generation**, never truth: it may
propose roles (blocker release, preparation, direct improvement, ...),
dependency opinions and replay groups for the solver's additional edits.  Every
hypothesis is later verified deterministically by
:mod:`solver_trace_replay_v1`; unverified hypotheses stay ``verified=false``.

Safety contract (fail-closed, sidecar only):

* the prompt carries **no** reward, FIV, success, risk, future-gain or training
  target -- the model sees schedule geometry only;
* the response model is ``extra="forbid"`` and additionally screens the
  forbidden field names (``delta_cmax``, ``success``, ``reward``, ``risk``,
  ``future_gain``, ``fiv``) explicitly, so any attempt to emit a label or
  verdict is rejected into a sidecar rejection record;
* Trajectory Memory is never touched; results are sidecar artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ir import Problem, Schedule
from ..providers import ModelRequest
from .llm_provider_adapter import LLMProviderAdapter
from .solver_schedule_diff_v1 import SolverEdit
from .solver_teacher_trace_v1 import SolverTeacherTrace, SolverTeacherTraceConfig

LLM_SCHEMA = "solver_teacher_llm_analysis_v1"
PROMPT_VERSION = "solver-teacher-trace-v1.1-component-decomposition"

LIKELY_ROLES = (
    "preparation",
    "blocker_release",
    "direct_improvement",
    "feasibility_repair",
    "sequence_repair",
    "resource_rebalancing",
    "critical_path_reduction",
    "secondary_optimization",
    "induced_propagation",
    "unclear",
)

FORBIDDEN_OUTPUT_FIELDS = (
    "delta_cmax", "success", "reward", "risk", "future_gain", "fiv",
)

ANALYSIS_OK = "analyzed"
ANALYSIS_REJECTED = "rejected"
ANALYSIS_SKIPPED = "skipped_no_teacher_edits"

__all__ = [
    "SolverTeacherLLMAuditorV1",
    "TeacherLLMAnalysisResult",
    "TeacherEditOpinion",
    "LLM_SCHEMA",
    "PROMPT_VERSION",
    "LIKELY_ROLES",
    "FORBIDDEN_OUTPUT_FIELDS",
]


class _TeacherEditOpinion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edit_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    likely_role: str = Field(min_length=1)
    dependency_on_seed: float = Field(default=0.5, ge=0.0, le=1.0)
    depends_on: tuple[str, ...] = ()
    enables: tuple[str, ...] = ()
    explanation: str = Field(default="", max_length=2000)


class _DeepSeekTeacherAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # trace_id is an optional echo: models sometimes omit it; when present it
    # must match (mismatch -> rejection).  The hard fail-closed screens are
    # the forbidden output fields and the extra="forbid" schema itself.
    trace_id: str = ""
    overall_interpretation: str = Field(default="", max_length=4000)
    edits: tuple[_TeacherEditOpinion, ...] = ()
    suggested_dependency_chains: tuple[tuple[str, ...], ...] = ()
    suggested_replay_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class TeacherEditOpinion:
    edit_id: str
    operation_id: str
    likely_role: str
    dependency_on_seed: float
    depends_on: tuple[str, ...]
    enables: tuple[str, ...]
    explanation: str

    def to_json(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "operation_id": self.operation_id,
            "likely_role": self.likely_role,
            "dependency_on_seed": float(self.dependency_on_seed),
            "depends_on": list(self.depends_on),
            "enables": list(self.enables),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class TeacherLLMAnalysisResult:
    trace_id: str
    analysis_status: str
    failure_reason: str = ""
    prompt_sha256: str = ""
    response_sha256: str = ""
    model: str = ""
    overall_interpretation: str = ""
    edit_opinions: tuple[TeacherEditOpinion, ...] = ()
    suggested_dependency_chains: tuple[tuple[str, ...], ...] = ()
    suggested_replay_groups: tuple[tuple[str, ...], ...] = ()
    api_called: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 0
    latency_seconds: float = 0.0
    truth_source: str = "deterministic replay/CP-SAT only; LLM is hypothesis-only"
    training_eligible: bool = False
    identified: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": LLM_SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "trace_id": self.trace_id,
            "analysis_status": self.analysis_status,
            "failure_reason": self.failure_reason,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "model": self.model,
            "overall_interpretation": self.overall_interpretation,
            "edit_opinions": [item.to_json() for item in self.edit_opinions],
            "suggested_dependency_chains": [
                list(chain) for chain in self.suggested_dependency_chains
            ],
            "suggested_replay_groups": [
                list(group) for group in self.suggested_replay_groups
            ],
            "api_called": self.api_called,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "attempts": self.attempts,
            "latency_seconds": self.latency_seconds,
            "truth_source": self.truth_source,
            "training_eligible": False,
            "identified": False,
        }


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _machine_loads(problem: Problem, schedule: Schedule) -> dict[str, dict[str, float]]:
    mode_map = problem.mode_map()
    loads: dict[str, dict[str, float]] = {}
    for assignment in schedule.assignments:
        machine = mode_map[assignment.mode_id][1].resources[0]
        entry = loads.setdefault(
            machine, {"busy": 0.0, "span": 0.0, "ops": 0.0}
        )
        entry["busy"] += float(assignment.end) - float(assignment.start)
        entry["ops"] += 1.0
    for machine, entry in loads.items():
        ends = [
            float(a.end) for a in schedule.assignments
            if mode_map[a.mode_id][1].resources[0] == machine
        ]
        entry["span"] = max(ends) if ends else 0.0
    return loads


class SolverTeacherLLMAuditorV1:
    """Ask DeepSeek to interpret solver teacher edits as hypotheses only."""

    def __init__(
        self,
        adapter: LLMProviderAdapter,
        *,
        config: SolverTeacherTraceConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or SolverTeacherTraceConfig()
        self.config.validate()

    def analyze(
        self,
        trace: SolverTeacherTrace,
        *,
        problem: Problem,
        before: Schedule,
        local: Schedule,
        global_schedule: Schedule,
        decisions: Sequence[SolverEdit],
        consequences: Sequence[SolverEdit],
    ) -> TeacherLLMAnalysisResult:
        if not decisions and not consequences:
            return TeacherLLMAnalysisResult(
                trace_id=trace.trace_id, analysis_status=ANALYSIS_SKIPPED,
            )
        prompt = self._build_prompt(
            trace, problem=problem, before=before, local=local,
            global_schedule=global_schedule, decisions=decisions,
            consequences=consequences,
        )
        prompt_sha = _canonical_sha({"prompt": prompt})
        response = None
        try:
            response = self.adapter.provider.complete(ModelRequest(
                system=(
                    "You are a scheduling-solver behaviour analyst. A CP-SAT "
                    "solver repaired a schedule globally; you receive the diff "
                    "between the frozen-local repair and the global repair. "
                    "Interpret WHY the solver made these additional edits. "
                    "You produce hypotheses only: every claim will be verified "
                    "deterministically by replay. Never output or predict "
                    "delta_cmax, success, reward, risk, future_gain, or FIV "
                    "values. Return strict JSON."
                ),
                user=prompt,
                temperature=self.config.llm_temperature,
                max_output_tokens=self.config.llm_max_tokens,
                require_json=True,
                thinking_mode=self.config.thinking_mode,
                metadata={
                    "task": LLM_SCHEMA,
                    "trace_id": trace.trace_id,
                    "prompt_version": PROMPT_VERSION,
                },
            ))
            parsed = self._parse_response(response.content, trace.trace_id)
        except (Exception, ValidationError) as error:
            # Provider failure, malformed JSON, forbidden fields, schema
            # violations and id mismatches all fail closed into a sidecar
            # rejection; the trace and memory remain untouched.
            return TeacherLLMAnalysisResult(
                trace_id=trace.trace_id,
                analysis_status=ANALYSIS_REJECTED,
                failure_reason=(
                    f"analysis_response_rejected:{type(error).__name__}:"
                    f"{str(error)[:200]}"
                ),
                prompt_sha256=prompt_sha,
                response_sha256=(
                    _canonical_sha({"content": response.content})
                    if response is not None else ""
                ),
                model=(response.response_model or response.requested_model)
                if response is not None else "",
                api_called=True,
                input_tokens=response.usage.input_tokens if response else 0,
                output_tokens=response.usage.output_tokens if response else 0,
                total_tokens=response.usage.total_tokens if response else 0,
                attempts=response.attempts if response else 0,
                latency_seconds=response.latency_seconds if response else 0.0,
            )
        return TeacherLLMAnalysisResult(
            trace_id=trace.trace_id,
            analysis_status=ANALYSIS_OK,
            prompt_sha256=prompt_sha,
            response_sha256=_canonical_sha({"content": response.content}),
            model=response.response_model or response.requested_model,
            overall_interpretation=parsed.overall_interpretation,
            edit_opinions=tuple(
                TeacherEditOpinion(
                    edit_id=item.edit_id,
                    operation_id=item.operation_id,
                    likely_role=item.likely_role,
                    dependency_on_seed=float(item.dependency_on_seed),
                    depends_on=tuple(item.depends_on),
                    enables=tuple(item.enables),
                    explanation=item.explanation,
                )
                for item in parsed.edits
            ),
            suggested_dependency_chains=tuple(
                tuple(chain) for chain in parsed.suggested_dependency_chains
            ),
            suggested_replay_groups=tuple(
                tuple(group) for group in parsed.suggested_replay_groups
            ),
            api_called=True,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            attempts=response.attempts,
            latency_seconds=response.latency_seconds,
        )

    def _parse_response(self, content: str, trace_id: str) -> _DeepSeekTeacherAnalysis:
        raw = json.loads(content)
        if not isinstance(raw, dict):
            raise ValueError("response_not_a_json_object")
        forbidden = [key for key in raw if str(key).lower() in FORBIDDEN_OUTPUT_FIELDS]
        if forbidden:
            raise ValueError(f"forbidden_output_field:{forbidden[0]}")
        for item in raw.get("edits", []) or []:
            if isinstance(item, dict):
                forbidden = [
                    key for key in item
                    if str(key).lower() in FORBIDDEN_OUTPUT_FIELDS
                ]
                if forbidden:
                    raise ValueError(f"forbidden_output_field:{forbidden[0]}")
        parsed = _DeepSeekTeacherAnalysis.model_validate_json(content)
        if parsed.trace_id and parsed.trace_id != trace_id:
            raise ValueError("trace_id_mismatch")
        for item in parsed.edits:
            if item.likely_role not in LIKELY_ROLES:
                raise ValueError(f"unknown_likely_role:{item.likely_role}")
        return parsed

    def _build_prompt(
        self,
        trace: SolverTeacherTrace,
        *,
        problem: Problem,
        before: Schedule,
        local: Schedule,
        global_schedule: Schedule,
        decisions: Sequence[SolverEdit],
        consequences: Sequence[SolverEdit],
    ) -> str:
        def cmax(schedule: Schedule) -> float:
            return max(
                (float(a.end) for a in schedule.assignments), default=0.0
            )

        def component(edit: SolverEdit) -> dict[str, Any]:
            return {
                "edit_id": edit.edit_id,
                "operation_id": edit.operation_id,
                "edit_type": edit.edit_type,
                "before": {
                    "machine": edit.before.get("machine"),
                    "start": edit.before.get("start"),
                    "end": edit.before.get("end"),
                },
                "after": {
                    "machine": edit.after.get("machine"),
                    "start": edit.after.get("start"),
                    "end": edit.after.get("end"),
                },
            }

        payload = {
            "task": (
                "interpret the CP-SAT solver's additional edits between the "
                "frozen-local repair and the free-global repair of the same "
                "seed proposal; hypotheses only. Teacher edits are decomposed "
                "into DECISIONS (machine/mode/sequence changes the solver "
                "chose) and CONSEQUENCES (timing-only shifts that may be pure "
                "propagation of those decisions)"
            ),
            "trace_id": trace.trace_id,
            "instance": {
                "operations": len(problem.operations),
                "machines": len(problem.resources),
                "cmax_before": cmax(before),
                "cmax_frozen_local": cmax(local),
                "cmax_free_global": cmax(global_schedule),
            },
            "seed_proposal": {
                "proposal_id": trace.seed_proposal.get("proposal_id"),
                "operator": trace.seed_proposal.get("operator"),
                "appearance_type": trace.seed_proposal.get("appearance_type"),
                "root_decision_id": trace.seed_proposal.get("root_decision_id"),
                "causal_chain": trace.seed_proposal.get("causal_chain"),
                "explicit_operations": list(trace.explicit_proposal_operations),
            },
            "teacher_decisions": [component(e) for e in decisions],
            "induced_consequences": [component(e) for e in consequences],
            "machine_loads": {
                "frozen_local": _machine_loads(problem, local),
                "free_global": _machine_loads(problem, global_schedule),
            },
            "questions": {
                "per_decision": {
                    "likely_role": (
                        "one of: " + ", ".join(
                            role for role in LIKELY_ROLES
                            if role != "induced_propagation"
                        )
                    ),
                    "dependency_on_seed": (
                        "0..1 -- how much this edit only makes sense given the "
                        "seed proposal moved its operations first"
                    ),
                    "depends_on": (
                        "edit_ids of teacher decisions that must happen before this one"
                    ),
                    "enables": (
                        "edit_ids of teacher decisions (or 'seed') this edit unblocks"
                    ),
                    "explanation": "one short paragraph, geometry-based",
                },
                "per_consequence": {
                    "likely_role": "induced_propagation",
                    "explanation": (
                        "which decision/seed edit's propagation explains this "
                        "timing shift (precedence release, freed machine window)"
                    ),
                },
                "overall_interpretation": (
                    "what was the solver's teacher strategy, e.g. release a "
                    "blocked machine window so the seed's move pays off, and "
                    "how the consequences propagate from the decisions"
                ),
                "suggested_replay_groups": (
                    "groups of DECISION edit_ids that should be replayed "
                    "together to test enabling relations (each group a JSON "
                    "array); consequence edits are never replayed"
                ),
            },
            "forbidden": [
                "Do not output, predict or correct delta_cmax, success, "
                "reward, risk, future_gain, or FIV values.",
                "Do not claim causality as established; replay verifies later.",
                "Do not propose training decisions.",
                "Do not treat consequence timing shifts as independent solver "
                "actions unless the propagation story fails.",
            ],
            "output_schema": {
                "trace_id": trace.trace_id,
                "overall_interpretation": "string",
                "edits": [
                    {
                        "edit_id": "string (decision or consequence edit_id)",
                        "operation_id": "string",
                        "likely_role": (
                            "enum from questions.per_decision.likely_role for "
                            "decisions; induced_propagation for consequences"
                        ),
                        "dependency_on_seed": "number 0..1",
                        "depends_on": ["edit_id"],
                        "enables": ["edit_id or 'seed'"],
                        "explanation": "string",
                    }
                ],
                "suggested_dependency_chains": [["edit_id", "edit_id"]],
                "suggested_replay_groups": [["decision_edit_id", "decision_edit_id"]],
            },
        }
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
