"""LLM-assisted candidate generation with deterministic validation gates.

The LLM is restricted to candidate proposal and explanation.  It never writes
Memory and never supplies reward, labels, success, risk, or future gain.  A
candidate reaches the durable :class:`ExperienceStore` only after exact legal
edit matching and a real CP-SAT counterfactual transition whose evidence gates
all pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..counterfactual import CounterfactualEvaluator, proposal_from_edits
from ..ir import Problem, Schedule
from ..m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    EDIT_TIMING_SHIFT,
    LegalEdit,
)
from ..memory import ExperienceStore, ProposalRecord
from .llm_provider_adapter import ExperienceGenerationConfig, LLMProviderAdapter

PROMPT_VERSION = "llm-experience-candidate-v1.1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMAction(_StrictModel):
    edit_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    from_machine: str | None = None
    to_machine: str | None = None
    resource_id: str | None = None
    left_id: str | None = None
    right_id: str | None = None
    insert_position: int | None = None
    predecessor_id: str | None = None
    successor_id: str | None = None
    target_start: float | None = None


class _LLMCandidate(_StrictModel):
    root_decision: str = Field(min_length=1)
    operator: Literal["routing", "sequencing", "insertion", "timing"]
    actions: tuple[LLMAction, ...] = Field(min_length=1)
    causal_reason: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class _LLMEnvelope(_StrictModel):
    candidates: tuple[_LLMCandidate, ...]


@dataclass(frozen=True)
class ExperienceCandidate:
    index: int
    root_decision: str
    operator: str
    edits: tuple[LegalEdit, ...]
    proposal: ProposalRecord
    causal_reason: str
    analysis_confidence: float | None


@dataclass(frozen=True)
class CandidateDisposition:
    index: int
    status: str
    reason: str
    memory_key: str = ""
    solver_status: str = ""
    delta_cmax: float | None = None


@dataclass(frozen=True)
class ExperienceGenerationBatch:
    prompt: str
    prompt_hash: str
    provider: str
    model_name: str
    candidates: tuple[_LLMCandidate, ...]
    failure_reason: str = ""
    prompt_version: str = PROMPT_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    attempts: int = 0


@dataclass(frozen=True)
class PipelineRunResult:
    generation: ExperienceGenerationBatch
    persisted_keys: tuple[str, ...]
    dispositions: tuple[CandidateDisposition, ...]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    fields = {}
    for name in (
        "site_id", "operation_id", "root_candidate_id", "operator_type",
        "operator_types", "nodes", "relations", "depth", "score",
    ):
        if hasattr(value, name):
            fields[name] = _jsonable(getattr(value, name))
    return fields or str(value)


def _state_summary(state: Any) -> dict[str, Any]:
    machine_load = getattr(state, "machine_load", {}) or {}
    cmax = float(getattr(state, "cmax", 0.0))
    utilization = {
        str(machine): (float(values[0]) / max(cmax, 1.0))
        for machine, values in machine_load.items()
    }


def _compact_chain(value: Any) -> dict[str, Any]:
    """Serialize only bounded causal evidence, never recursive search traces."""
    return {
        "decision_site_id": str(getattr(value, "decision_site_id", "")),
        "root_candidate_id": str(getattr(value, "root_candidate_id", "")),
        "operation_id": str(getattr(value, "operation_id", "")),
        "nodes": [str(item) for item in (getattr(value, "nodes", ()) or ())],
        "relations": [str(item) for item in (getattr(value, "relations", ()) or ())],
        "depth": int(getattr(value, "depth", 0) or 0),
        "causal_score": float(getattr(value, "causal_score", 0.0) or 0.0),
    }


def _compact_root(value: Any) -> dict[str, Any]:
    return {
        "site_id": str(getattr(value, "site_id", "")),
        "operation_id": str(getattr(value, "operation_id", "")),
        "decision_type": str(getattr(value, "decision_type", "")),
        "source_machine": getattr(value, "source_machine", None),
        "target_machine": getattr(value, "target_machine", None),
        "resource_id": getattr(value, "resource_id", None),
        "predecessor_id": getattr(value, "predecessor_id", None),
        "successor_id": getattr(value, "successor_id", None),
        "z_deviation": float(getattr(value, "z_deviation", 0.0) or 0.0),
        "edit_support": int(getattr(value, "edit_support", 0) or 0),
    }


def _compact_legal_edit(value: LegalEdit) -> dict[str, Any]:
    """Expose exact executable fields while omitting diagnostic feature vectors."""
    return {
        "edit_id": value.edit_id,
        "edit_type": value.edit_type,
        "operation_id": value.operation_id,
        "source_machine": value.source_machine,
        "target_machine": value.target_machine,
        "target_mode_id": value.target_mode_id,
        "resource_id": value.resource_id,
        "left_id": value.left_id,
        "right_id": value.right_id,
        "insert_position": value.insert_position,
        "predecessor_id": value.predecessor_id,
        "successor_id": value.successor_id,
        "target_start": value.target_start,
    }
    return {
        "appearance": str(getattr(state, "appearance_type", "")),
        "cmax": cmax,
        "critical_path_length": float(getattr(state, "critical_path_length", cmax)),
        "critical_machine": getattr(state, "critical_machine", None),
        "machine_utilization": utilization,
    }


def _root_identifiers(root_candidates: Sequence[Any]) -> set[str]:
    identifiers: set[str] = set()
    for root in root_candidates:
        if isinstance(root, str):
            identifiers.add(root)
            continue
        for holder in (root, getattr(root, "decision_site", None), getattr(root, "causal_chain", None)):
            if holder is None:
                continue
            for name in ("site_id", "operation_id", "root_candidate_id", "decision_site_id"):
                value = getattr(holder, name, None)
                if value:
                    identifiers.add(str(value))
    return identifiers


def _operator_names(operator_candidates: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for candidate in operator_candidates:
        if isinstance(candidate, str):
            names.add(candidate)
        else:
            value = getattr(candidate, "operator_type", None)
            if value:
                names.add(str(value))
    return names or {"routing", "sequencing", "insertion", "timing"}


def _chain_for_root(causal_chains: Sequence[Any], root: str) -> tuple[str, ...]:
    for chain in causal_chains:
        identifiers = {
            str(getattr(chain, name))
            for name in ("root_candidate_id", "decision_site_id")
            if getattr(chain, name, None)
        }
        nodes = tuple(str(item) for item in (getattr(chain, "nodes", ()) or ()))
        if root in identifiers or root in nodes:
            return nodes
    if causal_chains:
        return tuple(str(item) for item in (getattr(causal_chains[0], "nodes", ()) or ()))
    return ()


_OPERATOR_EDIT_TYPES = {
    "routing": {EDIT_ROUTE},
    "sequencing": {EDIT_SEQ_SWAP},
    "insertion": {EDIT_SEQ_INSERT},
    "timing": {EDIT_TIMING_SHIFT},
}


def _allowed_operator_names(
    operator_candidates: Sequence[Any],
    legal_constraints: Sequence[LegalEdit],
) -> set[str]:
    names = _operator_names(operator_candidates)
    for name, edit_types in _OPERATOR_EDIT_TYPES.items():
        if any(edit.edit_type in edit_types for edit in legal_constraints):
            names.add(name)
    return names


class LLMExperienceGenerator:
    """Build a bounded prompt and parse strict candidate JSON fail-closed."""

    def __init__(
        self,
        adapter: LLMProviderAdapter,
        *,
        config: ExperienceGenerationConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or adapter.config
        self.config.validate()
        self.operator_target_distribution: Mapping[str, float] = {}

    def build_prompt(
        self,
        state: Any,
        causal_chains: Sequence[Any],
        root_candidates: Sequence[Any],
        operator_candidates: Sequence[Any],
        legal_constraints: Sequence[LegalEdit],
    ) -> str:
        legal_rows = [_compact_legal_edit(edit) for edit in legal_constraints]
        accepted_root_ids = sorted(_root_identifiers(root_candidates))
        payload = {
            "task": "propose_intervention_experience_candidates_only",
            "candidate_limit": self.config.candidates_per_state,
            "state_summary": _state_summary(state),
            "causal_explorer_chains": [_compact_chain(item) for item in causal_chains],
            "accepted_root_ids": accepted_root_ids,
            "root_candidates": [_compact_root(item) for item in root_candidates],
            "allowed_operators": sorted(
                _allowed_operator_names(operator_candidates, legal_constraints)
            ),
            "operator_mix_target": dict(self.operator_target_distribution),
            "legal_constraints": legal_rows,
            "rules": [
                "Use only edit_id values from legal_constraints.",
                "root_decision must equal one exact string from accepted_root_ids.",
                "Every action field must agree exactly with its referenced legal edit.",
                "Do not predict reward, success, delta_cmax, future_gain, or risk.",
                "confidence is analysis-only and never affects reward or acceptance.",
                "Return strict JSON with no markdown or commentary.",
            ],
            "output_schema": {
                "candidates": [{
                    "root_decision": "exact string from accepted_root_ids",
                    "operator": "routing|sequencing|insertion|timing",
                    "actions": [{
                        "edit_id": "exact legal edit id",
                        "operation": "operation id",
                        "from_machine": None,
                        "to_machine": None,
                        "resource_id": None,
                        "left_id": None,
                        "right_id": None,
                        "insert_position": None,
                        "predecessor_id": None,
                        "successor_id": None,
                        "target_start": None,
                    }],
                    "causal_reason": "string",
                    "confidence": None,
                }]
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def generate(
        self,
        state: Any,
        causal_chains: Sequence[Any],
        root_candidates: Sequence[Any],
        operator_candidates: Sequence[Any],
        legal_constraints: Sequence[LegalEdit],
    ) -> ExperienceGenerationBatch:
        prompt = self.build_prompt(
            state, causal_chains, root_candidates, operator_candidates, legal_constraints
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            response = self.adapter.generate_completion(prompt, self.config)
        except Exception as error:  # provider errors/timeouts/rate limits all fail closed
            return ExperienceGenerationBatch(
                prompt=prompt, prompt_hash=prompt_hash,
                provider=self.config.llm_provider, model_name=self.config.model,
                candidates=(),
                failure_reason=(
                    f"provider_error:{type(error).__name__}:"
                    f"{str(error)[:1000]}"
                ),
            )
        try:
            raw = json.loads(response.content)
            envelope = _LLMEnvelope.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as error:
            return ExperienceGenerationBatch(
                prompt=prompt, prompt_hash=prompt_hash,
                provider=response.provider,
                model_name=response.response_model or response.requested_model,
                candidates=(),
                failure_reason=f"invalid_json_or_schema:{type(error).__name__}",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                latency_seconds=response.latency_seconds,
                attempts=response.attempts,
            )
        return ExperienceGenerationBatch(
            prompt=prompt, prompt_hash=prompt_hash,
            provider=response.provider,
            model_name=response.response_model or response.requested_model,
            candidates=tuple(envelope.candidates[: self.config.candidates_per_state]),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            latency_seconds=response.latency_seconds,
            attempts=response.attempts,
        )

    def validate_candidates(
        self,
        batch: ExperienceGenerationBatch,
        *,
        state: Any,
        causal_chains: Sequence[Any],
        root_candidates: Sequence[Any],
        operator_candidates: Sequence[Any],
        legal_constraints: Sequence[LegalEdit],
    ) -> tuple[tuple[ExperienceCandidate, ...], tuple[CandidateDisposition, ...]]:
        legal_index = {edit.edit_id: edit for edit in legal_constraints}
        roots = _root_identifiers(root_candidates)
        operators = _allowed_operator_names(operator_candidates, legal_constraints)
        valid: list[ExperienceCandidate] = []
        rejected: list[CandidateDisposition] = []
        for index, candidate in enumerate(batch.candidates):
            try:
                if candidate.root_decision not in roots:
                    raise ValueError("root_not_in_supplied_candidates")
                if candidate.operator not in operators:
                    raise ValueError("operator_not_allowed")
                if len({action.edit_id for action in candidate.actions}) != len(candidate.actions):
                    raise ValueError("duplicate_action")
                edits: list[LegalEdit] = []
                for action in candidate.actions:
                    edit = legal_index.get(action.edit_id)
                    if edit is None:
                        raise ValueError("action_not_in_legal_edit_set")
                    edit.validate()
                    if edit.edit_type not in _OPERATOR_EDIT_TYPES[candidate.operator]:
                        raise ValueError("operator_edit_type_mismatch")
                    self._validate_action_matches_edit(action, edit)
                    edits.append(edit)
                chain = _chain_for_root(causal_chains, candidate.root_decision)
                dependencies = tuple(
                    (left.edit_id, right.edit_id)
                    for left, right in zip(edits, edits[1:])
                )
                proposal = proposal_from_edits(
                    tuple(edits),
                    appearance_type=str(getattr(state, "appearance_type", "")),
                    root_nodes=(candidate.root_decision,),
                    proposal_id=f"llmexp:{batch.prompt_hash[:16]}:{index}",
                    dependency_edges=dependencies,
                    affected_region=tuple(dict.fromkeys(
                        [candidate.root_decision, *(edit.operation_id for edit in edits)]
                    )),
                    causal_chain=chain,
                    root_decision_id=candidate.root_decision,
                    operator_type=candidate.operator,
                    causal_search_trace=("llm_candidate_proposal_only",),
                )
                valid.append(ExperienceCandidate(
                    index=index,
                    root_decision=candidate.root_decision,
                    operator=candidate.operator,
                    edits=tuple(edits), proposal=proposal,
                    causal_reason=candidate.causal_reason,
                    analysis_confidence=candidate.confidence,
                ))
            except ValueError as error:
                rejected.append(CandidateDisposition(
                    index=index, status="discarded", reason=str(error)
                ))
        return tuple(valid), tuple(rejected)

    @staticmethod
    def _validate_action_matches_edit(action: LLMAction, edit: LegalEdit) -> None:
        comparisons = {
            "operation": (action.operation, edit.operation_id),
            "from_machine": (action.from_machine, edit.source_machine),
            "to_machine": (action.to_machine, edit.target_machine),
            "resource_id": (action.resource_id, edit.resource_id),
            "left_id": (action.left_id, edit.left_id),
            "right_id": (action.right_id, edit.right_id),
            "insert_position": (action.insert_position, edit.insert_position),
            "predecessor_id": (action.predecessor_id, edit.predecessor_id),
            "successor_id": (action.successor_id, edit.successor_id),
            "target_start": (action.target_start, edit.target_start),
        }
        for name, (provided, expected) in comparisons.items():
            if name == "operation" or provided is not None:
                if provided != expected:
                    raise ValueError(f"action_field_mismatch:{name}")


class LLMAssistedTrajectoryPipeline:
    """Run LLM -> schema -> legal gate -> macro proposal -> CP-SAT -> Memory."""

    def __init__(
        self,
        generator: LLMExperienceGenerator,
        store: ExperienceStore,
        *,
        solver_time: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.generator = generator
        self.store = store
        self.solver_time = float(solver_time)
        self.seed = int(seed)

    def run(
        self,
        problem: Problem,
        schedule: Schedule,
        *,
        state: Any,
        causal_chains: Sequence[Any],
        root_candidates: Sequence[Any],
        operator_candidates: Sequence[Any],
        legal_constraints: Sequence[LegalEdit],
    ) -> PipelineRunResult:
        batch = self.generator.generate(
            state, causal_chains, root_candidates, operator_candidates, legal_constraints
        )
        if batch.failure_reason:
            return PipelineRunResult(batch, (), (
                CandidateDisposition(-1, "discarded", batch.failure_reason),
            ))
        candidates, dispositions = self.generator.validate_candidates(
            batch,
            state=state,
            causal_chains=causal_chains,
            root_candidates=root_candidates,
            operator_candidates=operator_candidates,
            legal_constraints=legal_constraints,
        )
        statuses = list(dispositions)
        persisted: list[str] = []
        for candidate in candidates:
            # Isolation is essential: the existing evaluator records all solver
            # failures as useful evidence.  LLM experience generation has a
            # stricter contract and commits only fully executed transitions.
            staging = ExperienceStore()
            result = CounterfactualEvaluator(
                staging, solver_time=self.solver_time, seed=self.seed
            ).evaluate(
                problem, schedule, candidate.proposal, candidate.edits,
                store_outcome=True,
            )
            staged = staging.get(result.key)
            evidence = dict(staged.metadata) if staged is not None else {}
            gates = (
                result.new_schedule is not None,
                staged is not None and staged.outcome is not None,
                evidence.get("proposal_legal") is True,
                evidence.get("actions_executed") is True,
                evidence.get("delta_cmax_verified") is True,
                evidence.get("validator_passed") is True,
                evidence.get("counterfactual_mode") == "frozen_local",
                evidence.get("training_eligible") is True,
                not evidence.get("outside_closure_changes"),
            )
            if not all(gates):
                statuses.append(CandidateDisposition(
                    candidate.index, "discarded", "counterfactual_validation_failed",
                    solver_status=result.solver_status,
                    delta_cmax=result.delta_cmax,
                ))
                continue
            assert staged is not None and staged.outcome is not None
            generator_metadata = {
                **evidence,
                "generator_source": batch.provider,
                "model_name": batch.model_name,
                "llm_model": batch.model_name,
                "prompt_hash": batch.prompt_hash,
                "prompt_version": batch.prompt_version,
                "llm_causal_reason": candidate.causal_reason,
                "llm_analysis_confidence": candidate.analysis_confidence,
                "llm_fields_used_for_reward": [],
                "future_gain_source": "observed_cp_sat_trajectory_only",
                "observed_future_gain": 0.0,
                "formal_training": False,
                "formal_test_access": 0,
                "causal_identified": False,
            }
            key = self.store.append(
                staged.state, staged.proposal, outcome=None,
                metadata=generator_metadata,
            )
            self.store.record_outcome(
                key, staged.outcome, after_state=staged.after_state,
                metadata_update=generator_metadata,
            )
            persisted.append(key)
            statuses.append(CandidateDisposition(
                candidate.index, "persisted", "cp_sat_transition_verified",
                memory_key=key, solver_status=result.solver_status,
                delta_cmax=result.delta_cmax,
            ))
        return PipelineRunResult(batch, tuple(persisted), tuple(sorted(
            statuses, key=lambda item: item.index
        )))

    def append_verified_continuation(
        self,
        problem: Problem,
        schedule: Schedule,
        *,
        trajectory_key: str,
        candidate: ExperienceCandidate,
    ) -> CandidateDisposition:
        """Execute a later Pn and attach only a verified transition to P1.

        The caller must obtain ``candidate`` through the same schema/legal
        validation path against the later state.  Future gain is never accepted
        from the LLM: ``ExperienceStore.append_trajectory_step`` recomputes it
        from the original S0 and this CP-SAT-verified Sn.
        """
        if self.store.get(trajectory_key) is None:
            return CandidateDisposition(
                candidate.index, "discarded", "unknown_trajectory_key"
            )
        staging = ExperienceStore()
        result = CounterfactualEvaluator(
            staging, solver_time=self.solver_time, seed=self.seed
        ).evaluate(
            problem, schedule, candidate.proposal, candidate.edits,
            store_outcome=True,
        )
        staged = staging.get(result.key)
        evidence = dict(staged.metadata) if staged is not None else {}
        gates = (
            result.new_schedule is not None,
            staged is not None and staged.outcome is not None,
            evidence.get("proposal_legal") is True,
            evidence.get("actions_executed") is True,
            evidence.get("delta_cmax_verified") is True,
            evidence.get("validator_passed") is True,
            evidence.get("counterfactual_mode") == "frozen_local",
            evidence.get("training_eligible") is True,
            not evidence.get("outside_closure_changes"),
        )
        if not all(gates):
            return CandidateDisposition(
                candidate.index, "discarded", "counterfactual_validation_failed",
                solver_status=result.solver_status, delta_cmax=result.delta_cmax,
            )
        assert staged is not None and staged.outcome is not None and staged.after_state is not None
        self.store.append_trajectory_step(
            trajectory_key,
            staged.proposal,
            staged.outcome,
            after_state=staged.after_state,
            validation_metadata={
                **evidence,
                "continuation_source": "schema_and_legal_validated_llm_candidate",
                "future_gain_source": "observed_cp_sat_trajectory_only",
                "formal_training": False,
                "formal_test_access": 0,
                "causal_identified": False,
            },
        )
        return CandidateDisposition(
            candidate.index, "trajectory_extended",
            "cp_sat_continuation_verified",
            memory_key=trajectory_key, solver_status=result.solver_status,
            delta_cmax=result.delta_cmax,
        )


__all__ = [
    "CandidateDisposition",
    "ExperienceCandidate",
    "ExperienceGenerationBatch",
    "LLMAssistedTrajectoryPipeline",
    "LLMExperienceGenerator",
    "PipelineRunResult",
    "PROMPT_VERSION",
]
