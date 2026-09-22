"""Constrained LLM proposal harness for verified operator SFT trajectories.

The LLM receives a finite candidate set and returns candidate IDs. It cannot
create entities, bypass masks, execute actions, or judge solution quality.
Executor/Validator/Full-Oracle callbacks remain the only quality authority.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from .providers.base import ModelProvider, ModelRequest
from .training_v1 import (
    OperatorParameterCandidate,
    OperatorPolicyState,
    TeacherActionProposal,
    TeacherCandidateResult,
    canonical_action_id,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class LLMTeacherProposalBatch(FrozenModel):
    schema_version: str = "llm-operator-teacher-batch-1.1"
    prompt_template_version: str = "finite-candidate-teacher-1.1"
    provider: str
    model: str
    prompt_hash: str
    response_hash: str
    candidate_set_hash: str
    state_id: str
    requested_count: int = Field(ge=1)
    proposals: tuple[TeacherActionProposal, ...]
    rejected_items: tuple[dict[str, object], ...] = ()
    token_usage: dict[str, int] = Field(default_factory=dict)
    deterministic_request: bool = True


class CandidateEvaluation(FrozenModel):
    static_pass: bool
    light_pass: bool
    full_pass: bool
    deterministic_replay_pass: bool
    full_feasible_pass: bool | None = None
    objective_before: float
    objective_after: float | None = None
    best_lookahead_objective: float | None = None
    runtime_seconds: float = Field(default=0.0, ge=0.0)
    oracle_cost: float = Field(default=0.0, ge=0.0)
    failure_labels: tuple[str, ...] = ()


CandidateEvaluator = Callable[
    [OperatorPolicyState, OperatorParameterCandidate], CandidateEvaluation
]


def build_operator_teacher_request(
    state: OperatorPolicyState,
    *,
    proposal_count: int = 8,
    max_output_tokens: int = 4096,
) -> ModelRequest:
    legal = [
        {
            "candidate_id": item.candidate_id,
            "operator_id": item.operator_id,
            "parameters": item.parameters,
        }
        for item in state.parameter_candidates
        if item.legal
    ]
    system = (
        "You are a scheduling action proposal teacher. Select only candidate_id values "
        "from the supplied finite legal set. Do not invent entities or claim optimality. "
        "Return JSON {proposals:[{candidate_id,mechanism_code,evidence_ids}]}. "
        "Use at most the requested count and rank the most promising first."
    )
    user_payload = {
        "instance_id": state.instance_id,
        "family": state.family,
        "environment": state.environment,
        "state_id": state.state_id,
        "surface_block_ids": state.surface_block_ids,
        "root_cause_id": state.root_cause_id,
        "root_cause_node_ids": state.root_cause_node_ids,
        "causal_path_edge_ids": state.causal_path_edge_ids,
        "current_objective": state.current_objective,
        "lower_bound": state.lower_bound,
        "budget_features": state.budget_features,
        "recent_action_ids": state.recent_action_ids,
        "operator_masks": {
            "semantic": state.semantic_operator_mask,
            "local": state.local_operator_mask,
            "cause": state.cause_operator_mask,
            "final": state.final_operator_mask,
        },
        "selection_rule": (
            "Rank candidates by expected Full-Oracle objective improvement under the fixed budget; "
            "do not treat an improvement estimate as verified."
        ),
        "requested_count": proposal_count,
        "legal_candidates": legal,
    }
    return ModelRequest(
        system=system,
        user=json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        require_json=True,
        reasoning_effort="high",
        metadata={
            "stage": "operator_teacher_proposal",
            "state_id": state.state_id,
            "candidate_set_hash": state.candidate_set_hash,
        },
    )


def collect_llm_operator_proposals(
    provider: ModelProvider,
    state: OperatorPolicyState,
    *,
    proposal_count: int = 8,
    max_output_tokens: int = 4096,
) -> LLMTeacherProposalBatch:
    request = build_operator_teacher_request(
        state,
        proposal_count=proposal_count,
        max_output_tokens=max_output_tokens,
    )
    response = provider.complete(request)
    prompt_hash = hashlib.sha256(
        (request.system + "\n" + request.user).encode("utf-8")
    ).hexdigest()
    response_hash = hashlib.sha256(response.content.encode("utf-8")).hexdigest()
    try:
        raw = json.loads(response.content)
    except json.JSONDecodeError as error:
        raise ValueError("LLM operator teacher returned invalid JSON") from error
    items = raw.get("proposals", []) if isinstance(raw, dict) else []
    candidate_map = {
        item.candidate_id: item for item in state.parameter_candidates if item.legal
    }
    proposals: list[TeacherActionProposal] = []
    rejected: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items[:proposal_count]:
        if not isinstance(item, dict):
            rejected.append({"reason": "not_an_object", "value": repr(item)})
            continue
        candidate_id = str(item.get("candidate_id", ""))
        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            rejected.append({"reason": "unknown_or_illegal_candidate", "candidate_id": candidate_id})
            continue
        if candidate_id in seen:
            rejected.append({"reason": "duplicate_candidate", "candidate_id": candidate_id})
            continue
        seen.add(candidate_id)
        evidence_ids = item.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or not all(isinstance(value, str) for value in evidence_ids):
            evidence_ids = []
        proposals.append(
            TeacherActionProposal(
                source="llm",
                operator_id=candidate.operator_id,
                parameters=candidate.parameters,
                mechanism_code=(
                    str(item["mechanism_code"])
                    if item.get("mechanism_code") is not None else None
                ),
                evidence_ids=tuple(evidence_ids),
            )
        )
    return LLMTeacherProposalBatch(
        provider=response.provider,
        model=response.response_model or response.requested_model,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        candidate_set_hash=state.candidate_set_hash,
        state_id=state.state_id,
        requested_count=proposal_count,
        proposals=tuple(proposals),
        rejected_items=tuple(rejected),
        token_usage=response.usage.model_dump(),
    )


def evaluate_teacher_proposals(
    state: OperatorPolicyState,
    proposals: tuple[TeacherActionProposal, ...],
    evaluator: CandidateEvaluator,
) -> tuple[TeacherCandidateResult, ...]:
    candidate_by_signature = {
        (
            item.operator_id,
            json.dumps(item.parameters, sort_keys=True, separators=(",", ":")),
        ): item
        for item in state.parameter_candidates
    }
    results: list[TeacherCandidateResult] = []
    for index, proposal in enumerate(proposals):
        key = (
            proposal.operator_id,
            json.dumps(proposal.parameters, sort_keys=True, separators=(",", ":")),
        )
        candidate = candidate_by_signature.get(key)
        if candidate is None or not candidate.legal:
            results.append(
                TeacherCandidateResult(
                    candidate_id=f"rejected:{index}",
                    source=proposal.source,
                    action_id="",
                    parse_pass=True,
                    mask_pass=False,
                    static_pass=False,
                    light_pass=False,
                    full_pass=False,
                    deterministic_replay_pass=False,
                    objective_before=0.0,
                    failure_labels=("MASK_REJECTED",),
                )
            )
            continue
        evaluation = evaluator(state, candidate)
        results.append(
            TeacherCandidateResult(
                candidate_id=candidate.candidate_id,
                source=proposal.source,
                action_id=canonical_action_id(
                    state.schedule_hash,
                    state.registry_version,
                    candidate.operator_id,
                    candidate.parameters,
                ),
                parse_pass=True,
                mask_pass=True,
                static_pass=evaluation.static_pass,
                light_pass=evaluation.light_pass,
                full_pass=evaluation.full_pass,
                full_feasible_pass=evaluation.full_feasible_pass,
                deterministic_replay_pass=evaluation.deterministic_replay_pass,
                objective_before=evaluation.objective_before,
                objective_after=evaluation.objective_after,
                best_lookahead_objective=evaluation.best_lookahead_objective,
                runtime_seconds=evaluation.runtime_seconds,
                oracle_cost=evaluation.oracle_cost,
                failure_labels=evaluation.failure_labels,
            )
        )
    return tuple(results)


def best_verified_teacher_result(
    results: tuple[TeacherCandidateResult, ...],
) -> TeacherCandidateResult | None:
    """Return the best verified-under-budget result, never a global-optimal claim."""

    positive = [item for item in results if item.positive_demonstration]
    return min(
        positive,
        key=lambda item: (
            item.objective_after if item.objective_after is not None else float("inf"),
            item.oracle_cost,
            item.action_id,
        ),
        default=None,
    )


__all__ = [
    "CandidateEvaluation",
    "LLMTeacherProposalBatch",
    "best_verified_teacher_result",
    "build_operator_teacher_request",
    "collect_llm_operator_proposals",
    "evaluate_teacher_proposals",
]
