"""Validated LLM reflection and rule distillation.

No reflection is admitted directly.  Every proposal must cite evidence and
pass replay on held-out failures before it can enter the semantic DSL or BC
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .models import ClaimStatus, ExperimentRecord, SemanticClaim
from .semantic_compiler import SemanticProposal, verify_proposals, index_repository


@dataclass(frozen=True)
class ReflectionProposal:
    trigger: str
    semantic: SemanticProposal
    operator: str | None = None
    feature: str | None = None


@dataclass(frozen=True)
class DistillationDecision:
    proposal: ReflectionProposal
    semantic_claim: SemanticClaim
    replay_pass_rate: float
    accepted: bool
    reason: str


Replay = Callable[[ReflectionProposal, tuple[ExperimentRecord, ...]], float]


def reflection_triggers(
    records: Iterable[ExperimentRecord],
    *,
    uncertainty_threshold: float = 0.6,
    plateau: int = 5,
) -> tuple[str, ...]:
    values = list(records)
    triggers = set()
    if any(item.cip.uncertainty >= uncertainty_threshold for item in values):
        triggers.add("high_uncertainty")
    if len(values) >= plateau and not any(item.accepted for item in values[-plateau:]):
        triggers.add("consecutive_plateau")
    known = {
        failure.value
        for item in values
        for verification in item.verifications
        for failure in verification.failures
    }
    if any(
        failure
        for item in values
        for failure in item.metadata.get("unrecognizedFailureLabels", [])
        if failure not in known
    ):
        triggers.add("unseen_failure")
    return tuple(sorted(triggers))


def validate_and_distill(
    project_root: str,
    proposals: Iterable[ReflectionProposal],
    failures: Iterable[ExperimentRecord],
    *,
    replay: Replay,
    minimum_pass_rate: float = 0.8,
) -> tuple[DistillationDecision, ...]:
    proposals = tuple(proposals)
    records = tuple(failures)
    symbols = index_repository(project_root)
    claims = verify_proposals(
        project_root,
        symbols,
        (item.semantic for item in proposals),
    )
    decisions = []
    for proposal, claim in zip(proposals, claims):
        rate = replay(proposal, records)
        accepted = (
            claim.status == ClaimStatus.VERIFIED
            and rate >= minimum_pass_rate
        )
        decisions.append(
            DistillationDecision(
                proposal=proposal,
                semantic_claim=claim,
                replay_pass_rate=rate,
                accepted=accepted,
                reason=(
                    "evidence verified and replay threshold passed"
                    if accepted
                    else "proposal remains quarantined"
                ),
            )
        )
    return tuple(decisions)
