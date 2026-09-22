"""Shared operator-generation contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...m2_v5_schema_v1 import LegalEdit
from ..transition_reasoner_v1 import InterventionChain, ScheduleGraphView


@dataclass(frozen=True)
class OperatorStateGraph:
    schedule_graph: ScheduleGraphView
    legal_edits: tuple[LegalEdit, ...]


@dataclass(frozen=True)
class OperatorCandidate:
    operator_type: str
    root_decision: object
    chain: InterventionChain
    score: float


class InterventionOperator(Protocol):
    operator_type: str

    def generate(
        self, root_decision: object, state_graph: OperatorStateGraph
    ) -> tuple[OperatorCandidate, ...]: ...


def primitive_chain(edit: LegalEdit, *, explanation: str) -> InterventionChain:
    return InterventionChain(
        actions=(edit,), dependencies=(), explanation=(explanation,),
        score=float(edit.relevance), complete=True, depth=1,
    )


def selected_root_operation(root_decision: object) -> str:
    """Resolve the root selected by Causal Explorer, falling back to V1 site."""
    chain = getattr(root_decision, "causal_chain", None)
    candidate = getattr(chain, "root_candidate_id", "") if chain is not None else ""
    if candidate:
        return str(candidate)
    site = getattr(root_decision, "decision_site", root_decision)
    return str(getattr(site, "operation_id"))


__all__ = [
    "InterventionOperator",
    "OperatorCandidate",
    "OperatorStateGraph",
    "primitive_chain",
    "selected_root_operation",
]
