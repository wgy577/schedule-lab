"""Routing operator; Transition Reasoner is an internal expansion mechanism."""

from __future__ import annotations

from ...m2_v5_schema_v1 import EDIT_ROUTE
from ..transition_reasoner_v1 import RootDecisionRequest, TransitionReasoner
from .base import OperatorCandidate, OperatorStateGraph, primitive_chain


class RoutingOperator:
    operator_type = "routing"

    def __init__(self, transition_reasoner: TransitionReasoner | None = None) -> None:
        self.transition_reasoner = transition_reasoner or TransitionReasoner()

    def generate(self, root_decision, state_graph: OperatorStateGraph) -> tuple[OperatorCandidate, ...]:
        site = getattr(root_decision, "decision_site", root_decision)
        edits = tuple(
            edit for edit in state_graph.legal_edits
            if edit.edit_type == EDIT_ROUTE and edit.operation_id == site.operation_id
        )
        output: list[OperatorCandidate] = []
        for edit in edits:
            expansion = self.transition_reasoner.expand(
                RootDecisionRequest.from_site_edit(site, edit),
                state_graph.legal_edits,
                state_graph.schedule_graph,
            )
            # T1-PROPOSAL-CLEANUP (Mk1): emit the exact atomic single ROUTE
            # proposal for EVERY legal edit, in addition to any enabling chain
            # found by the transition reasoner.  A found-but-infeasible chain
            # must NOT suppress the feasible single -- Frozen-Local is the
            # authoritative feasibility judge.  If the chain is a single edit it
            # is an exact duplicate of the primitive single and is collapsed by
            # the action-structure dedup in build_operator_runtime.
            single = primitive_chain(
                edit,
                explanation=(
                    f"single routing: {edit.operation_id} "
                    f"{edit.source_machine}->{edit.target_machine}"
                ),
            )
            output.append(OperatorCandidate(
                operator_type=self.operator_type,
                root_decision=root_decision,
                chain=single,
                score=float(single.score),
            ))
            output.extend(OperatorCandidate(
                operator_type=self.operator_type,
                root_decision=root_decision,
                chain=chain,
                score=float(chain.score),
            ) for chain in expansion.chains)
        return tuple(output)


__all__ = ["RoutingOperator"]
