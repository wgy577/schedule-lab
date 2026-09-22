"""Resource-sequence insertion operator."""

from __future__ import annotations

from ...m2_v5_schema_v1 import EDIT_SEQ_INSERT
from .base import (
    OperatorCandidate,
    OperatorStateGraph,
    primitive_chain,
    selected_root_operation,
)


class InsertionOperator:
    operator_type = "insertion"

    def generate(self, root_decision, state_graph: OperatorStateGraph) -> tuple[OperatorCandidate, ...]:
        root_operation = selected_root_operation(root_decision)
        return tuple(OperatorCandidate(
            operator_type=self.operator_type,
            root_decision=root_decision,
            chain=primitive_chain(
                edit,
                explanation=(
                    f"insert {edit.operation_id} between "
                    f"{edit.predecessor_id or 'START'} and {edit.successor_id or 'END'}"
                ),
            ),
            score=float(edit.relevance),
        ) for edit in state_graph.legal_edits if (
            edit.edit_type == EDIT_SEQ_INSERT
            and edit.operation_id == root_operation
        ))


__all__ = ["InsertionOperator"]
