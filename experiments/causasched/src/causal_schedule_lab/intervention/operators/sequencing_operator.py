"""Adjacent resource-sequence swap operator."""

from __future__ import annotations

from ...m2_v5_schema_v1 import EDIT_SEQ_SWAP
from .base import (
    OperatorCandidate,
    OperatorStateGraph,
    primitive_chain,
    selected_root_operation,
)


class SequencingOperator:
    operator_type = "sequencing"

    def generate(self, root_decision, state_graph: OperatorStateGraph) -> tuple[OperatorCandidate, ...]:
        root_operation = selected_root_operation(root_decision)
        return tuple(OperatorCandidate(
            operator_type=self.operator_type,
            root_decision=root_decision,
            chain=primitive_chain(edit, explanation=f"swap {edit.left_id} and {edit.right_id}"),
            score=float(edit.relevance),
        ) for edit in state_graph.legal_edits if (
            edit.edit_type == EDIT_SEQ_SWAP
            and root_operation in {edit.left_id, edit.right_id}
        ))


__all__ = ["SequencingOperator"]
