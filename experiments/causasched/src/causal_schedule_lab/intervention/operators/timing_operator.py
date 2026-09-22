"""Exact idle-window timing-shift operator."""

from __future__ import annotations

from ...m2_v5_schema_v1 import EDIT_TIMING_SHIFT
from .base import (
    OperatorCandidate,
    OperatorStateGraph,
    primitive_chain,
    selected_root_operation,
)


class TimingOperator:
    operator_type = "timing"

    def generate(self, root_decision, state_graph: OperatorStateGraph) -> tuple[OperatorCandidate, ...]:
        root_operation = selected_root_operation(root_decision)
        return tuple(OperatorCandidate(
            operator_type=self.operator_type,
            root_decision=root_decision,
            chain=primitive_chain(
                edit, explanation=f"shift {edit.operation_id} to start {edit.target_start:g}"
            ),
            score=float(edit.relevance),
        ) for edit in state_graph.legal_edits if (
            edit.edit_type == EDIT_TIMING_SHIFT and edit.operation_id == root_operation
        ))


__all__ = ["TimingOperator"]
