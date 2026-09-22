"""Dispatch actionable roots to the four deterministic operator families."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .operators import (
    InsertionOperator,
    OperatorCandidate,
    OperatorStateGraph,
    RoutingOperator,
    SequencingOperator,
    TimingOperator,
)


@dataclass(frozen=True)
class OperatorReasoningConfig:
    enabled_operators: tuple[str, ...]
    max_candidates_per_root: int
    timing_shift_radius: int


def load_operator_reasoning_config(path: str | Path | None = None) -> OperatorReasoningConfig:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[3]
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or root / "configs" / "hyperparameters.yaml"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    section = payload.get("operator_reasoning")
    if not isinstance(section, Mapping):
        raise ValueError("missing operator_reasoning hyperparameters")
    return OperatorReasoningConfig(
        enabled_operators=tuple(str(item) for item in section["enabled_operators"]),
        max_candidates_per_root=int(section["max_candidates_per_root"]),
        timing_shift_radius=int(section["timing_shift_radius"]),
    )


class InterventionOperatorReasoner:
    """Operator generator.  CP-SAT is intentionally absent from this layer."""

    def __init__(self, config: OperatorReasoningConfig | None = None) -> None:
        self.config = config or load_operator_reasoning_config()
        self.operators = {
            "routing": RoutingOperator(),
            "sequencing": SequencingOperator(),
            "insertion": InsertionOperator(),
            "timing": TimingOperator(),
        }

    def generate(
        self, root_decision: object, state_graph: OperatorStateGraph
    ) -> tuple[OperatorCandidate, ...]:
        allowed = set(getattr(root_decision, "operator_types", self.config.enabled_operators))
        candidates: list[OperatorCandidate] = []
        for name in self.config.enabled_operators:
            if name in allowed:
                candidates.extend(self.operators[name].generate(root_decision, state_graph))
        candidates.sort(key=lambda row: (-row.score, row.operator_type,
                                          tuple(edit.edit_id for edit in row.chain.actions)))
        return tuple(candidates[: self.config.max_candidates_per_root])

    def generate_many(
        self, roots: Sequence[object], state_graph: OperatorStateGraph
    ) -> tuple[OperatorCandidate, ...]:
        return tuple(candidate for root in roots for candidate in self.generate(root, state_graph))


__all__ = [
    "InterventionOperatorReasoner",
    "OperatorReasoningConfig",
    "load_operator_reasoning_config",
]
