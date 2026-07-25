from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .models import AgentAction, CausalInterventionPoint, InterventionProposal, ProjectSemantics


GENERIC_OPERATORS = (
    "adjacent_swap",
    "insertion",
    "machine_reassignment",
    "critical_block_resequence",
    "stage_resequence",
    "release_adjustment",
    "blocking_chain_repair",
    "expand_closure",
)


def _signature(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionIndex:
    actions: tuple[tuple[str, int], ...]

    @classmethod
    def from_semantics(cls, semantics: ProjectSemantics) -> "ActionIndex":
        return cls(
            actions=tuple(
                (operator, level)
                for operator in semantics.allowed_interventions
                for level in (1, 2, 3)
            )
        )

    def decode(self, index: int) -> tuple[str, int]:
        return self.actions[index]

    def encode(self, operator: str, closure_level: int) -> int:
        return self.actions.index((operator, closure_level))

    def mask_for(self, cip: CausalInterventionPoint) -> list[bool]:
        recommended = set(cip.recommended_operators)
        mask = []
        for operator, level in self.actions:
            legal = operator in recommended
            if operator == "expand_closure":
                legal = cip.closure.level < 3 and level > cip.closure.level
            elif level < cip.closure.level:
                legal = False
            mask.append(legal)
        if not any(mask):
            mask = [
                operator in recommended or operator == "expand_closure"
                for operator, _ in self.actions
            ]
        return mask


def build_intervention(
    *,
    problem: Any,
    schedule: Any,
    cip: CausalInterventionPoint,
    action: AgentAction,
) -> InterventionProposal:
    all_operations = {operation.id for operation in problem.operations}
    released = set(cip.closure.operation_ids)
    operation_map = problem.operation_map()
    if action.closure_level > cip.closure.level:
        jobs = {
            operation_map[item].job_id
            for item in released
            if item in operation_map
        }
        released.update(
            operation.id for operation in problem.operations if operation.job_id in jobs
        )
    frozen = all_operations - released
    priority_overrides: dict[str, tuple[str, ...]] = {}
    mode_overrides: dict[str, str] = {}
    location = cip.diagnostic.location
    responsible = operation_map[cip.responsible.operation_id]

    if action.operator == "adjacent_swap" and len(location) >= 2:
        left_job = operation_map[location[0]].job_id
        right_job = operation_map[location[1]].job_id
        stage = responsible.index
        priority_overrides[f"O{stage + 1}"] = (right_job, left_job)
    elif action.operator == "insertion":
        jobs = tuple(
            dict.fromkeys(
                operation_map[item].job_id
                for item in location
                if item in operation_map
            )
        )
        if jobs:
            stage = responsible.index
            priority_overrides[f"O{stage + 1}"] = tuple(reversed(jobs))
    elif action.operator in {
        "critical_block_resequence",
        "stage_resequence",
        "blocking_chain_repair",
    }:
        jobs = sorted(
            {
                operation_map[item].job_id
                for item in released
                if item in operation_map
            }
        )
        priority_overrides[f"O{responsible.index + 1}"] = tuple(jobs)
    elif action.operator == "machine_reassignment":
        assignment = schedule.assignment_map()[responsible.id]
        alternatives = sorted(
            (
                mode
                for mode in responsible.modes
                if mode.id != assignment.mode_id
            ),
            key=lambda mode: (mode.duration, mode.id),
        )
        if alternatives:
            mode_overrides[responsible.id] = alternatives[0].id
    elif action.operator in {"release_adjustment", "expand_closure"}:
        pass
    elif action.operator not in GENERIC_OPERATORS:
        # Project plugins may define their own operator names. The generic
        # layer still enforces the released/frozen boundary; the selected
        # generator interprets domain-specific parameters.
        pass
    else:
        raise ValueError(f"unsupported or inapplicable operator: {action.operator}")

    payload = {
        "problem": problem.id,
        "incumbent": schedule.makespan,
        "cip": cip.id,
        "operator": action.operator,
        "closureLevel": action.closure_level,
        "released": sorted(released),
        "priorityOverrides": priority_overrides,
        "modeOverrides": mode_overrides,
    }
    return InterventionProposal(
        cip_id=cip.id,
        action=action,
        released_operations=tuple(sorted(released)),
        frozen_operations=tuple(sorted(frozen)),
        priority_overrides=priority_overrides,
        mode_overrides=mode_overrides,
        signature=_signature(payload),
    )
