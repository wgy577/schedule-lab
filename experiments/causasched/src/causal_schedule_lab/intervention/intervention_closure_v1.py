"""Bounded, provenance-bearing intervention closure for local attribution.

The closure is a deterministic declaration of which incumbent operations may
move during a frozen-local counterfactual.  CP-SAT never expands it implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..ir import Problem, Schedule
from ..m2_v5_schema_v1 import LegalEdit


@dataclass(frozen=True)
class FrozenLocalCounterfactualConfig:
    max_closure_depth: int
    max_closure_operations: int
    release_precedence_neighbors: bool
    release_resource_neighbors: bool

    def validate(self) -> None:
        if self.max_closure_depth < 0:
            raise ValueError("counterfactual.frozen_local.max_closure_depth must be >= 0")
        if self.max_closure_operations < 1:
            raise ValueError("counterfactual.frozen_local.max_closure_operations must be >= 1")


def _repository_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "hyperparameters.yaml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


def load_frozen_local_counterfactual_config(
    path: str | Path | None = None,
) -> FrozenLocalCounterfactualConfig:
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or (_repository_root() / "configs" / "hyperparameters.yaml")
    ).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    section = payload.get("counterfactual", {}).get("frozen_local", {})
    if not isinstance(section, Mapping):
        raise ValueError("missing counterfactual.frozen_local hyperparameters")
    config = FrozenLocalCounterfactualConfig(
        max_closure_depth=int(section["max_closure_depth"]),
        max_closure_operations=int(section["max_closure_operations"]),
        release_precedence_neighbors=bool(section["release_precedence_neighbors"]),
        release_resource_neighbors=bool(section["release_resource_neighbors"]),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class ClosureState:
    problem: Problem
    schedule: Schedule


@dataclass(frozen=True)
class ClosureMember:
    operation_id: str
    release_reason: str
    source: str
    depth: int


@dataclass(frozen=True)
class InterventionClosure:
    operation_ids: tuple[str, ...]
    members: tuple[ClosureMember, ...]
    max_depth: int
    truncated: bool

    def reason_for(self, operation_id: str) -> str | None:
        member = next((item for item in self.members if item.operation_id == operation_id), None)
        return member.release_reason if member else None


def _operation_from_action_id(action_id: str, known: set[str]) -> str | None:
    for operation_id in sorted(known, key=lambda value: (-len(value), value)):
        if re.search(rf"(^|[:|<@>]){re.escape(operation_id)}($|[:|<@>\-])", action_id):
            return operation_id
    return None


def _trace_operations(transition_trace: Any, known: set[str]) -> tuple[str, ...]:
    if transition_trace is None:
        return ()
    found: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, LegalEdit):
            found.append(value.operation_id)
        elif isinstance(value, Mapping):
            for key in ("operation_id", "blocker", "blocker_id", "operation"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate in known:
                    found.append(candidate)
            for key in ("actions", "dependencies", "blockers", "trace"):
                if key in value:
                    visit(value[key])
        elif isinstance(value, str):
            for operation_id in known:
                if re.search(rf"(?<![\w.]){re.escape(operation_id)}(?![\w.])", value):
                    found.append(operation_id)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                visit(item)
        else:
            actions = getattr(value, "actions", None)
            blockers = getattr(value, "blockers", None)
            dependencies = getattr(value, "dependencies", None)
            for item in (actions, blockers, dependencies):
                if item is not None:
                    visit(item)

    visit(transition_trace)
    return tuple(dict.fromkeys(item for item in found if item in known))


def build_intervention_closure(
    state: ClosureState,
    causal_chain: Any,
    root: Any,
    proposal: Any,
    transition_trace: Any,
    *,
    edits: Sequence[LegalEdit] = (),
    config: FrozenLocalCounterfactualConfig | None = None,
) -> InterventionClosure:
    r"""Build bounded :math:`\Omega(P)` with a reason for every released operation."""
    del causal_chain  # causal explanation alone does not grant solver freedom.
    cfg = config or load_frozen_local_counterfactual_config()
    cfg.validate()
    problem, schedule = state.problem, state.schedule
    known = {operation.id for operation in problem.operations}
    edit_by_id = {edit.edit_id: edit for edit in edits}
    members: dict[str, ClosureMember] = {}
    truncated = False

    def add(operation_id: str | None, reason: str, source: str, depth: int) -> None:
        nonlocal truncated
        if not operation_id or operation_id not in known or operation_id in members:
            return
        if len(members) >= cfg.max_closure_operations:
            truncated = True
            return
        members[operation_id] = ClosureMember(operation_id, reason, source, depth)

    for edit in edits:
        add(edit.operation_id, "explicit_proposal", edit.edit_id, 0)
        if edit.left_id:
            add(edit.left_id, "explicit_proposal", edit.edit_id, 0)
        if edit.right_id:
            add(edit.right_id, "explicit_proposal", edit.edit_id, 0)
        if edit.predecessor_id:
            add(edit.predecessor_id, "macro_dependency", edit.edit_id, 0)
        if edit.successor_id:
            add(edit.successor_id, "macro_dependency", edit.edit_id, 0)

    for edge in getattr(proposal, "dependency_edges", ()) or ():
        for endpoint in edge:
            edit = edit_by_id.get(str(endpoint))
            add(
                edit.operation_id if edit else _operation_from_action_id(str(endpoint), known),
                "macro_dependency", str(endpoint), 0,
            )
    root_id = getattr(root, "operation_id", root if isinstance(root, str) else None)
    if isinstance(root_id, str):
        add(root_id, "root_decision", str(root_id), 0)
    for operation_id in _trace_operations(transition_trace, known):
        add(operation_id, "transition_blocker", "transition_trace", 0)

    op_map = problem.operation_map()
    successors: dict[str, list[str]] = {operation_id: [] for operation_id in known}
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            successors.setdefault(predecessor, []).append(operation.id)

    if cfg.release_precedence_neighbors:
        frontier = list(members)
        for depth in range(1, cfg.max_closure_depth + 1):
            next_frontier: list[str] = []
            for operation_id in frontier:
                # Only downstream propagation is released automatically: moving an
                # operation may force its direct successor later.  Predecessors do
                # not move unless the proposal/transition trace names them.
                for neighbor in sorted(successors.get(operation_id, ())):
                    if neighbor not in members:
                        add(neighbor, "precedence_feasibility_neighbor", operation_id, depth)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

    if cfg.release_resource_neighbors and cfg.max_closure_depth > 0:
        mode_map = problem.mode_map()
        by_resource: dict[str, list[str]] = {}
        for assignment in sorted(schedule.assignments, key=lambda row: (row.start, row.operation_id)):
            for resource in mode_map[assignment.mode_id][1].resources:
                by_resource.setdefault(resource, []).append(assignment.operation_id)
        seeds = tuple(members)
        for operation_id in seeds:
            for sequence in by_resource.values():
                if operation_id not in sequence:
                    continue
                index = sequence.index(operation_id)
                for neighbor in sequence[max(0, index - 1):index + 2]:
                    if neighbor != operation_id:
                        add(neighbor, "resource_feasibility_neighbor", operation_id, 1)

    ordered = tuple(sorted(members.values(), key=lambda item: (item.depth, item.operation_id)))
    return InterventionClosure(
        operation_ids=tuple(item.operation_id for item in ordered),
        members=ordered,
        max_depth=max((item.depth for item in ordered), default=0),
        truncated=truncated,
    )


__all__ = [
    "ClosureMember", "ClosureState", "FrozenLocalCounterfactualConfig",
    "InterventionClosure", "build_intervention_closure",
    "load_frozen_local_counterfactual_config",
]
