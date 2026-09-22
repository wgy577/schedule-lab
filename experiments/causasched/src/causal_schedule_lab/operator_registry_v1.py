"""Deterministic finite operator/parameter candidates for the v1 policy.

The neural policy never invents an entity ID or unbounded parameter.  It
scores candidates generated here; LLM proposals must canonicalize to the same
finite set before any executor or Oracle call.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json

from .ir import Problem, Schedule
from .training_v1 import OperatorParameterCandidate, candidate_set_hash
from .unified_representation import (
    STANDARD_OPERATOR_CATALOG,
    build_operator_masks,
    derive_family_profile,
)


REGISTRY_VERSION = "standard-six-family-operators-1.0"
STOP_OPERATOR_ID = "stop"


def _candidate_id(operator_id: str, parameters: dict[str, object]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{operator_id}:{payload}".encode()).hexdigest()[:16]
    return f"{operator_id}:{digest}"


def _resource_sequences(problem: Problem, schedule: Schedule) -> dict[str, list]:
    result: dict[str, list] = defaultdict(list)
    mode_map = problem.mode_map()
    for assignment in schedule.assignments:
        for resource_id in mode_map[assignment.mode_id][1].resources:
            result[resource_id].append(assignment)
    for values in result.values():
        values.sort(key=lambda item: (item.start, item.end, item.operation_id))
    return result


def _default_cause_decisions(problem: Problem, root_operations: set[str]) -> set[str]:
    operation_map = problem.operation_map()
    decisions = {"resource_sequence", "start_time"}
    if any(len(operation_map[item].modes) > 1 for item in root_operations):
        decisions.update(("mode_selection", "resource_assignment"))
    if any(operation_map[item].stage_id is not None for item in root_operations):
        decisions.add("stage_sequence")
    return decisions


def _expand_neighborhood(
    problem: Problem,
    schedule: Schedule,
    root_operations: set[str],
    radius: int,
) -> set[str]:
    """Expand a root block to its scheduling neighborhood within ``radius``.

    The operator scope of a root block is not the block alone: a single-op block
    cannot swap with itself, so the operator's reach is widened to the block's
    neighborhood.  Two notions of "neighbor" are unioned, each within ``radius``
    hops:

    - resource-sequence neighbors: operations scheduled on the same machine
      sequence within ``radius`` positions of a root operation, and
    - precedence neighbors: the job's predecessor/successor operations within
      ``radius`` precedence hops.

    ``radius == 0`` returns the root block unchanged (the default behavior).
    """

    if radius <= 0 or not root_operations:
        return set(root_operations)
    operation_map = problem.operation_map()
    expanded = set(root_operations)

    # resource-sequence neighborhood (the "调度块" on a machine)
    sequences = _resource_sequences(problem, schedule)
    for sequence in sequences.values():
        ids = [item.operation_id for item in sequence]
        for index, operation_id in enumerate(ids):
            if operation_id not in expanded:
                continue
            for j in range(max(0, index - radius), min(len(ids), index + radius + 1)):
                expanded.add(ids[j])

    # precedence neighborhood (job path): successors are derived by inverting
    # the predecessor relation across all operations (the IR stores predecessors).
    successors: dict[str, list[str]] = defaultdict(list)
    for operation in operation_map.values():
        for predecessor in operation.predecessors:
            successors[predecessor].append(operation.id)
    for _ in range(radius):
        frontier: list[str] = []
        for operation_id in sorted(expanded):
            operation = operation_map[operation_id]
            frontier.extend(operation.predecessors)
            frontier.extend(successors.get(operation_id, ()))
        expanded.update(frontier)

    return expanded


def generate_operator_candidates(
    problem: Problem,
    schedule: Schedule,
    *,
    root_operations: tuple[str, ...],
    cause_decisions: tuple[str, ...] = (),
    decision_time: int = 0,
    maximum_per_operator: int = 64,
    neighborhood_radius: int = 0,
    subject_operations_only: bool = False,
) -> tuple[OperatorParameterCandidate, ...]:
    """Generate deterministic, quota-balanced parameter candidates.

    ``neighborhood_radius`` expands the root block to its scheduling
    neighborhood (see ``_expand_neighborhood``) so operators like swap / insert /
    resequence have legal targets beyond the block's own operations.  ``0`` keeps
    the block as-is (default).  A single-op block can never swap with itself; the
    expansion is what gives it a partner to swap with.
    """

    if maximum_per_operator <= 0:
        raise ValueError("maximum_per_operator must be positive")
    operation_map = problem.operation_map()
    unknown = set(root_operations) - set(operation_map)
    if unknown:
        raise ValueError(f"unknown root operations: {sorted(unknown)}")
    root = set(root_operations)
    # Operator scope is the block's neighborhood, not the block alone.
    scope = _expand_neighborhood(problem, schedule, root, neighborhood_radius)
    decisions = set(cause_decisions) or _default_cause_decisions(problem, scope)
    profile = derive_family_profile(problem)
    mask_entries = {
        item.operator_id: item
        for item in build_operator_masks(
            problem,
            schedule,
            profile,
            decision_time=decision_time,
            selected_operations=tuple(sorted(scope)),
        ).entries
    }
    specs = {item.id: item for item in STANDARD_OPERATOR_CATALOG}
    sequences = _resource_sequences(problem, schedule)
    assignment_map = schedule.assignment_map()
    mode_map = problem.mode_map()
    raw: dict[str, list[dict[str, object]]] = defaultdict(list)
    subjects = root if subject_operations_only else scope

    for resource_id, sequence in sorted(sequences.items()):
        for left, right in zip(sequence, sequence[1:]):
            pair = {left.operation_id, right.operation_id}
            if pair & subjects:
                raw["adjacent_resource_swap"].append(
                    {
                        "resource_id": resource_id,
                        "left_operation_id": left.operation_id,
                        "right_operation_id": right.operation_id,
                    }
                )
            if left.end == right.start and pair & subjects:
                raw["critical_block_resequence"].append(
                    {
                        "resource_id": resource_id,
                        "operation_ids": [left.operation_id, right.operation_id],
                    }
                )
        for operation_id in sorted(subjects):
            if operation_id not in assignment_map:
                continue
            selected_mode = mode_map[assignment_map[operation_id].mode_id][1]
            if resource_id not in selected_mode.resources:
                continue
            for position in range(len(sequence) + 1):
                predecessor = sequence[position - 1].operation_id if position else None
                successor = sequence[position].operation_id if position < len(sequence) else None
                if operation_id in {predecessor, successor}:
                    continue
                raw["resource_sequence_insertion"].append(
                    {
                        "operation_id": operation_id,
                        "resource_id": resource_id,
                        "position": position,
                        "predecessor_id": predecessor,
                        "successor_id": successor,
                    }
                )

    for operation_id in sorted(subjects):
        operation = operation_map[operation_id]
        current_mode = assignment_map.get(operation_id).mode_id if operation_id in assignment_map else None
        for mode in sorted(operation.modes, key=lambda item: item.id):
            if mode.id == current_mode:
                continue
            parameters = {
                "operation_id": operation_id,
                "mode_id": mode.id,
                "target_resource_ids": list(mode.resources),
            }
            raw["machine_reassignment"].append(parameters)
            if operation.stage_id is not None:
                raw["stage_machine_reassignment"].append(
                    {"stage_id": operation.stage_id, **parameters}
                )

    by_stage: dict[str, list[str]] = defaultdict(list)
    for operation_id in sorted(subjects):
        stage = operation_map[operation_id].stage_id
        if stage is not None:
            by_stage[stage].append(operation_id)
    for stage_id, operation_ids in sorted(by_stage.items()):
        ordered = sorted(
            operation_ids,
            key=lambda item: (
                assignment_map[item].start if item in assignment_map else 10**18,
                item,
            ),
        )
        for left, right in zip(ordered, ordered[1:]):
            raw["stage_resequence"].append(
                {"stage_id": stage_id, "left_operation_id": left, "right_operation_id": right}
            )

    arrived_jobs = {
        entity
        for event in problem.events
        if event.active and event.kind == "job_arrival" and event.time <= decision_time
        for entity in event.scope
    }
    for job_id in sorted(arrived_jobs):
        unscheduled = sorted(
            (
                item for item in problem.operations
                if item.job_id == job_id and item.id not in assignment_map
            ),
            key=lambda item: (item.index, item.id),
        )
        if unscheduled:
            first = unscheduled[0]
            for mode in sorted(first.modes, key=lambda item: item.id):
                raw["dynamic_job_insertion"].append(
                    {"job_id": job_id, "operation_id": first.id, "mode_id": mode.id}
                )

    broken_resources: set[str] = set()
    for event in sorted(problem.events, key=lambda item: (item.time, item.id)):
        if not event.active or event.time > decision_time:
            continue
        if event.kind == "resource_breakdown":
            broken_resources.update(event.scope)
        elif event.kind == "resource_recovery":
            broken_resources.difference_update(event.scope)
    for resource_id in sorted(broken_resources):
        affected = [
            item.operation_id for item in sequences.get(resource_id, ()) if item.end > decision_time
        ]
        if affected:
            raw["breakdown_reschedule"].append(
                {"resource_id": resource_id, "operation_ids": affected}
            )
    if problem.environment != "static" and scope:
        raw["dynamic_suffix_reschedule"].append(
            {"decision_time": decision_time, "operation_ids": sorted(scope)}
        )

    candidates: list[OperatorParameterCandidate] = []
    for operator_id, spec in sorted(specs.items()):
        mask = mask_entries[operator_id]
        cause_legal = bool(set(spec.modified_decisions) & decisions)
        parameters_seen: set[str] = set()
        ordered_parameters = sorted(
            raw.get(operator_id, ()),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
        for parameters in ordered_parameters:
            canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
            if canonical in parameters_seen:
                continue
            parameters_seen.add(canonical)
            candidates.append(
                OperatorParameterCandidate(
                    candidate_id=_candidate_id(operator_id, parameters),
                    operator_id=operator_id,
                    parameters=parameters,
                    semantic_legal=mask.static_legal,
                    local_legal=mask.dynamic_legal,
                    cause_legal=cause_legal,
                    parameter_legal=True,
                )
            )
            if len(parameters_seen) >= maximum_per_operator:
                break

    # Stop is legal only as a representable action; teacher labeling decides
    # whether exhaustive candidate evaluation justifies it as a positive target.
    stop_parameters: dict[str, object] = {}
    candidates.append(
        OperatorParameterCandidate(
            candidate_id=_candidate_id(STOP_OPERATOR_ID, stop_parameters),
            operator_id=STOP_OPERATOR_ID,
            parameters=stop_parameters,
            semantic_legal=True,
            local_legal=True,
            cause_legal=True,
            parameter_legal=True,
        )
    )
    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


def operator_masks_from_candidates(
    candidates: tuple[OperatorParameterCandidate, ...],
) -> dict[str, dict[str, bool]]:
    operators = tuple(sorted({item.operator_id for item in candidates}))
    result: dict[str, dict[str, bool]] = {}
    for field, attribute in (
        ("semantic", "semantic_legal"),
        ("local", "local_legal"),
        ("cause", "cause_legal"),
        ("final", "legal"),
    ):
        result[field] = {
            operator: any(bool(getattr(item, attribute)) for item in candidates if item.operator_id == operator)
            for operator in operators
        }
    return result


__all__ = [
    "REGISTRY_VERSION",
    "STOP_OPERATOR_ID",
    "candidate_set_hash",
    "generate_operator_candidates",
    "operator_masks_from_candidates",
]
