"""Unified, serializable representation for six standard shop families.

The module deliberately stops at JSP, FSP, FJSP, HFSP, DJSP and DFJSP.  It
combines the canonical problem/schedule IR and UTSEG with deterministic
capability labels, state feasibility masks and operator applicability masks.
Reward design and policy training do not belong here.
"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .graph import build_utseg
from .ir import Problem, Schedule
from .models import SchedulingGraph


StandardFamily = Literal["JSP", "FSP", "FJSP", "HFSP", "DJSP", "DFJSP"]
SUPPORTED_FAMILIES: tuple[str, ...] = (
    "JSP",
    "FSP",
    "FJSP",
    "HFSP",
    "DJSP",
    "DFJSP",
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class CapabilitySignature(FrozenModel):
    """Multi-label structural signature derived from the canonical problem."""

    fixed_machine_assignments: bool
    alternative_machines: bool
    common_stage_route: bool
    job_specific_route: bool
    parallel_machines_per_stage: bool
    dynamic_events: bool
    dynamic_job_arrivals: bool
    machine_breakdowns: bool
    processing_time_updates: bool
    rescheduling_required: bool


class FamilyProfile(FrozenModel):
    declared_family: StandardFamily
    inferred_family: StandardFamily
    consistent: bool
    capabilities: CapabilitySignature
    evidence: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class InsertionPosition(FrozenModel):
    operation_id: str
    mode_id: str
    resource_id: str
    position: int = Field(ge=0)
    earliest_start: int = Field(ge=0)
    latest_start: int | None = Field(default=None, ge=0)


class FeasibilityMaskSnapshot(FrozenModel):
    """A deterministic state snapshot, not an enumeration of all schedules."""

    decision_time: int = Field(ge=0)
    operation_ids: tuple[str, ...]
    mode_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    released_operations: tuple[str, ...]
    frozen_operation_mask: dict[str, bool]
    ready_operation_mask: dict[str, bool]
    eligible_mode_mask: dict[str, dict[str, bool]]
    eligible_resource_mask: dict[str, dict[str, bool]]
    effective_resource_mask: dict[str, dict[str, bool]]
    resource_available_mask: dict[str, bool]
    arrived_job_mask: dict[str, bool]
    completed_operation_mask: dict[str, bool]
    running_operation_mask: dict[str, bool]
    unscheduled_operation_mask: dict[str, bool]
    active_event_ids: tuple[str, ...]
    active_breakdown_resources: tuple[str, ...]
    insertion_positions: tuple[InsertionPosition, ...]


class OperatorSpec(FrozenModel):
    id: str
    description: str
    family_support: tuple[StandardFamily, ...]
    required_capabilities: tuple[str, ...] = ()
    any_capabilities: tuple[str, ...] = ()
    modified_decisions: tuple[str, ...]
    preserves: tuple[str, ...]
    scope: str


class OperatorApplicability(FrozenModel):
    operator_id: str
    static_legal: bool
    dynamic_legal: bool
    legal: bool
    reasons: tuple[str, ...] = ()


class OperatorMaskSnapshot(FrozenModel):
    selected_operations: tuple[str, ...]
    entries: tuple[OperatorApplicability, ...]

    @computed_field
    @property
    def mask(self) -> dict[str, bool]:
        """Final state-dependent operator mask serialized with the snapshot."""

        return {item.operator_id: item.legal for item in self.entries}


class UnifiedSchedulingRepresentation(FrozenModel):
    """Lossless problem/schedule payload plus derived learning-side labels."""

    schema_version: str = "six-family-1.0"
    problem: Problem
    schedule: Schedule
    graph: SchedulingGraph
    family: FamilyProfile
    feasibility: FeasibilityMaskSnapshot
    operators: OperatorMaskSnapshot


def order_schedule_for_serialization(
    schedule: Schedule,
    *,
    tie_break_seed: int = 0,
) -> Schedule:
    """Order assignments by time and reproducibly shuffle exact-time ties.

    The JSON contract is start time ascending, then end time ascending. Items
    with identical start and end times have no temporal ordering, so a local
    seeded RNG shuffles them. Sorting each group by operation ID first makes
    the result independent of the source tuple order.
    """

    groups: dict[tuple[int, int], list] = defaultdict(list)
    for assignment in schedule.assignments:
        groups[(assignment.start, assignment.end)].append(assignment)
    rng = random.Random(tie_break_seed)
    ordered = []
    for interval in sorted(groups):
        group = sorted(groups[interval], key=lambda item: item.operation_id)
        rng.shuffle(group)
        ordered.extend(group)
    return schedule.model_copy(
        update={
            "assignments": tuple(ordered),
            "metadata": {
                **schedule.metadata,
                "assignment_order": {
                    "primary": "start_ascending",
                    "secondary": "end_ascending",
                    "exact_tie": "seeded_random",
                    "seed": tie_break_seed,
                },
            },
        }
    )


STANDARD_OPERATOR_CATALOG: tuple[OperatorSpec, ...] = (
    OperatorSpec(
        id="adjacent_resource_swap",
        description="Swap two adjacent operations on one resource sequence.",
        family_support=SUPPORTED_FAMILIES,
        modified_decisions=("resource_sequence",),
        preserves=("mode_selection", "processing_duration"),
        scope="two_operations_one_resource",
    ),
    OperatorSpec(
        id="resource_sequence_insertion",
        description="Remove one operation and insert it at another legal resource position.",
        family_support=SUPPORTED_FAMILIES,
        modified_decisions=("resource_sequence", "start_time"),
        preserves=("mode_selection",),
        scope="one_operation_one_resource_sequence",
    ),
    OperatorSpec(
        id="critical_block_resequence",
        description="Resequence a contiguous zero-gap block on a capacity-one resource.",
        family_support=SUPPORTED_FAMILIES,
        modified_decisions=("resource_sequence",),
        preserves=("mode_selection",),
        scope="critical_resource_block",
    ),
    OperatorSpec(
        id="machine_reassignment",
        description="Change the selected processing mode to another eligible machine.",
        family_support=("FJSP", "HFSP", "DFJSP"),
        required_capabilities=("alternative_machines",),
        modified_decisions=("mode_selection", "resource_assignment"),
        preserves=("job_precedence",),
        scope="one_operation",
    ),
    OperatorSpec(
        id="stage_resequence",
        description="Change job order within one shared flow-shop stage.",
        family_support=("FSP", "HFSP"),
        required_capabilities=("common_stage_route",),
        modified_decisions=("stage_sequence",),
        preserves=("stage_route",),
        scope="one_stage_local_block",
    ),
    OperatorSpec(
        id="stage_machine_reassignment",
        description="Move an operation between parallel machines of the same stage.",
        family_support=("HFSP",),
        required_capabilities=("common_stage_route", "parallel_machines_per_stage"),
        modified_decisions=("mode_selection", "resource_assignment"),
        preserves=("stage_route", "job_precedence"),
        scope="one_stage_one_operation",
    ),
    OperatorSpec(
        id="dynamic_job_insertion",
        description="Insert an arrived but unscheduled job into the current schedule.",
        family_support=("DJSP", "DFJSP"),
        required_capabilities=("dynamic_job_arrivals",),
        modified_decisions=("resource_sequence", "start_time"),
        preserves=("completed_prefix",),
        scope="arrived_job_suffix",
    ),
    OperatorSpec(
        id="breakdown_reschedule",
        description="Repair operations affected by an active machine breakdown.",
        family_support=("DJSP", "DFJSP"),
        required_capabilities=("machine_breakdowns",),
        modified_decisions=("resource_assignment", "resource_sequence", "start_time"),
        preserves=("completed_prefix",),
        scope="affected_resource_suffix",
    ),
    OperatorSpec(
        id="dynamic_suffix_reschedule",
        description="Reschedule the non-frozen suffix after an active dynamic event.",
        family_support=("DJSP", "DFJSP"),
        required_capabilities=("dynamic_events",),
        modified_decisions=("mode_selection", "resource_sequence", "start_time"),
        preserves=("completed_prefix",),
        scope="event_affected_suffix",
    ),
)


def _distinct_resource_options(operation) -> set[tuple[str, ...]]:
    return {tuple(sorted(mode.resources)) for mode in operation.modes}


def derive_family_profile(problem: Problem) -> FamilyProfile:
    if problem.kind not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"six-family representation does not support {problem.kind!r}"
        )
    flexible_operations = tuple(
        operation.id
        for operation in problem.operations
        if len(_distinct_resource_options(operation)) > 1
    )
    by_job: dict[str, list] = defaultdict(list)
    for operation in problem.operations:
        by_job[operation.job_id].append(operation)
    stage_routes = tuple(
        tuple(
            operation.stage_id
            for operation in sorted(operations, key=lambda item: item.index)
        )
        for _, operations in sorted(by_job.items())
    )
    common_stage_route = bool(stage_routes) and all(
        route == stage_routes[0] and all(stage is not None for stage in route)
        for route in stage_routes
    )
    resources_by_stage: dict[str, set[str]] = defaultdict(set)
    for operation in problem.operations:
        if operation.stage_id is None:
            continue
        for mode in operation.modes:
            resources_by_stage[operation.stage_id].update(mode.resources)
    parallel_stages = tuple(
        sorted(stage for stage, resources in resources_by_stage.items() if len(resources) > 1)
    )
    event_kinds = {event.kind for event in problem.events if event.active}
    dynamic_events = problem.environment != "static" or bool(event_kinds)
    capabilities = CapabilitySignature(
        fixed_machine_assignments=not flexible_operations,
        alternative_machines=bool(flexible_operations),
        common_stage_route=common_stage_route,
        job_specific_route=not common_stage_route,
        parallel_machines_per_stage=bool(parallel_stages),
        dynamic_events=dynamic_events,
        dynamic_job_arrivals="job_arrival" in event_kinds,
        machine_breakdowns=bool(
            {"resource_breakdown", "resource_recovery"} & event_kinds
        ),
        processing_time_updates="processing_time_update" in event_kinds,
        rescheduling_required=dynamic_events,
    )
    if dynamic_events:
        inferred: StandardFamily = "DFJSP" if flexible_operations else "DJSP"
    elif common_stage_route:
        inferred = "HFSP" if flexible_operations else "FSP"
    else:
        inferred = "FJSP" if flexible_operations else "JSP"
    return FamilyProfile(
        declared_family=problem.kind,
        inferred_family=inferred,
        consistent=problem.kind == inferred,
        capabilities=capabilities,
        evidence={
            "alternative_machines": flexible_operations,
            "common_stage_route": tuple(
                stage for stage in (stage_routes[0] if common_stage_route else ()) if stage
            ),
            "parallel_machines_per_stage": parallel_stages,
            "dynamic_events": tuple(sorted(event_kinds)),
        },
    )


def _active_broken_resources(problem: Problem, decision_time: int) -> set[str]:
    state: dict[str, bool] = {}
    events = sorted(
        (event for event in problem.events if event.active and event.time <= decision_time),
        key=lambda item: (item.time, item.id),
    )
    for event in events:
        if event.kind not in {"resource_breakdown", "resource_recovery"}:
            continue
        broken = event.kind == "resource_breakdown"
        for entity in event.scope:
            if entity in problem.resource_map():
                state[entity] = broken
    return {resource for resource, broken in state.items() if broken}


def _interval_in_calendar(resource, start: int, end: int) -> bool:
    return not resource.calendar or any(
        start >= left and end <= right for left, right in resource.calendar
    )


def _mode_has_future_calendar_window(problem: Problem, mode, decision_time: int) -> bool:
    """Return whether all mode resources have a provable future calendar window.

    This is deliberately only a calendar/breakdown eligibility check. Resource
    occupancy remains represented separately by ``resource_available_mask`` and
    insertion positions; opaque project constraints are never guessed here.
    """

    resource_map = problem.resource_map()
    for resource_id in mode.resources:
        resource = resource_map[resource_id]
        if resource.calendar and not any(
            max(left, decision_time) + mode.duration <= right
            for left, right in resource.calendar
        ):
            return False
    return True


def build_feasibility_masks(
    problem: Problem,
    schedule: Schedule,
    *,
    decision_time: int = 0,
    released_operations: tuple[str, ...] = (),
) -> FeasibilityMaskSnapshot:
    """Materialize the current legal action surface for solver/model use.

    ``released_operations`` are removed from the incumbent overlay before the
    masks are built.  This gives a local-repair state while preserving the
    complete incumbent in ``schedule``.
    """

    if decision_time < 0:
        raise ValueError("decision_time must be non-negative")
    operation_map = problem.operation_map()
    unknown = set(released_operations) - set(operation_map)
    if unknown:
        raise ValueError(f"unknown released operations: {sorted(unknown)}")
    released = set(released_operations)
    all_operation_ids = tuple(sorted(operation_map))
    incumbent = schedule.assignment_map()
    active_assignments = {
        operation_id: assignment
        for operation_id, assignment in incumbent.items()
        if operation_id not in released
    }
    jobs = problem.job_map()
    arrived_jobs = {
        job.id for job in problem.jobs if job.release <= decision_time
    }
    jobs_with_arrival_events = {
        entity
        for event in problem.events
        if event.active and event.kind == "job_arrival"
        for entity in event.scope
        if entity in jobs
    }
    arrived_jobs.difference_update(jobs_with_arrival_events)
    for event in problem.events:
        if event.active and event.kind == "job_arrival" and event.time <= decision_time:
            arrived_jobs.update(
                entity
                for entity in event.scope
                if entity in jobs and jobs[entity].release <= decision_time
            )
    if problem.environment == "static":
        arrived_jobs = set(jobs)
    active_events = tuple(
        sorted(
            (
                event
                for event in problem.events
                if event.active and event.time <= decision_time
            ),
            key=lambda item: (item.time, item.id),
        )
    )
    ready: dict[str, bool] = {}
    for operation_id in all_operation_ids:
        operation = operation_map[operation_id]
        exposed = not released or operation_id in released
        ready[operation_id] = bool(
            exposed
            and operation_id not in active_assignments
            and operation.job_id in arrived_jobs
            and max(operation.release, jobs[operation.job_id].release) <= decision_time
            and all(item in active_assignments for item in operation.predecessors)
        )

    resource_ids = tuple(sorted(item.id for item in problem.resources))
    mode_ids = tuple(
        sorted(mode.id for operation in problem.operations for mode in operation.modes)
    )
    eligible_modes: dict[str, dict[str, bool]] = {}
    eligible_resources: dict[str, dict[str, bool]] = {}
    for operation_id in all_operation_ids:
        operation = operation_map[operation_id]
        operation_mode_ids = {mode.id for mode in operation.modes}
        operation_resource_ids = {
            resource for mode in operation.modes for resource in mode.resources
        }
        eligible_modes[operation_id] = {
            mode_id: mode_id in operation_mode_ids for mode_id in mode_ids
        }
        eligible_resources[operation_id] = {
            resource_id: resource_id in operation_resource_ids
            for resource_id in resource_ids
        }

    broken = _active_broken_resources(problem, decision_time)
    resource_map = problem.resource_map()
    by_resource: dict[str, list] = defaultdict(list)
    mode_map = problem.mode_map()
    for assignment in active_assignments.values():
        _, mode = mode_map[assignment.mode_id]
        for resource_id in mode.resources:
            by_resource[resource_id].append(assignment)
    for assignments in by_resource.values():
        assignments.sort(key=lambda item: (item.start, item.end, item.operation_id))
    resource_available = {}
    for resource_id in resource_ids:
        resource = resource_map[resource_id]
        active_count = sum(
            item.start <= decision_time < item.end
            for item in by_resource.get(resource_id, ())
        )
        calendar_open = not resource.calendar or any(
            left <= decision_time < right for left, right in resource.calendar
        )
        resource_available[resource_id] = bool(
            resource_id not in broken
            and calendar_open
            and active_count < resource.capacity
        )

    completed = {
        operation_id: bool(
            operation_id in incumbent
            and incumbent[operation_id].end <= decision_time
        )
        for operation_id in all_operation_ids
    }
    running = {
        operation_id: bool(
            operation_id in incumbent
            and incumbent[operation_id].start <= decision_time
            < incumbent[operation_id].end
        )
        for operation_id in all_operation_ids
    }
    unscheduled = {
        operation_id: operation_id not in incumbent
        for operation_id in all_operation_ids
    }
    frozen = {
        operation_id: bool(
            operation_id in incumbent and operation_id not in released
        )
        for operation_id in all_operation_ids
    }

    effective_resources: dict[str, dict[str, bool]] = {}
    for operation_id in all_operation_ids:
        operation = operation_map[operation_id]
        assignment = incumbent.get(operation_id)
        selected_resources: set[str] = set()
        if assignment is not None and assignment.mode_id in mode_map:
            selected_resources.update(mode_map[assignment.mode_id][1].resources)
        operation_is_frozen = frozen[operation_id]
        effective_modes = tuple(
            mode
            for mode in operation.modes
            if not (set(mode.resources) & broken)
            and _mode_has_future_calendar_window(problem, mode, decision_time)
            and (
                not operation_is_frozen
                or assignment is None
                or set(mode.resources) == selected_resources
            )
        )
        effective_resources[operation_id] = {
            resource_id: any(
                resource_id in mode.resources for mode in effective_modes
            )
            for resource_id in resource_ids
        }

    positions: list[InsertionPosition] = []
    for operation_id in all_operation_ids:
        if not ready[operation_id]:
            continue
        operation = operation_map[operation_id]
        predecessor_end = max(
            (active_assignments[item].end for item in operation.predecessors),
            default=0,
        )
        release = max(
            decision_time,
            operation.release,
            jobs[operation.job_id].release,
            predecessor_end,
        )
        for mode in operation.modes:
            # The six standard families use one processing resource per mode.
            # Multi-resource extensions remain representable in IR but do not
            # receive a misleading single-machine insertion mask here.
            if len(mode.resources) != 1:
                continue
            resource_id = mode.resources[0]
            if resource_id in broken:
                continue
            resource = resource_map[resource_id]
            if resource.capacity != 1:
                continue
            sequence = by_resource.get(resource_id, [])
            for position in range(len(sequence) + 1):
                previous_end = sequence[position - 1].end if position else 0
                next_start = sequence[position].start if position < len(sequence) else None
                earliest = max(release, previous_end)
                end = earliest + mode.duration
                if next_start is not None and end > next_start:
                    continue
                if not _interval_in_calendar(resource, earliest, end):
                    continue
                positions.append(
                    InsertionPosition(
                        operation_id=operation_id,
                        mode_id=mode.id,
                        resource_id=resource_id,
                        position=position,
                        earliest_start=earliest,
                        latest_start=(
                            None if next_start is None else next_start - mode.duration
                        ),
                    )
                )

    return FeasibilityMaskSnapshot(
        decision_time=decision_time,
        operation_ids=all_operation_ids,
        mode_ids=mode_ids,
        resource_ids=resource_ids,
        released_operations=tuple(sorted(released)),
        frozen_operation_mask=frozen,
        ready_operation_mask=ready,
        eligible_mode_mask=eligible_modes,
        eligible_resource_mask=eligible_resources,
        effective_resource_mask=effective_resources,
        resource_available_mask=resource_available,
        arrived_job_mask={job_id: job_id in arrived_jobs for job_id in sorted(jobs)},
        completed_operation_mask=completed,
        running_operation_mask=running,
        unscheduled_operation_mask=unscheduled,
        active_event_ids=tuple(event.id for event in active_events),
        active_breakdown_resources=tuple(sorted(broken)),
        insertion_positions=tuple(
            sorted(
                positions,
                key=lambda item: (
                    item.operation_id,
                    item.mode_id,
                    item.resource_id,
                    item.position,
                ),
            )
        ),
    )


def _resource_sequences(problem: Problem, schedule: Schedule) -> dict[str, list]:
    result: dict[str, list] = defaultdict(list)
    modes = problem.mode_map()
    for assignment in schedule.assignments:
        selected = modes.get(assignment.mode_id)
        if selected is None:
            continue
        for resource_id in selected[1].resources:
            result[resource_id].append(assignment)
    for values in result.values():
        values.sort(key=lambda item: (item.start, item.end, item.operation_id))
    return result


def build_operator_masks(
    problem: Problem,
    schedule: Schedule,
    family: FamilyProfile,
    *,
    decision_time: int = 0,
    selected_operations: tuple[str, ...] = (),
) -> OperatorMaskSnapshot:
    """Combine family/constraint labels with current-Gantt preconditions."""

    operation_map = problem.operation_map()
    unknown = set(selected_operations) - set(operation_map)
    if unknown:
        raise ValueError(f"unknown selected operations: {sorted(unknown)}")
    selected = set(selected_operations) or set(operation_map)
    sequences = _resource_sequences(problem, schedule)
    assignments = schedule.assignment_map()
    capabilities = family.capabilities.model_dump()
    active_events = tuple(
        event
        for event in problem.events
        if event.active and event.time <= decision_time
    )
    broken = _active_broken_resources(problem, decision_time)

    entries: list[OperatorApplicability] = []
    for spec in STANDARD_OPERATOR_CATALOG:
        reasons: list[str] = []
        static_legal = family.declared_family in spec.family_support
        if not static_legal:
            reasons.append(f"family:{family.declared_family}:unsupported")
        for capability in spec.required_capabilities:
            if not bool(capabilities[capability]):
                static_legal = False
                reasons.append(f"missing_capability:{capability}")
        if spec.any_capabilities and not any(
            bool(capabilities[item]) for item in spec.any_capabilities
        ):
            static_legal = False
            reasons.append("missing_any_capability:" + ",".join(spec.any_capabilities))

        dynamic_legal = False
        if spec.id == "adjacent_resource_swap":
            dynamic_legal = any(
                sum(item.operation_id in selected for item in values) >= 2
                for values in sequences.values()
            )
        elif spec.id == "resource_sequence_insertion":
            dynamic_legal = any(
                operation_id in assignments
                and any(
                    len(sequences.get(resource_id, ())) >= 2
                    for mode in operation_map[operation_id].modes
                    for resource_id in mode.resources
                )
                for operation_id in selected
            )
        elif spec.id == "critical_block_resequence":
            dynamic_legal = any(
                any(left.end == right.start for left, right in zip(values, values[1:]))
                and sum(item.operation_id in selected for item in values) >= 2
                for values in sequences.values()
            )
        elif spec.id == "machine_reassignment":
            dynamic_legal = any(
                len(_distinct_resource_options(operation_map[item])) > 1
                for item in selected
            )
        elif spec.id == "stage_resequence":
            by_stage: dict[str, int] = defaultdict(int)
            for operation_id in selected:
                stage_id = operation_map[operation_id].stage_id
                if stage_id is not None:
                    by_stage[stage_id] += 1
            dynamic_legal = any(count >= 2 for count in by_stage.values())
        elif spec.id == "stage_machine_reassignment":
            dynamic_legal = any(
                operation_map[item].stage_id is not None
                and len(_distinct_resource_options(operation_map[item])) > 1
                for item in selected
            )
        elif spec.id == "dynamic_job_insertion":
            arrived = {
                entity
                for event in active_events
                if event.kind == "job_arrival"
                for entity in event.scope
            }
            dynamic_legal = any(
                operation.job_id in arrived and operation.id not in assignments
                for operation in problem.operations
            )
        elif spec.id == "breakdown_reschedule":
            dynamic_legal = any(
                resource in broken
                and any(item.end > decision_time for item in sequences.get(resource, ()))
                for resource in broken
            )
        elif spec.id == "dynamic_suffix_reschedule":
            dynamic_legal = bool(active_events)
        if not dynamic_legal:
            reasons.append("no_current_gantt_candidate")
        entries.append(
            OperatorApplicability(
                operator_id=spec.id,
                static_legal=static_legal,
                dynamic_legal=dynamic_legal,
                legal=static_legal and dynamic_legal,
                reasons=tuple(reasons),
            )
        )
    return OperatorMaskSnapshot(
        selected_operations=tuple(sorted(selected)),
        entries=tuple(entries),
    )


def build_unified_representation(
    problem: Problem,
    schedule: Schedule,
    *,
    project_id: str | None = None,
    decision_time: int = 0,
    released_operations: tuple[str, ...] = (),
    selected_operations: tuple[str, ...] = (),
    assignment_order_seed: int = 0,
) -> UnifiedSchedulingRepresentation:
    """Build the complete six-family payload used by downstream small models."""

    ordered_schedule = order_schedule_for_serialization(
        schedule,
        tie_break_seed=assignment_order_seed,
    )
    family = derive_family_profile(problem)
    feasibility = build_feasibility_masks(
        problem,
        ordered_schedule,
        decision_time=decision_time,
        released_operations=released_operations,
    )
    operators = build_operator_masks(
        problem,
        ordered_schedule,
        family,
        decision_time=decision_time,
        selected_operations=selected_operations or released_operations,
    )
    return UnifiedSchedulingRepresentation(
        problem=problem,
        schedule=ordered_schedule,
        graph=build_utseg(
            problem,
            ordered_schedule,
            project_id=project_id or f"six-family:{problem.id}",
        ),
        family=family,
        feasibility=feasibility,
        operators=operators,
    )
