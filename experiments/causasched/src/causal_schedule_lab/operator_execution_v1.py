"""Deterministic executor for the finite first-version operator candidates.

The registry is only a proposal space.  This module is the authority that
turns a supported candidate into explicit CP-SAT restrictions, validates the
result, invokes an optional domain Oracle, and checks deterministic replay.
Unsupported candidates fail closed instead of silently behaving like an
unconditioned repair.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from .core_validation import validate_schedule
from .ir import Problem, Schedule
from .llm_operator_teacher import CandidateEvaluation
from .objective import compare_objectives, evaluate_objective
from .repair import GeneratedCandidate
from .solvers.cp_sat import CPSATResult, solve_cp_sat
from .training_v1 import OperatorParameterCandidate, OperatorPolicyState
from .validation import schedule_hash


DomainOracle = Callable[[Problem, Schedule, GeneratedCandidate], dict[str, Any]]


EXECUTOR_VERSION = "finite-candidate-cpsat-executor-1.0"


@dataclass(frozen=True)
class OperatorExecutionPlan:
    released_operations: tuple[str, ...]
    frozen_operations: tuple[str, ...]
    forced_mode_ids: dict[str, str]
    enforced_orderings: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ExecutedOperatorCandidate:
    evaluation: CandidateEvaluation
    schedule: Schedule | None
    candidate_hash: str | None
    replay_hash: str | None
    executor_version: str
    solver_status: str
    details: dict[str, Any]


def _operation_parameters(candidate: OperatorParameterCandidate) -> set[str]:
    result: set[str] = set()
    for key, value in candidate.parameters.items():
        if key.endswith("operation_id") and isinstance(value, str):
            result.add(value)
        elif key.endswith("operation_ids") and isinstance(value, list):
            result.update(item for item in value if isinstance(item, str))
    return result


def _job_successor_closure(problem: Problem, seeds: set[str]) -> set[str]:
    operation_map = problem.operation_map()
    released = set(seeds)
    changed = True
    while changed:
        changed = False
        for operation in problem.operations:
            if operation.id in released:
                continue
            if any(item in released for item in operation.predecessors):
                released.add(operation.id)
                changed = True
        # A missing/implicit predecessor list must not let a job suffix remain
        # frozen behind a moved operation.
        for seed in tuple(released):
            source = operation_map.get(seed)
            if source is None:
                continue
            for operation in problem.operations:
                if operation.job_id == source.job_id and operation.index >= source.index:
                    released.add(operation.id)
    return released


def _target_resource_neighborhood(
    problem: Problem,
    schedule: Schedule,
    target_resource_ids,
    anchor_start: int,
    anchor_end: int,
    radius: int,
) -> set[str]:
    """Operations on the target machine(s) temporally near the moved slot.

    A ``machine_reassignment`` forces an operation onto a new resource, but the
    target machine's existing operations are normally frozen -- so the moved op
    has nowhere to fit and gets pushed late (or lands at the same makespan on an
    equal-speed machine).  Releasing the target machine's operations that occupy
    the band around the moved op's original ``[anchor_start, anchor_end]`` slot
    (plus ``radius`` sequence neighbours on each side) lets CP-SAT reshuffle the
    target machine to make room.  ``radius == 0`` releases only the operations
    whose interval overlaps the anchor slot.
    """
    if radius < 0 or anchor_end <= anchor_start:
        return set()
    mode_map = problem.mode_map()
    assignment_map = schedule.assignment_map()
    neighborhood: set[str] = set()
    for resource_id in target_resource_ids:
        sequence = sorted(
            (
                assignment for assignment in schedule.assignments
                if resource_id in mode_map[assignment.mode_id][1].resources
            ),
            key=lambda item: (item.start, item.end, item.operation_id),
        )
        if not sequence:
            continue
        # Indices of operations whose interval intersects the anchor slot.
        overlapping = [
            index for index, item in enumerate(sequence)
            if not (item.end <= anchor_start or item.start >= anchor_end)
        ]
        if overlapping:
            band = range(min(overlapping), max(overlapping) + 1)
        else:
            # Slot falls in a gap: take the two ops bracketing the gap, if any.
            anchor_mid = (anchor_start + anchor_end) // 2
            below = max(
                (i for i, item in enumerate(sequence) if item.end <= anchor_mid),
                default=None,
            )
            above = min(
                (i for i, item in enumerate(sequence) if item.start >= anchor_mid),
                default=None,
            )
            band = [i for i in (below, above) if i is not None]
        if not band:
            continue
        lo = max(0, min(band) - radius)
        hi = min(len(sequence), max(band) + radius + 1)
        for index in range(lo, hi):
            neighborhood.add(sequence[index].operation_id)
    return neighborhood


def build_operator_execution_plan(
    problem: Problem,
    schedule: Schedule,
    state: OperatorPolicyState,
    candidate: OperatorParameterCandidate,
    *,
    reassignment_release_radius: int = 2,
) -> OperatorExecutionPlan:
    """Compile a finite candidate into solver-enforced decisions.

    ``reassignment_release_radius`` only affects ``machine_reassignment`` /
    ``stage_machine_reassignment``: besides the moved op's job-successor closure,
    it releases the target machine's operations temporally adjacent to the
    moved op's original slot (``radius`` sequence neighbours on each side) so
    CP-SAT can reshuffle the target machine to make room.  ``0`` recovers the
    old behaviour of freezing the entire target machine.
    """

    if not candidate.legal:
        raise ValueError("illegal candidate cannot be executed")
    if candidate.candidate_id not in {item.candidate_id for item in state.parameter_candidates}:
        raise ValueError("candidate is not part of the state's frozen candidate set")
    if candidate.operator_id == "stop":
        raise ValueError("stop is a terminal decision, not an executable repair")

    operation_map = problem.operation_map()
    assignment_map = schedule.assignment_map()
    seeds = set(state.root_cause_node_ids) & set(operation_map)
    seeds.update(_operation_parameters(candidate) & set(operation_map))
    forced_modes: dict[str, str] = {}
    orderings: list[tuple[str, str]] = []
    parameters = candidate.parameters

    if candidate.operator_id == "adjacent_resource_swap":
        left = str(parameters["left_operation_id"])
        right = str(parameters["right_operation_id"])
        orderings.append((right, left))
    elif candidate.operator_id == "resource_sequence_insertion":
        operation_id = str(parameters["operation_id"])
        predecessor = parameters.get("predecessor_id")
        successor = parameters.get("successor_id")
        if isinstance(predecessor, str):
            orderings.append((predecessor, operation_id))
            seeds.add(predecessor)
        if isinstance(successor, str):
            orderings.append((operation_id, successor))
            seeds.add(successor)
    elif candidate.operator_id == "critical_block_resequence":
        operations = [str(item) for item in parameters.get("operation_ids", [])]
        orderings.extend(zip(reversed(operations[1:]), reversed(operations[:-1])))
    elif candidate.operator_id == "stage_resequence":
        left = str(parameters["left_operation_id"])
        right = str(parameters["right_operation_id"])
        orderings.append((right, left))
    elif candidate.operator_id in {"machine_reassignment", "stage_machine_reassignment"}:
        operation_id = str(parameters["operation_id"])
        forced_modes[operation_id] = str(parameters["mode_id"])
        # Release the target machine's operations around the moved op's original
        # slot so CP-SAT can reshuffle the target machine (otherwise the moved op
        # is squeezed into a frozen sequence and can only land late / equal).
        target_resource_ids = parameters.get("target_resource_ids") or ()
        incumbent_assignment = assignment_map.get(operation_id)
        if target_resource_ids and incumbent_assignment is not None:
            seeds.update(
                _target_resource_neighborhood(
                    problem,
                    schedule,
                    target_resource_ids,
                    incumbent_assignment.start,
                    incumbent_assignment.end,
                    reassignment_release_radius,
                )
            )
    elif candidate.operator_id in {"dynamic_job_insertion", "dynamic_suffix_reschedule"}:
        # The finite candidate names the released dynamic suffix.  The Problem
        # event/constraint encoding remains the source of feasibility truth.
        pass
    elif candidate.operator_id == "breakdown_reschedule":
        raise NotImplementedError(
            "breakdown_reschedule requires a typed unavailable-resource solver constraint"
        )
    else:
        raise NotImplementedError(f"operator {candidate.operator_id!r} has no V1 executor")

    for left, right in orderings:
        if left not in operation_map or right not in operation_map:
            raise ValueError("candidate ordering references an unknown operation")
        seeds.update((left, right))
    for operation_id, mode_id in forced_modes.items():
        if operation_id not in operation_map:
            raise ValueError("candidate mode override references an unknown operation")
        if mode_id not in {item.id for item in operation_map[operation_id].modes}:
            raise ValueError("candidate mode override is not eligible")
        if assignment_map.get(operation_id) is None:
            raise ValueError("candidate mode override requires an incumbent assignment")

    released = _job_successor_closure(problem, seeds)
    all_operations = {item.id for item in problem.operations}
    return OperatorExecutionPlan(
        released_operations=tuple(sorted(released)),
        frozen_operations=tuple(sorted(all_operations - released)),
        forced_mode_ids=forced_modes,
        enforced_orderings=tuple(orderings),
    )


class DeterministicOperatorExecutor:
    def __init__(
        self,
        *,
        seed: int = 0,
        deterministic_time: float = 1.0,
        stability_weight: int = 1,
        domain_oracle: DomainOracle | None = None,
        oracle_id: str = "generic-core-validator",
        reassignment_release_radius: int = 2,
    ) -> None:
        self.seed = seed
        self.deterministic_time = deterministic_time
        self.stability_weight = stability_weight
        self.domain_oracle = domain_oracle
        self.oracle_id = oracle_id
        self.reassignment_release_radius = reassignment_release_radius

    def _solve(
        self,
        problem: Problem,
        incumbent: Schedule,
        plan: OperatorExecutionPlan,
    ) -> CPSATResult:
        return solve_cp_sat(
            problem,
            incumbent=incumbent,
            frozen_operation_ids=set(plan.frozen_operations),
            forced_mode_ids=plan.forced_mode_ids,
            enforced_orderings=plan.enforced_orderings,
            seed=self.seed,
            workers=1,
            max_deterministic_time=self.deterministic_time,
            stability_weight=self.stability_weight,
        )

    def execute(
        self,
        problem: Problem,
        incumbent: Schedule,
        state: OperatorPolicyState,
        candidate: OperatorParameterCandidate,
    ) -> ExecutedOperatorCandidate:
        started = time.perf_counter()
        before = evaluate_objective(problem, incumbent, baseline=incumbent)
        try:
            plan = build_operator_execution_plan(
                problem, incumbent, state, candidate,
                reassignment_release_radius=self.reassignment_release_radius,
            )
        except (ValueError, NotImplementedError, KeyError) as error:
            return ExecutedOperatorCandidate(
                evaluation=CandidateEvaluation(
                    static_pass=False,
                    light_pass=False,
                    full_pass=False,
                    deterministic_replay_pass=False,
                    full_feasible_pass=False,
                    objective_before=before.values[0],
                    runtime_seconds=time.perf_counter() - started,
                    failure_labels=(f"EXECUTOR_REJECTED:{type(error).__name__}",),
                ),
                schedule=None,
                candidate_hash=None,
                replay_hash=None,
                executor_version=EXECUTOR_VERSION,
                solver_status="REJECTED",
                details={"reason": str(error), "oracleId": self.oracle_id},
            )

        first = self._solve(problem, incumbent, plan)
        if first.schedule is None:
            return ExecutedOperatorCandidate(
                evaluation=CandidateEvaluation(
                    static_pass=False,
                    light_pass=False,
                    full_pass=False,
                    deterministic_replay_pass=False,
                    full_feasible_pass=False,
                    objective_before=before.values[0],
                    runtime_seconds=time.perf_counter() - started,
                    failure_labels=("SOLVER_NO_SCHEDULE",),
                ),
                schedule=None,
                candidate_hash=None,
                replay_hash=None,
                executor_version=EXECUTOR_VERSION,
                solver_status=first.status,
                details={"plan": plan.__dict__, "oracleId": self.oracle_id},
            )

        generic = validate_schedule(problem, first.schedule)
        candidate_hash = schedule_hash(first.schedule)
        replay = self._solve(problem, incumbent, plan)
        replay_hash = None if replay.schedule is None else schedule_hash(replay.schedule)
        replay_pass = replay_hash == candidate_hash
        generated = GeneratedCandidate(
            schedule=first.schedule,
            status=first.status,
            fidelity="full",
            domain_validated=self.domain_oracle is None,
            details={"executorVersion": EXECUTOR_VERSION},
        )
        domain = {"passed": True, "oracleId": self.oracle_id}
        if self.domain_oracle is not None:
            domain = self.domain_oracle(problem, incumbent, generated)
        after = evaluate_objective(problem, first.schedule, baseline=incumbent)
        comparison, _, _ = compare_objectives(after, before)
        full_feasible_pass = generic.feasible and bool(domain.get("passed", False))
        full_pass = full_feasible_pass and comparison < 0
        failures: list[str] = []
        if not generic.feasible:
            failures.append("STATIC_VALIDATION_FAILED")
        if comparison >= 0:
            failures.append("NO_TRUE_IMPROVEMENT")
        if not domain.get("passed", False):
            failures.append("DOMAIN_ORACLE_FAILED")
        if not replay_pass:
            failures.append("DETERMINISTIC_REPLAY_MISMATCH")
        return ExecutedOperatorCandidate(
            evaluation=CandidateEvaluation(
                static_pass=generic.feasible,
                light_pass=generic.feasible and comparison < 0,
                full_pass=full_pass,
                deterministic_replay_pass=replay_pass,
                full_feasible_pass=full_feasible_pass,
                objective_before=before.values[0],
                objective_after=after.values[0],
                runtime_seconds=time.perf_counter() - started,
                failure_labels=tuple(failures),
            ),
            schedule=first.schedule,
            candidate_hash=candidate_hash,
            replay_hash=replay_hash,
            executor_version=EXECUTOR_VERSION,
            solver_status=first.status,
            details={
                "plan": plan.__dict__,
                "genericValidation": generic.as_dict(),
                "domainOracle": domain,
                "oracleId": self.oracle_id,
                "objectiveBefore": before.as_dict(),
                "objectiveAfter": after.as_dict(),
            },
        )

    def evaluator(
        self,
        problem: Problem,
        incumbent: Schedule,
    ) -> Callable[[OperatorPolicyState, OperatorParameterCandidate], CandidateEvaluation]:
        return lambda state, candidate: self.execute(
            problem, incumbent, state, candidate
        ).evaluation


__all__ = [
    "EXECUTOR_VERSION",
    "DeterministicOperatorExecutor",
    "ExecutedOperatorCandidate",
    "OperatorExecutionPlan",
    "build_operator_execution_plan",
]
