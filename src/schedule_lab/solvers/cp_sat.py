from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from ..model import Assignment, Problem, Schedule


@dataclass(frozen=True)
class CpSatResult:
    schedule: Schedule | None
    status: str
    objective_bound: float | None


def solve_cp_sat(
    problem: Problem,
    *,
    time_limit: float = 10.0,
    seed: int = 0,
    workers: int = 1,
    warm_start: Schedule | None = None,
    stability_weight: int = 0,
    frozen_operation_ids: set[str] | None = None,
    deterministic_time: float | None = None,
) -> CpSatResult:
    frozen_operation_ids = frozen_operation_ids or set()
    if frozen_operation_ids and warm_start is None:
        raise ValueError("frozen operations require a warm_start incumbent")
    model = cp_model.CpModel()
    horizon = max((operation.release for operation in problem.operations), default=0) + sum(
        max(mode.duration for mode in operation.modes) for operation in problem.operations
    )
    start_vars: dict[str, cp_model.IntVar] = {}
    end_vars: dict[str, cp_model.IntVar] = {}
    presence: dict[str, cp_model.BoolVar] = {}
    resource_intervals: dict[str, list[cp_model.IntervalVar]] = {resource.id: [] for resource in problem.resources}
    resource_demands: dict[str, list[int]] = {resource.id: [] for resource in problem.resources}

    for operation in problem.operations:
        start = model.new_int_var(operation.release, horizon, f"start:{operation.id}")
        end = model.new_int_var(operation.release, horizon, f"end:{operation.id}")
        start_vars[operation.id] = start
        end_vars[operation.id] = end

    for operation in problem.operations:
        start = start_vars[operation.id]
        end = end_vars[operation.id]
        operation_presence = []
        for mode in operation.modes:
            selected = model.new_bool_var(f"mode:{mode.id}")
            presence[mode.id] = selected
            operation_presence.append(selected)
            interval = model.new_optional_interval_var(start, mode.duration, end, selected, f"interval:{mode.id}")
            for resource_id in mode.resources:
                resource_intervals[resource_id].append(interval)
                resource_demands[resource_id].append(1)
        model.add_exactly_one(operation_presence)
        for predecessor in operation.predecessors:
            model.add(start >= end_vars[predecessor])

    for resource in problem.resources:
        intervals = resource_intervals[resource.id]
        if resource.capacity == 1:
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals, resource_demands[resource.id], resource.capacity)

    for link in problem.choice_links:
        keys = sorted(set(link.mode_keys.values()))
        key_index = {key: index for index, key in enumerate(keys)}
        selected_key = model.new_int_var(0, len(keys) - 1, f"choice:{link.id}")
        for operation_id in link.operation_ids:
            operation = next(item for item in problem.operations if item.id == operation_id)
            for mode in operation.modes:
                key = link.mode_keys.get(mode.id)
                if key is not None:
                    model.add(selected_key == key_index[key]).only_enforce_if(presence[mode.id])

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(end_vars.values()))
    secondary_terms: list[cp_model.LinearExpr] = []
    if warm_start is not None:
        baseline = warm_start.assignment_map()
        for operation in problem.operations:
            previous = baseline.get(operation.id)
            if previous is None:
                continue
            model.add_hint(start_vars[operation.id], previous.start)
            for mode in operation.modes:
                model.add_hint(presence[mode.id], int(mode.id == previous.mode_id))
            if operation.id in frozen_operation_ids:
                model.add(start_vars[operation.id] == previous.start)
                model.add(presence[previous.mode_id] == 1)
            if stability_weight:
                deviation = model.new_int_var(0, horizon, f"shift:{operation.id}")
                model.add_abs_equality(deviation, start_vars[operation.id] - previous.start)
                secondary_terms.append(deviation * stability_weight)
                if previous.mode_id in presence:
                    secondary_terms.append(1 - presence[previous.mode_id])
    secondary_bound = horizon * max(1, len(problem.operations)) * max(1, stability_weight) + len(problem.operations)
    model.minimize(makespan * (secondary_bound + 1) + sum(secondary_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if deterministic_time is not None:
        solver.parameters.max_deterministic_time = deterministic_time
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = workers
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return CpSatResult(schedule=None, status=status, objective_bound=None)
    assignments = []
    for operation in problem.operations:
        selected_mode = next(mode for mode in operation.modes if solver.boolean_value(presence[mode.id]))
        assignments.append(
            Assignment(
                operation_id=operation.id,
                mode_id=selected_mode.id,
                start=solver.value(start_vars[operation.id]),
                end=solver.value(end_vars[operation.id]),
            )
        )
    schedule = Schedule(
        problem_id=problem.id,
        assignments=tuple(assignments),
        metadata={
            "solver": "ortools-cp-sat",
            "status": status,
            "seed": seed,
            "domain_validated": not problem.metadata.get("requires_domain_validation", False),
        },
    )
    return CpSatResult(schedule=schedule, status=status, objective_bound=solver.best_objective_bound)
