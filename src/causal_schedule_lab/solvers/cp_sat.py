from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from ..ir import Assignment, Problem, Schedule


@dataclass(frozen=True)
class CPSATResult:
    schedule: Schedule | None
    status: str
    objective_bound: float | None
    conflicts: int
    branches: int
    deterministic_time: float


def solve_cp_sat(
    problem: Problem,
    *,
    incumbent: Schedule | None = None,
    frozen_operation_ids: set[str] | None = None,
    seed: int = 0,
    workers: int = 1,
    max_deterministic_time: float = 1.0,
    max_conflicts: int | None = None,
    stability_weight: int = 1,
) -> CPSATResult:
    frozen = frozen_operation_ids or set()
    if frozen and incumbent is None:
        raise ValueError("frozen operations require an incumbent")
    model = cp_model.CpModel()
    operation_map = problem.operation_map()
    jobs = problem.job_map()
    horizon = max(
        1,
        max((job.release for job in problem.jobs), default=0)
        + sum(max(mode.duration for mode in op.modes) for op in problem.operations),
    )
    starts = {}
    ends = {}
    presence = {}
    intervals: dict[str, list[cp_model.IntervalVar]] = {
        resource.id: [] for resource in problem.resources
    }
    demands: dict[str, list[int]] = {
        resource.id: [] for resource in problem.resources
    }
    for operation in problem.operations:
        release = max(operation.release, jobs[operation.job_id].release)
        starts[operation.id] = model.new_int_var(
            release,
            horizon,
            f"start:{operation.id}",
        )
        ends[operation.id] = model.new_int_var(
            release + 1,
            horizon,
            f"end:{operation.id}",
        )
        choices = []
        for mode in operation.modes:
            selected = model.new_bool_var(f"mode:{mode.id}")
            presence[mode.id] = selected
            choices.append(selected)
            interval = model.new_optional_interval_var(
                starts[operation.id],
                mode.duration,
                ends[operation.id],
                selected,
                f"interval:{mode.id}",
            )
            for resource in mode.resources:
                intervals[resource].append(interval)
                demands[resource].append(1)
        model.add_exactly_one(choices)
        for predecessor in operation.predecessors:
            model.add(starts[operation.id] >= ends[predecessor])
    for resource in problem.resources:
        if resource.capacity == 1:
            model.add_no_overlap(intervals[resource.id])
        else:
            model.add_cumulative(
                intervals[resource.id],
                demands[resource.id],
                resource.capacity,
            )
        if resource.calendar:
            unavailable = []
            cursor = 0
            for window_start, window_end in sorted(resource.calendar):
                if window_start > cursor:
                    unavailable.append((cursor, window_start))
                cursor = max(cursor, window_end)
            if cursor < horizon:
                unavailable.append((cursor, horizon))
            for index, (start, end) in enumerate(unavailable):
                if end > start:
                    intervals[resource.id].append(
                        model.new_fixed_size_interval_var(
                            start,
                            end - start,
                            f"calendar-block:{resource.id}:{index}",
                        )
                    )
                    demands[resource.id].append(resource.capacity)
            # Rebuild after calendar blocks have been appended.
            if resource.capacity == 1:
                model.add_no_overlap(intervals[resource.id])
            else:
                model.add_cumulative(
                    intervals[resource.id],
                    demands[resource.id],
                    resource.capacity,
                )
    for link in problem.choice_links:
        keys = sorted(set(link.mode_keys.values()))
        chosen = model.new_int_var(0, len(keys) - 1, f"choice:{link.id}")
        key_index = {key: index for index, key in enumerate(keys)}
        for operation_id in link.operation_ids:
            for mode in operation_map[operation_id].modes:
                key = link.mode_keys.get(mode.id)
                if key is not None:
                    model.add(chosen == key_index[key]).only_enforce_if(
                        presence[mode.id]
                    )
    for constraint in problem.constraints:
        if constraint.encoded_by == "oracle":
            continue
        scope = [item for item in constraint.scope if item in starts]
        parameters = constraint.parameters
        if constraint.kind == "time_window":
            for operation_id in scope:
                if "earliest_start" in parameters:
                    model.add(
                        starts[operation_id] >= int(parameters["earliest_start"])
                    )
                if "latest_start" in parameters:
                    model.add(
                        starts[operation_id] <= int(parameters["latest_start"])
                    )
                if "latest_end" in parameters:
                    model.add(ends[operation_id] <= int(parameters["latest_end"]))
        elif constraint.kind in {"no_wait", "transport", "blocking"}:
            for left, right in zip(scope, scope[1:]):
                if constraint.kind == "no_wait":
                    model.add(starts[right] == ends[left])
                    continue
                minimum = int(parameters.get("min_lag", 0))
                model.add(starts[right] >= ends[left] + minimum)
                if "max_lag" in parameters:
                    model.add(
                        starts[right]
                        <= ends[left] + int(parameters["max_lag"])
                    )

    baseline = {} if incumbent is None else incumbent.assignment_map()
    start_deviations = []
    mode_changes = []
    for operation in problem.operations:
        previous = baseline.get(operation.id)
        if previous is None:
            continue
        model.add_hint(starts[operation.id], previous.start)
        for mode in operation.modes:
            model.add_hint(presence[mode.id], int(mode.id == previous.mode_id))
        if operation.id in frozen:
            model.add(starts[operation.id] == previous.start)
            model.add(ends[operation.id] == previous.end)
            model.add(presence[previous.mode_id] == 1)
        deviation = model.new_int_var(0, horizon, f"deviation:{operation.id}")
        model.add_abs_equality(
            deviation,
            starts[operation.id] - previous.start,
        )
        start_deviations.append(deviation)
        changed = model.new_bool_var(f"changed:{operation.id}")
        model.add(changed + presence[previous.mode_id] == 1)
        mode_changes.append(changed)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(ends.values()))
    completion = {}
    tardiness = []
    flow = []
    for job in problem.jobs:
        job_ends = [
            ends[operation.id]
            for operation in problem.operations
            if operation.job_id == job.id
        ]
        completion[job.id] = model.new_int_var(0, horizon, f"completion:{job.id}")
        model.add_max_equality(completion[job.id], job_ends)
        flow.append(completion[job.id] - job.release)
        if job.due is not None:
            late = model.new_int_var(0, horizon, f"tardiness:{job.id}")
            model.add_max_equality(late, [completion[job.id] - job.due, 0])
            multiplier = max(1, round(job.weight * 100))
            tardiness.append(late * multiplier)

    components = {
        "makespan": makespan,
        "total_tardiness": sum(tardiness),
        "total_flow_time": sum(flow),
        "change_cost": (
            sum(mode_changes) * horizon * max(1, len(problem.operations))
            + sum(start_deviations) * stability_weight
        ),
    }
    supported = [
        component.name
        for component in problem.objective
        if component.name in components and component.sense == "minimize"
    ]
    if not supported:
        supported = ["makespan", "change_cost"]

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.max_deterministic_time = (
        max_deterministic_time / max(1, len(supported))
    )
    if max_conflicts is not None:
        solver.parameters.max_number_of_conflicts = max_conflicts
    total_conflicts = 0
    total_branches = 0
    total_wall_time = 0.0
    status_code = cp_model.UNKNOWN
    status = "UNKNOWN"
    objective_bound = None
    for component_name in supported:
        model.minimize(components[component_name])
        status_code = solver.solve(model)
        status = solver.status_name(status_code)
        total_conflicts += solver.num_conflicts
        total_branches += solver.num_branches
        total_wall_time += solver.wall_time
        objective_bound = solver.best_objective_bound
        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return CPSATResult(
                None,
                status,
                objective_bound,
                total_conflicts,
                total_branches,
                total_wall_time,
            )
        # Preserve each higher-priority optimum before optimizing the next
        # component. This is true lexicographic optimization and avoids
        # coefficient overflow on large scheduling instances.
        model.add(components[component_name] == round(solver.objective_value))
    assignments = []
    for operation in problem.operations:
        mode = next(
            item
            for item in operation.modes
            if solver.boolean_value(presence[item.id])
        )
        assignments.append(
            Assignment(
                operation_id=operation.id,
                mode_id=mode.id,
                start=solver.value(starts[operation.id]),
                end=solver.value(ends[operation.id]),
                route_id=mode.route_id,
                provenance="cp-sat-repair",
            )
        )
    return CPSATResult(
        Schedule(
            problem_id=problem.id,
            assignments=tuple(assignments),
            metadata={
                "solver": "ortools-cp-sat",
                "seed": seed,
                "workers": workers,
                "deterministicBudget": max_deterministic_time,
                "status": status,
            },
        ),
        status,
        objective_bound,
        total_conflicts,
        total_branches,
        total_wall_time,
    )
