from __future__ import annotations

from dataclasses import dataclass

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
    frozen_mode_operation_ids: set[str] | None = None,
    frozen_start_operation_ids: set[str] | None = None,
    forced_mode_ids: dict[str, str] | None = None,
    enforced_orderings: tuple[tuple[str, str], ...] = (),
    forced_start_times: dict[str, int] | None = None,
    seed: int = 0,
    workers: int = 1,
    max_deterministic_time: float = 1.0,
    max_conflicts: int | None = None,
    stability_weight: int = 1,
    max_makespan: int | None = None,
) -> CPSATResult:
    # Keep non-solver imports/tests usable without OR-Tools. Never substitute a solver.
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError("CP-SAT requires OR-Tools; run: python -m pip install 'ortools>=9.12,<10'") from exc
    frozen = frozen_operation_ids or set()
    frozen_modes = set(frozen_mode_operation_ids or ()) | frozen
    frozen_starts = set(frozen_start_operation_ids or ()) | frozen
    forced_modes = forced_mode_ids or {}
    forced_starts = forced_start_times or {}
    if (frozen_modes or frozen_starts) and incumbent is None:
        raise ValueError("frozen operations require an incumbent")
    known_operations = {item.id for item in problem.operations}
    unknown_frozen = (frozen_modes | frozen_starts) - known_operations
    if unknown_frozen:
        raise ValueError(f"frozen sets reference unknown operations: {sorted(unknown_frozen)}")
    unknown_forced = set(forced_modes) - {item.id for item in problem.operations}
    if unknown_forced:
        raise ValueError(f"forced modes reference unknown operations: {sorted(unknown_forced)}")
    unknown_starts = set(forced_starts) - {item.id for item in problem.operations}
    if unknown_starts:
        raise ValueError(f"forced starts reference unknown operations: {sorted(unknown_starts)}")
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
    res_starts: dict[str, list[cp_model.IntVar]] = {
        resource.id: [] for resource in problem.resources
    }
    res_ends: dict[str, list[cp_model.IntVar]] = {
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
                res_starts[resource].append(starts[operation.id])
                res_ends[resource].append(ends[operation.id])
        model.add_exactly_one(choices)
        for predecessor in operation.predecessors:
            model.add(starts[operation.id] >= ends[predecessor])
        forced_mode_id = forced_modes.get(operation.id)
        if forced_mode_id is not None:
            if forced_mode_id not in {item.id for item in operation.modes}:
                raise ValueError(
                    f"forced mode {forced_mode_id!r} is not eligible for {operation.id!r}"
                )
            model.add(presence[forced_mode_id] == 1)
        if operation.id in forced_starts:
            model.add(starts[operation.id] == int(forced_starts[operation.id]))
    for left, right in enforced_orderings:
        if left not in starts or right not in starts:
            raise ValueError(f"ordering references unknown operations: {(left, right)!r}")
        if left == right:
            raise ValueError("ordering endpoints must differ")
        model.add(starts[right] >= ends[left])
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
        if operation.id in frozen_starts:
            model.add(starts[operation.id] == previous.start)
            model.add(ends[operation.id] == previous.end)
        if operation.id in frozen_modes:
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
    if max_makespan is not None:
        model.add(makespan <= max_makespan)
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

    # Per-resource effective load (sum of chosen-mode durations on that machine).
    busy_terms = {resource.id: [] for resource in problem.resources}
    for operation in problem.operations:
        for mode in operation.modes:
            if len(mode.resources) == 1:
                busy_terms[mode.resources[0]].append(
                    mode.duration * presence[mode.id]
                )
    balances = []
    if "balance" in {c.name for c in problem.objective}:
        max_busy = model.new_int_var(0, horizon, "max_busy")
        min_busy = model.new_int_var(0, horizon, "min_busy")
        for rid, terms in busy_terms.items():
            if not terms:
                continue
            load = sum(terms)
            model.add(max_busy >= load)
            model.add(min_busy <= load)
        balances.append(max_busy - min_busy)
    balance = sum(balances)

    compactive = []
    for resource in problem.resources:
        sts, ens = res_starts[resource.id], res_ends[resource.id]
        if not sts:
            continue
        span_end = model.new_int_var(0, horizon, f"span_end:{resource.id}")
        span_start = model.new_int_var(0, horizon, f"span_start:{resource.id}")
        for s in sts:
            model.add(span_end >= s)
        for e in ens:
            model.add(span_start <= e)
        compactive.append(span_end - span_start)
    compact = sum(compactive)

    components = {
        "makespan": makespan,
        "total_tardiness": sum(tardiness),
        "total_flow_time": sum(flow),
        "change_cost": (
            sum(mode_changes) * horizon * max(1, len(problem.operations))
            + sum(start_deviations) * stability_weight
        ),
        "compact": compact,
        "balance": balance,
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
