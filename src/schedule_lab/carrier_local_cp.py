from __future__ import annotations

from typing import Any, Iterable

from ortools.sat.python import cp_model


DEFAULT_CAPACITIES = {5: 2, 6: 2, 7: 2}


def solve_fixed_mode_neighborhood(
    raw_schedule: list[dict[str, Any]],
    neighborhood_jobs: Iterable[int],
    *,
    seed: int = 0,
    deterministic_time: float = 1.0,
    capacities: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Repair local timing/order with incumbent modes; return replay priorities.

    This is a proposal model. State-dependent travel and collision delays remain
    the responsibility of the legacy domain replay.
    """

    selected_jobs = set(map(int, neighborhood_jobs))
    if not selected_jobs:
        raise ValueError("neighborhood_jobs is empty")
    capacities = {**DEFAULT_CAPACITIES, **(capacities or {})}
    scale = 10
    normalized = [
        {
            "job": int(item["job"]),
            "op": int(item["op"]),
            "machine": int(item["machine"]),
            "start": int(round(float(item["start"]) * scale)),
            "end": int(round(float(item["end"]) * scale)),
            "duration": int(round(float(item.get("dur", float(item["end"]) - float(item["start"]))) * scale)),
        }
        for item in raw_schedule
    ]
    selected = [item for item in normalized if item["job"] in selected_jobs]
    outside = [item for item in normalized if item["job"] not in selected_jobs]
    if not selected:
        raise ValueError("no schedule operations match neighborhood_jobs")
    horizon = max(item["end"] for item in normalized) + sum(item["duration"] for item in selected)
    model = cp_model.CpModel()
    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    resource_intervals: dict[int, list[cp_model.IntervalVar]] = {}
    resource_demands: dict[int, list[int]] = {}

    for item in selected:
        key = (item["job"], item["op"])
        start = model.new_int_var(0, horizon, f"start:J{key[0]}.O{key[1]}")
        end = model.new_int_var(0, horizon, f"end:J{key[0]}.O{key[1]}")
        interval = model.new_fixed_size_interval_var(start, item["duration"], f"interval:J{key[0]}.O{key[1]}")
        model.add(end == start + item["duration"])
        starts[key], ends[key] = start, end
        resource_intervals.setdefault(item["machine"], []).append(interval)
        resource_demands.setdefault(item["machine"], []).append(1)
        model.add_hint(start, item["start"])

    for item in outside:
        interval = model.new_fixed_size_interval_var(
            item["start"], item["duration"], f"fixed:J{item['job']}.O{item['op']}"
        )
        resource_intervals.setdefault(item["machine"], []).append(interval)
        resource_demands.setdefault(item["machine"], []).append(1)

    for job in sorted(selected_jobs):
        operations = sorted((item for item in selected if item["job"] == job), key=lambda item: item["op"])
        for left, right in zip(operations, operations[1:]):
            model.add(starts[(right["job"], right["op"])] >= ends[(left["job"], left["op"])])

    for machine, intervals in resource_intervals.items():
        capacity = capacities.get(machine, 1)
        if capacity == 1:
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals, resource_demands[machine], capacity)

    outside_end = max((item["end"] for item in outside), default=0)
    makespan = model.new_int_var(outside_end, horizon, "makespan")
    model.add_max_equality(makespan, [*ends.values(), outside_end])
    shifts: list[cp_model.IntVar] = []
    for item in selected:
        key = (item["job"], item["op"])
        shift = model.new_int_var(0, horizon, f"shift:J{key[0]}.O{key[1]}")
        model.add_abs_equality(shift, starts[key] - item["start"])
        shifts.append(shift)
    shift_bound = horizon * len(selected)
    model.minimize(makespan * (shift_bound + 1) + sum(shifts))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_deterministic_time = deterministic_time
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status, "prioritiesByOperation": {}, "proposalOnly": True}
    assignments = [
        {
            "job": item["job"],
            "op": item["op"],
            "machine": item["machine"],
            "start": solver.value(starts[(item["job"], item["op"])]) / scale,
            "end": solver.value(ends[(item["job"], item["op"])]) / scale,
        }
        for item in selected
    ]
    priorities = {
        str(operation): [
            item["job"]
            for item in sorted(
                (assignment for assignment in assignments if assignment["op"] == operation),
                key=lambda assignment: (assignment["start"], assignment["end"], assignment["job"]),
            )
        ]
        for operation in sorted({item["op"] for item in selected})
    }
    return {
        "status": status,
        "proposalOnly": True,
        "fixedModes": True,
        "workers": 1,
        "seed": seed,
        "deterministicTime": deterministic_time,
        "proxyMakespan": solver.value(makespan) / scale,
        "objectiveBound": solver.best_objective_bound,
        "prioritiesByOperation": priorities,
        "assignments": sorted(assignments, key=lambda item: (item["job"], item["op"])),
    }
