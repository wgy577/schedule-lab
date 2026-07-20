from __future__ import annotations

from pyjobshop import Model

from ..model import Assignment, Problem, Schedule


def solve_pyjobshop(
    problem: Problem,
    *,
    time_limit: float = 10.0,
    workers: int = 1,
) -> Schedule:
    """Independent high-level CP backend for problems without custom choice links."""

    if problem.choice_links:
        raise NotImplementedError("generic cross-family choice links currently use the direct CP-SAT backend")
    model = Model()
    resources = {}
    for resource in problem.resources:
        if resource.capacity == 1:
            resources[resource.id] = model.add_machine(name=resource.name)
        else:
            resources[resource.id] = model.add_renewable(capacity=resource.capacity, name=resource.name)
    jobs = {}
    tasks = {}
    operation_order = []
    mode_by_index: dict[int, str] = {}
    mode_index = 0
    for operation in problem.operations:
        if operation.job_id not in jobs:
            jobs[operation.job_id] = model.add_job(name=operation.job_id)
        job = jobs[operation.job_id]
        task = model.add_task(job=job, earliest_start=operation.release, name=operation.id)
        tasks[operation.id] = task
        operation_order.append(operation)
        for mode in operation.modes:
            required = [resources[resource_id] for resource_id in mode.resources]
            model.add_mode(task, required[0] if len(required) == 1 else required, mode.duration, name=mode.id)
            mode_by_index[mode_index] = mode.id
            mode_index += 1
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            model.add_end_before_start(tasks[predecessor], tasks[operation.id])
    result = model.solve(solver="ortools", time_limit=time_limit, display=False, num_workers=workers)
    if result.best is None:
        raise RuntimeError(f"PyJobShop did not find a solution: {result.status}")
    assignments = tuple(
        Assignment(
            operation_id=operation.id,
            mode_id=mode_by_index[scheduled.mode],
            start=scheduled.start,
            end=scheduled.end,
        )
        for operation, scheduled in zip(operation_order, result.best.tasks)
    )
    return Schedule(
        problem_id=problem.id,
        assignments=assignments,
        metadata={
            "solver": "pyjobshop-ortools",
            "status": str(result.status),
            "domain_validated": not problem.metadata.get("requires_domain_validation", False),
        },
    )
