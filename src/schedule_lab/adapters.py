from __future__ import annotations

from collections.abc import Sequence

from .model import Mode, Operation, Problem, Resource


def _operation_id(job: int, operation: int) -> str:
    return f"J{job + 1}.O{operation + 1}"


def build_jsp(
    routes: Sequence[Sequence[tuple[str, int]]],
    *,
    problem_id: str = "jsp",
) -> Problem:
    resource_names = sorted({resource for route in routes for resource, _ in route})
    resources = tuple(Resource(id=name, name=name) for name in resource_names)
    operations: list[Operation] = []
    for job_index, route in enumerate(routes):
        predecessor: str | None = None
        for operation_index, (resource, duration) in enumerate(route):
            operation_id = _operation_id(job_index, operation_index)
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=f"J{job_index + 1}",
                    index=operation_index,
                    predecessors=() if predecessor is None else (predecessor,),
                    modes=(Mode(id=f"{operation_id}@{resource}", duration=duration, resources=(resource,)),),
                )
            )
            predecessor = operation_id
    return Problem(id=problem_id, kind="JSP", resources=resources, operations=tuple(operations))


def build_fsp(
    processing_times: Sequence[Sequence[int]],
    *,
    problem_id: str = "fsp",
) -> Problem:
    if not processing_times:
        raise ValueError("processing_times cannot be empty")
    machine_count = len(processing_times[0])
    if any(len(row) != machine_count for row in processing_times):
        raise ValueError("all FSP jobs must have the same number of stages")
    routes = [
        [(f"M{stage + 1}", duration) for stage, duration in enumerate(row)]
        for row in processing_times
    ]
    problem = build_jsp(routes, problem_id=problem_id)
    return problem.model_copy(update={"kind": "FSP"})


def build_fjsp(
    jobs: Sequence[Sequence[Sequence[tuple[str, int]]]],
    *,
    problem_id: str = "fjsp",
) -> Problem:
    resource_names = sorted(
        {resource for job in jobs for operation in job for resource, _ in operation}
    )
    resources = tuple(Resource(id=name, name=name) for name in resource_names)
    operations: list[Operation] = []
    for job_index, job in enumerate(jobs):
        predecessor: str | None = None
        for operation_index, alternatives in enumerate(job):
            operation_id = _operation_id(job_index, operation_index)
            modes = tuple(
                Mode(
                    id=f"{operation_id}@{resource}",
                    duration=duration,
                    resources=(resource,),
                )
                for resource, duration in alternatives
            )
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=f"J{job_index + 1}",
                    index=operation_index,
                    predecessors=() if predecessor is None else (predecessor,),
                    modes=modes,
                )
            )
            predecessor = operation_id
    return Problem(id=problem_id, kind="FJSP", resources=resources, operations=tuple(operations))


def build_hfsp(
    processing_times: Sequence[Sequence[int]],
    machines_per_stage: Sequence[int],
    *,
    problem_id: str = "hfsp",
) -> Problem:
    if not processing_times:
        raise ValueError("processing_times cannot be empty")
    if any(len(row) != len(machines_per_stage) for row in processing_times):
        raise ValueError("each HFSP job needs one processing time per stage")
    resources = tuple(
        Resource(id=f"S{stage + 1}M{machine + 1}", name=f"Stage {stage + 1} / Machine {machine + 1}")
        for stage, count in enumerate(machines_per_stage)
        for machine in range(count)
    )
    operations: list[Operation] = []
    for job_index, durations in enumerate(processing_times):
        predecessor: str | None = None
        for stage, duration in enumerate(durations):
            operation_id = _operation_id(job_index, stage)
            modes = tuple(
                Mode(
                    id=f"{operation_id}@S{stage + 1}M{machine + 1}",
                    duration=duration,
                    resources=(f"S{stage + 1}M{machine + 1}",),
                )
                for machine in range(machines_per_stage[stage])
            )
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=f"J{job_index + 1}",
                    index=stage,
                    predecessors=() if predecessor is None else (predecessor,),
                    modes=modes,
                )
            )
            predecessor = operation_id
    return Problem(id=problem_id, kind="HFSP", resources=resources, operations=tuple(operations))
