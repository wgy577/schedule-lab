from __future__ import annotations

import random
from typing import Iterable

from .ir import (
    DynamicEvent,
    Job,
    Mode,
    ObjectiveComponent,
    Operation,
    Problem,
    Resource,
)


DEFAULT_OBJECTIVE = (
    ObjectiveComponent(name="makespan"),
    ObjectiveComponent(name="total_tardiness"),
    ObjectiveComponent(name="total_flow_time"),
    ObjectiveComponent(name="change_cost"),
)


def build_jsp(
    routes: list[list[tuple[str, int]]],
    *,
    problem_id: str = "jsp",
    due_factor: float = 1.5,
) -> Problem:
    resource_ids = sorted({machine for route in routes for machine, _ in route})
    jobs = []
    operations = []
    for job_index, route in enumerate(routes):
        job_id = f"J{job_index + 1}"
        due = round(sum(duration for _, duration in route) * due_factor)
        jobs.append(Job(id=job_id, due=due))
        previous = None
        for index, (machine, duration) in enumerate(route):
            operation_id = f"{job_id}.O{index + 1}"
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=job_id,
                    index=index,
                    predecessors=() if previous is None else (previous,),
                    modes=(
                        Mode(
                            id=f"{operation_id}@{machine}",
                            duration=duration,
                            resources=(machine,),
                        ),
                    ),
                )
            )
            previous = operation_id
    return Problem(
        id=problem_id,
        kind="JSP",
        jobs=tuple(jobs),
        resources=tuple(Resource(id=item, name=item) for item in resource_ids),
        operations=tuple(operations),
        objective=DEFAULT_OBJECTIVE,
    )


def build_fsp(
    processing: list[list[int]],
    *,
    problem_id: str = "fsp",
) -> Problem:
    if not processing:
        raise ValueError("processing matrix is empty")
    stages = len(processing[0])
    routes = [
        [(f"M{stage + 1}", duration) for stage, duration in enumerate(row)]
        for row in processing
    ]
    base = build_jsp(routes, problem_id=problem_id)
    return base.model_copy(
        update={
            "kind": "FSP",
            "operations": tuple(
                operation.model_copy(
                    update={"stage_id": f"S{operation.index + 1}"}
                )
                for operation in base.operations
            ),
        }
    )


def build_fjsp(
    alternatives: list[list[list[tuple[str, int]]]],
    *,
    problem_id: str = "fjsp",
) -> Problem:
    resources = sorted(
        {
            machine
            for job in alternatives
            for operation in job
            for machine, _ in operation
        }
    )
    jobs = []
    operations = []
    for job_index, operation_alternatives in enumerate(alternatives):
        job_id = f"J{job_index + 1}"
        jobs.append(Job(id=job_id))
        previous = None
        for index, choices in enumerate(operation_alternatives):
            operation_id = f"{job_id}.O{index + 1}"
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=job_id,
                    index=index,
                    predecessors=() if previous is None else (previous,),
                    modes=tuple(
                        Mode(
                            id=f"{operation_id}@{machine}",
                            duration=duration,
                            resources=(machine,),
                        )
                        for machine, duration in choices
                    ),
                )
            )
            previous = operation_id
    return Problem(
        id=problem_id,
        kind="FJSP",
        jobs=tuple(jobs),
        resources=tuple(Resource(id=item, name=item) for item in resources),
        operations=tuple(operations),
        objective=DEFAULT_OBJECTIVE,
    )


def build_hfsp(
    processing: list[list[int]],
    machines_per_stage: list[int],
    *,
    problem_id: str = "hfsp",
) -> Problem:
    if not processing or len(processing[0]) != len(machines_per_stage):
        raise ValueError("processing matrix and stage vector disagree")
    resources = [
        Resource(
            id=f"S{stage + 1}.M{machine + 1}",
            name=f"Stage {stage + 1} Machine {machine + 1}",
            family=f"stage-{stage + 1}",
            tags=(f"stage:{stage + 1}",),
        )
        for stage, count in enumerate(machines_per_stage)
        for machine in range(count)
    ]
    jobs = []
    operations = []
    for job_index, row in enumerate(processing):
        job_id = f"J{job_index + 1}"
        jobs.append(Job(id=job_id))
        previous = None
        for stage, duration in enumerate(row):
            operation_id = f"{job_id}.O{stage + 1}"
            choices = [
                item for item in resources if item.family == f"stage-{stage + 1}"
            ]
            operations.append(
                Operation(
                    id=operation_id,
                    job_id=job_id,
                    index=stage,
                    stage_id=f"S{stage + 1}",
                    predecessors=() if previous is None else (previous,),
                    modes=tuple(
                        Mode(
                            id=f"{operation_id}@{resource.id}",
                            duration=duration,
                            resources=(resource.id,),
                        )
                        for resource in choices
                    ),
                )
            )
            previous = operation_id
    return Problem(
        id=problem_id,
        kind="HFSP",
        jobs=tuple(jobs),
        resources=tuple(resources),
        operations=tuple(operations),
        objective=DEFAULT_OBJECTIVE,
    )


def build_djsp(
    routes: list[list[tuple[str, int]]],
    release_times: list[int],
    *,
    problem_id: str = "djsp",
) -> Problem:
    """Build a dynamic JSP whose arrivals are explicit in both jobs and events."""

    if len(routes) != len(release_times):
        raise ValueError("routes and release_times disagree")
    base = build_jsp(routes, problem_id=problem_id)
    jobs = tuple(
        job.model_copy(update={"release": int(release_times[index])})
        for index, job in enumerate(base.jobs)
    )
    events = tuple(
        DynamicEvent(
            id=f"arrival:{job.id}",
            kind="job_arrival",
            time=job.release,
            scope=(job.id,),
        )
        for job in jobs
    )
    return base.model_copy(
        update={
            "kind": "DJSP",
            "environment": "dynamic",
            "jobs": jobs,
            "events": events,
        }
    )


def build_dfjsp(
    alternatives: list[list[list[tuple[str, int]]]],
    release_times: list[int],
    *,
    problem_id: str = "dfjsp",
) -> Problem:
    """Build a dynamic FJSP with explicit arrival events."""

    if len(alternatives) != len(release_times):
        raise ValueError("alternatives and release_times disagree")
    base = build_fjsp(alternatives, problem_id=problem_id)
    jobs = tuple(
        job.model_copy(update={"release": int(release_times[index])})
        for index, job in enumerate(base.jobs)
    )
    events = tuple(
        DynamicEvent(
            id=f"arrival:{job.id}",
            kind="job_arrival",
            time=job.release,
            scope=(job.id,),
        )
        for job in jobs
    )
    return base.model_copy(
        update={
            "kind": "DFJSP",
            "environment": "dynamic",
            "jobs": jobs,
            "events": events,
        }
    )


def six_family_examples() -> dict[str, Problem]:
    """Deterministic fixtures for the six-family unified representation."""

    static = example_problems()
    djsp = build_djsp(
        [
            [("M1", 3), ("M2", 2)],
            [("M2", 2), ("M1", 4)],
            [("M1", 2), ("M2", 3)],
        ],
        [0, 2, 5],
        problem_id="example-djsp",
    )
    dfjsp = build_dfjsp(
        [
            [[("M1", 3), ("M2", 4)], [("M2", 2), ("M3", 3)]],
            [[("M1", 2), ("M3", 3)], [("M2", 4), ("M3", 2)]],
            [[("M2", 2), ("M3", 4)], [("M1", 3), ("M3", 2)]],
        ],
        [0, 3, 6],
        problem_id="example-dfjsp",
    )
    return {**static, "djsp": djsp, "dfjsp": dfjsp}


def example_problems() -> dict[str, Problem]:
    return {
        "jsp": build_jsp(
            [
                [("M1", 3), ("M2", 2), ("M3", 2)],
                [("M1", 2), ("M3", 1), ("M2", 4)],
                [("M2", 4), ("M3", 3)],
            ],
            problem_id="example-jsp",
        ),
        "fsp": build_fsp(
            [[3, 2, 4], [2, 4, 3], [4, 3, 2], [3, 5, 1]],
            problem_id="example-fsp",
        ),
        "fjsp": build_fjsp(
            [
                [[("M1", 3), ("M2", 4)], [("M2", 2), ("M3", 3)]],
                [[("M1", 2), ("M3", 3)], [("M2", 4), ("M3", 2)]],
                [[("M2", 2), ("M3", 4)], [("M1", 3), ("M3", 2)]],
            ],
            problem_id="example-fjsp",
        ),
        "hfsp": build_hfsp(
            [[3, 4, 2], [2, 5, 3], [4, 2, 4], [3, 3, 2]],
            [2, 2, 1],
            problem_id="example-hfsp",
        ),
    }


def random_fjsp(
    *,
    jobs: int,
    operations: int,
    machines: int,
    flexibility: int = 2,
    duration_range: tuple[int, int] = (1, 20),
    seed: int = 0,
    problem_id: str | None = None,
) -> Problem:
    rng = random.Random(seed)
    machine_ids = [f"M{index + 1}" for index in range(machines)]
    data = []
    for _ in range(jobs):
        job = []
        for _ in range(operations):
            selected = rng.sample(
                machine_ids,
                k=min(flexibility, machines),
            )
            job.append(
                [
                    (machine, rng.randint(*duration_range))
                    for machine in selected
                ]
            )
        data.append(job)
    return build_fjsp(
        data,
        problem_id=problem_id
        or f"fjsp-j{jobs}-o{operations}-m{machines}-s{seed}",
    )


def benchmark_suite(
    *,
    families: Iterable[str] = ("jsp", "fsp", "fjsp", "hfsp"),
    seed: int = 0,
    instances_per_family: int = 1,
) -> dict[str, Problem]:
    examples = example_problems()
    suite: dict[str, Problem] = {}
    for family in families:
        if family not in examples:
            raise ValueError(f"unsupported benchmark family: {family}")
        for index in range(instances_per_family):
            key = family if instances_per_family == 1 else f"{family}-{index}"
            if index == 0:
                suite[key] = examples[family]
            elif family == "fjsp":
                suite[key] = random_fjsp(
                    jobs=5,
                    operations=4,
                    machines=4,
                    seed=seed + index,
                    problem_id=f"benchmark-{family}-{seed}-{index}",
                )
            else:
                # Deterministic duration perturbation retains each family's
                # routing/resource semantics while creating another instance.
                rng = random.Random(seed + index)
                base = examples[family]
                suite[key] = base.model_copy(
                    update={
                        "id": f"benchmark-{family}-{seed}-{index}",
                        "operations": tuple(
                            operation.model_copy(
                                update={
                                    "modes": tuple(
                                        mode.model_copy(
                                            update={
                                                "duration": max(
                                                    1,
                                                    mode.duration + rng.choice((-1, 0, 1)),
                                                )
                                            }
                                        )
                                        for mode in operation.modes
                                    )
                                }
                            )
                            for operation in base.operations
                        ),
                    }
                )
    return suite
