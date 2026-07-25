from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProblemKind = Literal["JSP", "FSP", "FJSP", "HFSP", "GENERIC"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Job(FrozenModel):
    id: str
    release: int = Field(default=0, ge=0)
    due: int | None = Field(default=None, ge=0)
    weight: float = Field(default=1.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Resource(FrozenModel):
    id: str
    name: str
    capacity: int = Field(default=1, ge=1)
    family: str = "machine"
    calendar: tuple[tuple[int, int], ...] = ()
    location: str | None = None
    tags: tuple[str, ...] = ()


class Mode(FrozenModel):
    id: str
    duration: int = Field(gt=0)
    resources: tuple[str, ...]
    setup_family: str | None = None
    route_id: str | None = None
    cost: float = Field(default=0.0, ge=0)
    energy: float = Field(default=0.0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resources_are_unique(self) -> "Mode":
        if not self.resources:
            raise ValueError("a mode needs at least one resource")
        if len(self.resources) != len(set(self.resources)):
            raise ValueError(f"mode {self.id} contains duplicate resources")
        return self


class Operation(FrozenModel):
    id: str
    job_id: str
    index: int = Field(ge=0)
    modes: tuple[Mode, ...]
    predecessors: tuple[str, ...] = ()
    release: int = Field(default=0, ge=0)
    due: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def modes_are_unique(self) -> "Operation":
        if not self.modes:
            raise ValueError(f"operation {self.id} has no mode")
        ids = [mode.id for mode in self.modes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"operation {self.id} has duplicate modes")
        return self


class ChoiceLink(FrozenModel):
    id: str
    operation_ids: tuple[str, ...]
    mode_keys: dict[str, str]


class ConstraintSpec(FrozenModel):
    id: str
    kind: Literal[
        "precedence",
        "capacity",
        "calendar",
        "setup",
        "blocking",
        "no_wait",
        "time_window",
        "transport",
        "binding",
        "domain",
    ]
    scope: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    encoded_by: Literal["core", "solver", "oracle"] = "core"


class ObjectiveComponent(FrozenModel):
    name: str
    sense: Literal["minimize", "maximize"] = "minimize"
    tolerance: float = Field(default=0.0, ge=0)
    weight: float = Field(default=1.0, gt=0)


class Problem(FrozenModel):
    id: str
    kind: ProblemKind
    jobs: tuple[Job, ...]
    resources: tuple[Resource, ...]
    operations: tuple[Operation, ...]
    choice_links: tuple[ChoiceLink, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    objective: tuple[ObjectiveComponent, ...] = (
        ObjectiveComponent(name="makespan"),
        ObjectiveComponent(name="total_tardiness"),
        ObjectiveComponent(name="total_flow_time"),
        ObjectiveComponent(name="change_cost"),
    )
    time_scale: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_are_valid(self) -> "Problem":
        job_ids = [item.id for item in self.jobs]
        resource_ids = [item.id for item in self.resources]
        operation_ids = [item.id for item in self.operations]
        mode_ids = [mode.id for operation in self.operations for mode in operation.modes]
        for label, values in (
            ("job", job_ids),
            ("resource", resource_ids),
            ("operation", operation_ids),
            ("mode", mode_ids),
        ):
            duplicates = [key for key, count in Counter(values).items() if count > 1]
            if duplicates:
                raise ValueError(f"duplicate {label} ids: {duplicates}")
        jobs = set(job_ids)
        resources = set(resource_ids)
        operations = set(operation_ids)
        modes = set(mode_ids)
        for operation in self.operations:
            if operation.job_id not in jobs:
                raise ValueError(f"unknown job for {operation.id}: {operation.job_id}")
            unknown_predecessors = set(operation.predecessors) - operations
            if unknown_predecessors:
                raise ValueError(
                    f"unknown predecessors for {operation.id}: {unknown_predecessors}"
                )
            if operation.id in operation.predecessors:
                raise ValueError(f"{operation.id} cannot precede itself")
            for mode in operation.modes:
                unknown_resources = set(mode.resources) - resources
                if unknown_resources:
                    raise ValueError(
                        f"unknown resources for {mode.id}: {unknown_resources}"
                    )
        for link in self.choice_links:
            if set(link.operation_ids) - operations:
                raise ValueError(f"choice link {link.id} references unknown operations")
            if set(link.mode_keys) - modes:
                raise ValueError(f"choice link {link.id} references unknown modes")
        return self

    def job_map(self) -> dict[str, Job]:
        return {item.id: item for item in self.jobs}

    def resource_map(self) -> dict[str, Resource]:
        return {item.id: item for item in self.resources}

    def operation_map(self) -> dict[str, Operation]:
        return {item.id: item for item in self.operations}

    def mode_map(self) -> dict[str, tuple[Operation, Mode]]:
        return {
            mode.id: (operation, mode)
            for operation in self.operations
            for mode in operation.modes
        }


class Assignment(FrozenModel):
    operation_id: str
    mode_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    route_id: str | None = None
    provenance: str = "incumbent"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def interval_is_positive(self) -> "Assignment":
        if self.end <= self.start:
            raise ValueError("assignment end must exceed start")
        return self


class Schedule(FrozenModel):
    problem_id: str
    assignments: tuple[Assignment, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def assignment_map(self) -> dict[str, Assignment]:
        return {item.operation_id: item for item in self.assignments}

    @property
    def makespan(self) -> int:
        return max((item.end for item in self.assignments), default=0)
