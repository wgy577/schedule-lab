from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProblemKind = Literal["JSP", "FSP", "FJSP", "HFSP", "CARRIER", "GENERIC"]


class Resource(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    capacity: int = Field(default=1, ge=1)
    tags: tuple[str, ...] = ()


class Mode(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    duration: int = Field(gt=0)
    resources: tuple[str, ...]

    @model_validator(mode="after")
    def resources_are_unique(self) -> "Mode":
        if not self.resources:
            raise ValueError("a mode must require at least one resource")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError(f"mode {self.id} contains duplicate resources")
        return self


class Operation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    job_id: str
    index: int = Field(ge=0)
    modes: tuple[Mode, ...]
    predecessors: tuple[str, ...] = ()
    release: int = Field(default=0, ge=0)
    due: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def has_unique_modes(self) -> "Operation":
        if not self.modes:
            raise ValueError(f"operation {self.id} has no execution mode")
        ids = [mode.id for mode in self.modes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"operation {self.id} has duplicate mode ids")
        return self


class ChoiceLink(BaseModel):
    """Forces linked operations to select modes carrying the same abstract key.

    This expresses constraints such as preparation spot P2 implying catapult C2,
    even though the actual resource identifiers differ.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    operation_ids: tuple[str, ...]
    mode_keys: dict[str, str]


class Problem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: ProblemKind = "GENERIC"
    resources: tuple[Resource, ...]
    operations: tuple[Operation, ...]
    choice_links: tuple[ChoiceLink, ...] = ()
    time_scale: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_are_valid(self) -> "Problem":
        resource_ids = [resource.id for resource in self.resources]
        operation_ids = [operation.id for operation in self.operations]
        mode_ids = [mode.id for operation in self.operations for mode in operation.modes]
        for label, values in (
            ("resource", resource_ids),
            ("operation", operation_ids),
            ("mode", mode_ids),
        ):
            duplicates = [item for item, count in Counter(values).items() if count > 1]
            if duplicates:
                raise ValueError(f"duplicate {label} ids: {duplicates}")
        resource_set = set(resource_ids)
        operation_set = set(operation_ids)
        mode_set = set(mode_ids)
        for operation in self.operations:
            missing_resources = {
                resource
                for mode in operation.modes
                for resource in mode.resources
                if resource not in resource_set
            }
            if missing_resources:
                raise ValueError(f"operation {operation.id} references unknown resources {missing_resources}")
            missing_predecessors = set(operation.predecessors) - operation_set
            if missing_predecessors:
                raise ValueError(f"operation {operation.id} references unknown predecessors {missing_predecessors}")
            if operation.id in operation.predecessors:
                raise ValueError(f"operation {operation.id} cannot precede itself")
        for link in self.choice_links:
            missing_operations = set(link.operation_ids) - operation_set
            if missing_operations:
                raise ValueError(f"choice link {link.id} references unknown operations {missing_operations}")
            missing_modes = set(link.mode_keys) - mode_set
            if missing_modes:
                raise ValueError(f"choice link {link.id} references unknown modes {missing_modes}")
            for operation_id in link.operation_ids:
                operation = next(item for item in self.operations if item.id == operation_id)
                keys = {link.mode_keys[mode.id] for mode in operation.modes if mode.id in link.mode_keys}
                if not keys:
                    raise ValueError(f"choice link {link.id} has no keyed mode for {operation_id}")
        return self

    def operation_map(self) -> dict[str, Operation]:
        return {operation.id: operation for operation in self.operations}

    def resource_map(self) -> dict[str, Resource]:
        return {resource.id: resource for resource in self.resources}

    def mode_map(self) -> dict[str, tuple[Operation, Mode]]:
        return {
            mode.id: (operation, mode)
            for operation in self.operations
            for mode in operation.modes
        }


class Assignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: str
    mode_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def positive_interval(self) -> "Assignment":
        if self.end <= self.start:
            raise ValueError("assignment end must be greater than start")
        return self


class Schedule(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_id: str
    assignments: tuple[Assignment, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def assignment_map(self) -> dict[str, Assignment]:
        return {assignment.operation_id: assignment for assignment in self.assignments}

    @property
    def makespan(self) -> int:
        return max((assignment.end for assignment in self.assignments), default=0)
