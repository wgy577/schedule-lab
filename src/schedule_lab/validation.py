from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .model import Problem, Schedule


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning"]
    code: str
    message: str
    operations: tuple[str, ...] = ()


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    feasible: bool
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")


def validate_schedule(problem: Problem, schedule: Schedule) -> ValidationResult:
    issues: list[ValidationIssue] = []
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    resource_map = problem.resource_map()
    assignments = schedule.assignment_map()
    counts = Counter(assignment.operation_id for assignment in schedule.assignments)

    if schedule.problem_id != problem.id:
        issues.append(ValidationIssue(severity="error", code="problem_id", message="schedule belongs to another problem"))
    for operation_id in operation_map:
        if counts[operation_id] == 0:
            issues.append(ValidationIssue(severity="error", code="missing_operation", message=f"missing {operation_id}", operations=(operation_id,)))
        elif counts[operation_id] > 1:
            issues.append(ValidationIssue(severity="error", code="duplicate_operation", message=f"duplicate {operation_id}", operations=(operation_id,)))
    for assignment in schedule.assignments:
        if assignment.operation_id not in operation_map:
            issues.append(ValidationIssue(severity="error", code="unknown_operation", message=f"unknown {assignment.operation_id}"))
            continue
        mode_entry = mode_map.get(assignment.mode_id)
        if mode_entry is None or mode_entry[0].id != assignment.operation_id:
            issues.append(ValidationIssue(severity="error", code="invalid_mode", message=f"{assignment.mode_id} is not a mode of {assignment.operation_id}", operations=(assignment.operation_id,)))
            continue
        operation, mode = mode_entry
        if assignment.end - assignment.start != mode.duration:
            issues.append(ValidationIssue(severity="error", code="duration", message=f"{assignment.operation_id} duration does not match {assignment.mode_id}", operations=(assignment.operation_id,)))
        if assignment.start < operation.release:
            issues.append(ValidationIssue(severity="error", code="release", message=f"{assignment.operation_id} starts before release", operations=(assignment.operation_id,)))

    for operation in problem.operations:
        successor = assignments.get(operation.id)
        if successor is None:
            continue
        for predecessor_id in operation.predecessors:
            predecessor = assignments.get(predecessor_id)
            if predecessor is not None and successor.start < predecessor.end:
                issues.append(ValidationIssue(severity="error", code="precedence", message=f"{operation.id} starts before {predecessor_id} ends", operations=(predecessor_id, operation.id)))

    intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for assignment in schedule.assignments:
        mode_entry = mode_map.get(assignment.mode_id)
        if mode_entry is None:
            continue
        for resource_id in mode_entry[1].resources:
            intervals[resource_id].append((assignment.start, assignment.end, assignment.operation_id))
    for resource_id, resource_intervals in intervals.items():
        events: list[tuple[int, int, str]] = []
        for start, end, operation_id in resource_intervals:
            events.extend(((start, 1, operation_id), (end, -1, operation_id)))
        active = 0
        active_operations: set[str] = set()
        for _, delta, operation_id in sorted(events, key=lambda item: (item[0], item[1])):
            if delta < 0:
                active -= 1
                active_operations.discard(operation_id)
            else:
                active += 1
                active_operations.add(operation_id)
                if active > resource_map[resource_id].capacity:
                    issues.append(ValidationIssue(severity="error", code="resource_capacity", message=f"{resource_id} exceeds capacity {resource_map[resource_id].capacity}", operations=tuple(sorted(active_operations))))
                    break

    for link in problem.choice_links:
        selected_keys: dict[str, str] = {}
        for operation_id in link.operation_ids:
            assignment = assignments.get(operation_id)
            if assignment is not None and assignment.mode_id in link.mode_keys:
                selected_keys[operation_id] = link.mode_keys[assignment.mode_id]
        if len(set(selected_keys.values())) > 1:
            issues.append(ValidationIssue(severity="error", code="choice_link", message=f"choice link {link.id} selected inconsistent resource families", operations=tuple(selected_keys)))

    if problem.metadata.get("requires_domain_validation") and not schedule.metadata.get("domain_validated"):
        issues.append(ValidationIssue(severity="warning", code="domain_validation", message="core constraints passed, but domain-specific trajectory/collision validation is not certified"))
    return ValidationResult(feasible=not any(issue.severity == "error" for issue in issues), issues=tuple(issues))
