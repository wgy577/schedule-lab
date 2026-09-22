from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .ir import Problem, Schedule


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    feasible: bool
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...] = ()
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "errors": [
                {
                    "code": item.code,
                    "message": item.message,
                    "entities": list(item.entities),
                }
                for item in self.errors
            ],
            "warnings": [
                {
                    "code": item.code,
                    "message": item.message,
                    "entities": list(item.entities),
                }
                for item in self.warnings
            ],
            "details": self.details or {},
        }


def validate_schedule(problem: Problem, schedule: Schedule) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    job_map = problem.job_map()
    resource_map = problem.resource_map()
    assignments = schedule.assignment_map()
    expected = set(operation_map)
    actual = [item.operation_id for item in schedule.assignments]
    duplicates = sorted(item for item, count in Counter(actual).items() if count > 1)
    missing = sorted(expected - set(actual))
    unknown = sorted(set(actual) - expected)
    if missing or duplicates or unknown:
        errors.append(
            ValidationIssue(
                "completeness",
                f"missing={missing}, duplicates={duplicates}, unknown={unknown}",
                tuple(missing + duplicates + unknown),
            )
        )
    if schedule.problem_id != problem.id:
        errors.append(
            ValidationIssue("problem_id", "schedule belongs to another problem")
        )
    by_resource: dict[str, list[Any]] = defaultdict(list)
    for assignment in schedule.assignments:
        operation = operation_map.get(assignment.operation_id)
        selected = mode_map.get(assignment.mode_id)
        if operation is None or selected is None:
            continue
        mode_operation, mode = selected
        if mode_operation.id != operation.id:
            errors.append(
                ValidationIssue(
                    "machine_eligibility",
                    f"{assignment.mode_id} is not eligible for {operation.id}",
                    (operation.id, assignment.mode_id),
                )
            )
            continue
        if assignment.end - assignment.start != mode.duration:
            errors.append(
                ValidationIssue(
                    "duration",
                    f"duration mismatch for {operation.id}",
                    (operation.id,),
                )
            )
        if assignment.start < max(operation.release, job_map[operation.job_id].release):
            errors.append(
                ValidationIssue(
                    "release",
                    f"{operation.id} starts before release",
                    (operation.id,),
                )
            )
        for predecessor in operation.predecessors:
            previous = assignments.get(predecessor)
            if previous is not None and assignment.start < previous.end:
                errors.append(
                    ValidationIssue(
                        "precedence",
                        f"{operation.id} starts before {predecessor} ends",
                        (predecessor, operation.id),
                    )
                )
        for resource in mode.resources:
            by_resource[resource].append(assignment)
            calendar = resource_map[resource].calendar
            if calendar and not any(
                assignment.start >= window_start
                and assignment.end <= window_end
                for window_start, window_end in calendar
            ):
                errors.append(
                    ValidationIssue(
                        "calendar",
                        f"{operation.id} lies outside {resource} availability",
                        (operation.id, resource),
                    )
                )
    for resource_id, values in by_resource.items():
        events = []
        for item in values:
            events.append((item.start, 1, item.operation_id))
            events.append((item.end, -1, item.operation_id))
        events.sort(key=lambda item: (item[0], item[1], item[2]))
        active = 0
        for time, delta, operation_id in events:
            active += delta
            if active > resource_map[resource_id].capacity:
                errors.append(
                    ValidationIssue(
                        "resource_capacity",
                        f"{resource_id} exceeds capacity at {time}",
                        (resource_id, operation_id),
                    )
                )
                break
    for link in problem.choice_links:
        selected_keys = set()
        for operation_id in link.operation_ids:
            assignment = assignments.get(operation_id)
            if assignment is None:
                continue
            key = link.mode_keys.get(assignment.mode_id)
            if key is not None:
                selected_keys.add(key)
        if len(selected_keys) > 1:
            errors.append(
                ValidationIssue(
                    "choice_link",
                    f"inconsistent choice link {link.id}: {selected_keys}",
                    link.operation_ids,
                )
            )
    for constraint in problem.constraints:
        if constraint.encoded_by != "core":
            continue
        scope = [
            assignments[item]
            for item in constraint.scope
            if item in assignments
        ]
        parameters = constraint.parameters
        if constraint.kind == "time_window":
            for assignment in scope:
                earliest = parameters.get("earliest_start")
                latest_start = parameters.get("latest_start")
                latest_end = parameters.get("latest_end")
                if earliest is not None and assignment.start < int(earliest):
                    errors.append(
                        ValidationIssue(
                            "time_window",
                            f"{assignment.operation_id} starts before {earliest}",
                            (constraint.id, assignment.operation_id),
                        )
                    )
                if latest_start is not None and assignment.start > int(latest_start):
                    errors.append(
                        ValidationIssue(
                            "time_window",
                            f"{assignment.operation_id} starts after {latest_start}",
                            (constraint.id, assignment.operation_id),
                        )
                    )
                if latest_end is not None and assignment.end > int(latest_end):
                    errors.append(
                        ValidationIssue(
                            "time_window",
                            f"{assignment.operation_id} ends after {latest_end}",
                            (constraint.id, assignment.operation_id),
                        )
                    )
        elif constraint.kind in {"no_wait", "transport", "blocking"}:
            for left, right in zip(scope, scope[1:]):
                lag = right.start - left.end
                if constraint.kind == "no_wait":
                    minimum = maximum = 0
                else:
                    minimum = int(parameters.get("min_lag", 0))
                    maximum = parameters.get("max_lag")
                if lag < minimum or (
                    maximum is not None and lag > int(maximum)
                ):
                    errors.append(
                        ValidationIssue(
                            constraint.kind,
                            f"lag {lag} violates [{minimum}, {maximum}]",
                            (constraint.id, left.operation_id, right.operation_id),
                        )
                    )
        elif constraint.kind == "binding":
            relation = str(parameters.get("relation", "same_resource"))
            if relation == "same_resource" and scope:
                selected = []
                for assignment in scope:
                    _, mode = mode_map[assignment.mode_id]
                    selected.append(set(mode.resources))
                if not set.intersection(*selected):
                    errors.append(
                        ValidationIssue(
                            "binding",
                            "bound operations do not share a resource",
                            (constraint.id, *constraint.scope),
                        )
                    )
            elif relation == "same_route" and len(
                {item.route_id for item in scope}
            ) > 1:
                errors.append(
                    ValidationIssue(
                        "binding",
                        "bound operations use different routes",
                        (constraint.id, *constraint.scope),
                    )
                )
        elif constraint.kind == "setup":
            resource_id = str(parameters.get("resource", ""))
            setup_matrix = parameters.get("matrix", {})
            ordered = sorted(
                by_resource.get(resource_id, []),
                key=lambda item: (item.start, item.end, item.operation_id),
            )
            for left, right in zip(ordered, ordered[1:]):
                _, left_mode = mode_map[left.mode_id]
                _, right_mode = mode_map[right.mode_id]
                key = f"{left_mode.setup_family}->{right_mode.setup_family}"
                required = int(setup_matrix.get(key, parameters.get("default", 0)))
                if right.start - left.end < required:
                    errors.append(
                        ValidationIssue(
                            "setup",
                            f"{resource_id} requires setup {required} before "
                            f"{right.operation_id}",
                            (constraint.id, left.operation_id, right.operation_id),
                        )
                    )
        elif constraint.kind not in {"precedence", "capacity", "calendar"}:
            warnings.append(
                ValidationIssue(
                    "unhandled_core_constraint",
                    f"no generic validator for {constraint.kind}",
                    (constraint.id,),
                )
            )
    oracle_constraints = [
        item.id for item in problem.constraints if item.encoded_by == "oracle"
    ]
    solver_constraints = [
        item.id for item in problem.constraints if item.encoded_by == "solver"
    ]
    return ValidationReport(
        feasible=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        details={
            "oracleRequiredConstraints": oracle_constraints,
            "solverCertifiedConstraints": solver_constraints,
        },
    )
