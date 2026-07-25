from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable

from .models import (
    CausalClosure,
    CausalInterventionPoint,
    CausalPath,
    DecisionType,
    DiagnosticPoint,
    EdgeType,
    ExperimentRecord,
    NodeType,
    ProjectSemantics,
    ResponsiblePoint,
    SchedulingGraph,
)


def _id(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _selected_resources(problem: Any, schedule: Any) -> dict[str, tuple[str, ...]]:
    mode_map = problem.mode_map()
    return {
        assignment.operation_id: tuple(mode_map[assignment.mode_id][1].resources)
        for assignment in schedule.assignments
    }


class ClosurePredictor:
    """Rule baseline for the minimal causal propagation closure."""

    def predict(
        self,
        *,
        problem: Any,
        schedule: Any,
        diagnostic: DiagnosticPoint,
        responsible: ResponsiblePoint,
        level: int = 1,
    ) -> CausalClosure:
        operation_map = problem.operation_map()
        assignment_map = schedule.assignment_map()
        selected_resources = _selected_resources(problem, schedule)
        by_job: dict[str, list[Any]] = defaultdict(list)
        by_resource: dict[str, list[Any]] = defaultdict(list)
        for operation in problem.operations:
            by_job[operation.job_id].append(operation)
        for assignment in schedule.assignments:
            for resource in selected_resources[assignment.operation_id]:
                by_resource[resource].append(assignment)
        for values in by_job.values():
            values.sort(key=lambda item: (item.index, item.id))
        for values in by_resource.values():
            values.sort(key=lambda item: (item.start, item.end, item.operation_id))

        center_operations = set(diagnostic.location)
        center_operations.add(responsible.operation_id)
        center_jobs = {
            operation_map[item].job_id
            for item in center_operations
            if item in operation_map
        }
        responsible_stage = operation_map[responsible.operation_id].index
        closure = {
            operation.id
            for job in center_jobs
            for operation in by_job[job]
            if operation.index >= (responsible_stage if level == 1 else 0)
        }
        resource_ids = {
            resource
            for operation_id in closure
            for resource in selected_resources[operation_id]
        }
        frontier = set(closure)
        for _ in range(max(0, level - 1)):
            expanded = set(frontier)
            for operation_id in frontier:
                operation = operation_map[operation_id]
                expanded.update(operation.predecessors)
                expanded.update(
                    item.id
                    for item in by_job[operation.job_id]
                    if abs(item.index - operation.index) <= 1
                )
                for resource in selected_resources[operation_id]:
                    assignments = by_resource[resource]
                    index = next(
                        i
                        for i, assignment in enumerate(assignments)
                        if assignment.operation_id == operation_id
                    )
                    expanded.update(
                        assignment.operation_id
                        for assignment in assignments[max(0, index - 1) : index + 2]
                    )
            frontier = expanded - closure
            closure.update(expanded)
        all_operations = {operation.id for operation in problem.operations}
        boundary_edges = 0
        for operation_id in closure:
            operation = operation_map[operation_id]
            boundary_edges += sum(predecessor not in closure for predecessor in operation.predecessors)
            for resource in selected_resources[operation_id]:
                boundary_edges += sum(
                    assignment.operation_id not in closure
                    for assignment in by_resource[resource]
                    if not (
                        assignment.end <= assignment_map[operation_id].start
                        or assignment.start >= assignment_map[operation_id].end
                    )
                )
        risk = min(1.0, boundary_edges / max(1, len(closure)))
        return CausalClosure(
            operation_ids=tuple(sorted(closure)),
            resource_ids=tuple(sorted(resource_ids)),
            level=level,
            predicted_outside_risk=risk,
            reason=(
                "responsible-job suffix and diagnostic boundary"
                if level == 1
                else "expanded through precedence and shared-resource neighbors"
            ),
        )


class CIPRanker:
    """Strong deterministic baseline before learned GNN/MLP ranking."""

    def __init__(self, history: Iterable[ExperimentRecord] = ()) -> None:
        self.history = tuple(history)

    def _posterior(self, operators: tuple[str, ...]) -> tuple[float, float]:
        observations = [
            record
            for record in self.history
            if record.action.operator in operators
        ]
        successes = sum(record.accepted for record in observations)
        validity = (successes + 1.0) / (len(observations) + 2.0)
        uncertainty = 1.0 / (len(observations) + 1.0) ** 0.5
        return validity, uncertainty

    def rank(self, candidates: Iterable[CausalInterventionPoint]) -> list[CausalInterventionPoint]:
        ranked = []
        for candidate in candidates:
            validity, uncertainty = self._posterior(candidate.recommended_operators)
            closure_cost = len(candidate.closure.operation_ids)
            risk_cost = 1.0 + 4.0 * candidate.closure.predicted_outside_risk
            predicted_cost = max(1.0, closure_cost * risk_cost)
            predicted_gain = candidate.diagnostic.magnitude * (
                0.5 + candidate.responsible.modifiability
            )
            score = (
                max(0.0, predicted_gain)
                * validity
                / predicted_cost
                + 0.15 * uncertainty
            )
            ranked.append(
                candidate.model_copy(
                    update={
                        "predicted_improvement": predicted_gain,
                        "predicted_validity": validity,
                        "predicted_cost": predicted_cost,
                        "uncertainty": uncertainty,
                        "score": score,
                    }
                )
            )
        return sorted(
            ranked,
            key=lambda item: (
                int(item.diagnostic.evidence.get("semanticPriority", 5)),
                -item.score,
                -item.diagnostic.magnitude,
                item.id,
            ),
        )


class CausalCoreDiscoverer:
    def __init__(
        self,
        semantics: ProjectSemantics,
        *,
        closure_predictor: ClosurePredictor | None = None,
        ranker: CIPRanker | None = None,
    ) -> None:
        self.semantics = semantics
        self.closure_predictor = closure_predictor or ClosurePredictor()
        self.ranker = ranker or CIPRanker()

    def discover(
        self,
        *,
        problem: Any,
        schedule: Any,
        graph: SchedulingGraph,
        top_k: int = 8,
        closure_level: int = 1,
    ) -> list[CausalInterventionPoint]:
        operation_map = problem.operation_map()
        assignment_map = schedule.assignment_map()
        resource_map = problem.resource_map()
        selected_resources = _selected_resources(problem, schedule)
        graph_nodes = graph.node_map()
        by_resource: dict[str, list[Any]] = defaultdict(list)
        by_job: dict[str, list[Any]] = defaultdict(list)
        for operation in problem.operations:
            by_job[operation.job_id].append(operation)
        for assignment in schedule.assignments:
            for resource in selected_resources[assignment.operation_id]:
                by_resource[resource].append(assignment)
        for values in by_job.values():
            values.sort(key=lambda item: (item.index, item.id))
        for values in by_resource.values():
            values.sort(key=lambda item: (item.start, item.end, item.operation_id))

        responsibility_stages = dict(
            self.semantics.metadata.get(
                "responsibilityStageByDiagnostic",
                {},
            )
        )
        operators_by_diagnostic = dict(
            self.semantics.metadata.get(
                "operatorsByDiagnostic",
                {},
            )
        )
        preferred_global_stage = responsibility_stages.get("global_sink_gap")
        final_stage = max(
            (operation.index for operation in problem.operations),
            default=0,
        )
        raw: list[CausalInterventionPoint] = []
        for resource_id, assignments in sorted(by_resource.items()):
            resource = resource_map[resource_id]
            if resource.capacity != 1:
                continue
            tags = set(resource.tags)
            is_global_sink = "global_launch" in tags
            for left, right in zip(assignments, assignments[1:]):
                gap = right.start - left.end
                if gap <= 0:
                    continue
                right_operation = operation_map[right.operation_id]
                job_operations = by_job[right_operation.job_id]
                is_sink_stage = (
                    right_operation.index == final_stage
                    and problem.kind in {"FSP", "HFSP"}
                )
                if is_global_sink and preferred_global_stage is not None:
                    eligible = [
                        operation
                        for operation in job_operations
                        if operation.index == int(preferred_global_stage)
                    ]
                else:
                    eligible = [
                        operation
                        for operation in job_operations
                        if operation.index < right_operation.index
                    ]
                if not eligible:
                    eligible = [right_operation]
                responsible_operation = max(
                    eligible,
                    key=lambda operation: (
                        float(graph_nodes[operation.id].features.get("wait", 0)),
                        float(graph_nodes[operation.id].features.get("resource_idle_before", 0)),
                        operation.index,
                    ),
                )
                modifiability = min(
                    1.0,
                    0.25
                    + 0.25 * len(responsible_operation.modes)
                    + 0.05 * float(
                        graph_nodes[responsible_operation.id].features.get("wait", 0)
                    ),
                )
                diagnostic = DiagnosticPoint(
                    id=f"gap:{resource_id}:{left.operation_id}:{right.operation_id}",
                    type=(
                        "global_sink_gap"
                        if is_global_sink
                        else "sink_stage_gap"
                        if is_sink_stage
                        else "resource_idle_gap"
                    ),
                    location=(left.operation_id, right.operation_id),
                    resource_id=resource_id,
                    window=(float(left.end), float(right.start)),
                    magnitude=float(gap) / getattr(problem, "time_scale", 1),
                    evidence={
                        "leftEnd": left.end,
                        "rightStart": right.start,
                        "resourceTags": sorted(tags),
                        "problemFamily": problem.kind,
                        "semanticPriority": (
                            0 if is_global_sink or is_sink_stage else 3
                        ),
                    },
                )
                responsible = ResponsiblePoint(
                    operation_id=responsible_operation.id,
                    decision_type=DecisionType.SEQUENCE,
                    reason=(
                        f"project-declared upstream stage O{int(preferred_global_stage) + 1}"
                        if is_global_sink and preferred_global_stage is not None
                        else "largest modifiable upstream waiting contribution to sink arrival"
                        if is_sink_stage
                        else "largest observable upstream waiting contribution"
                    ),
                    modifiability=modifiability,
                )
                path_operations = [
                    operation.id
                    for operation in job_operations
                    if responsible_operation.index <= operation.index <= right_operation.index
                ]
                path = CausalPath(
                    nodes=tuple(path_operations + [resource_id]),
                    edge_types=tuple(
                        [EdgeType.PRECEDENCE] * max(0, len(path_operations) - 1)
                        + [EdgeType.ELIGIBILITY]
                    ),
                    explanation=(
                        f"{responsible_operation.id} can change arrival of "
                        f"{right.operation_id} and the idle interval on {resource_id}"
                    ),
                )
                closure = self.closure_predictor.predict(
                    problem=problem,
                    schedule=schedule,
                    diagnostic=diagnostic,
                    responsible=responsible,
                    level=closure_level,
                )
                configured_operators = operators_by_diagnostic.get(
                    diagnostic.type
                )
                if configured_operators:
                    operators = tuple(configured_operators)
                elif is_sink_stage:
                    operators = (
                        "stage_resequence",
                        "insertion",
                        "blocking_chain_repair",
                        "expand_closure",
                    )
                else:
                    operators = (
                        "adjacent_swap",
                        "insertion",
                        "machine_reassignment",
                        "critical_block_resequence",
                        "expand_closure",
                    )
                raw.append(
                    CausalInterventionPoint(
                        id=_id(
                            {
                                "diagnostic": diagnostic.id,
                                "responsible": responsible.operation_id,
                                "closure": closure.operation_ids,
                            }
                        ),
                        diagnostic=diagnostic,
                        responsible=responsible,
                        causal_path=path,
                        closure=closure,
                        recommended_operators=tuple(
                            item
                            for item in operators
                            if item in self.semantics.allowed_interventions
                        ),
                    )
                )

        # Zero-gap adjacent operations are critical machine blocks, especially
        # for JSP.  They were invisible to an idle-gap-only detector.
        for resource_id, assignments in sorted(by_resource.items()):
            if resource_map[resource_id].capacity != 1:
                continue
            for left, right in zip(assignments, assignments[1:]):
                if right.start != left.end:
                    continue
                left_operation = operation_map[left.operation_id]
                right_operation = operation_map[right.operation_id]
                diagnostic = DiagnosticPoint(
                    id=f"block:{resource_id}:{left.operation_id}:{right.operation_id}",
                    type="critical_resource_block",
                    location=(left.operation_id, right.operation_id),
                    resource_id=resource_id,
                    window=(float(left.start), float(right.end)),
                    magnitude=float(
                        (left.end - left.start) + (right.end - right.start)
                    )
                    / getattr(problem, "time_scale", 1),
                    evidence={
                        "problemFamily": problem.kind,
                        "semanticPriority": 0 if problem.kind == "JSP" else 2,
                    },
                )
                responsible_operation = right_operation
                responsible = ResponsiblePoint(
                    operation_id=responsible_operation.id,
                    decision_type=DecisionType.SEQUENCE,
                    reason="adjacent operation participates in a capacity-one critical block",
                    modifiability=min(
                        1.0,
                        0.25 + 0.25 * len(responsible_operation.modes),
                    ),
                )
                closure = self.closure_predictor.predict(
                    problem=problem,
                    schedule=schedule,
                    diagnostic=diagnostic,
                    responsible=responsible,
                    level=closure_level,
                )
                path = CausalPath(
                    nodes=(left_operation.id, right_operation.id, resource_id),
                    edge_types=(
                        EdgeType.RESOURCE_SEQUENCE,
                        EdgeType.ELIGIBILITY,
                    ),
                    explanation="adjacent order controls the critical resource block and downstream completion",
                )
                raw.append(
                    CausalInterventionPoint(
                        id=_id(
                            {
                                "diagnostic": diagnostic.id,
                                "responsible": responsible.operation_id,
                                "closure": closure.operation_ids,
                            }
                        ),
                        diagnostic=diagnostic,
                        responsible=responsible,
                        causal_path=path,
                        closure=closure,
                        recommended_operators=tuple(
                            item
                            for item in (
                                "adjacent_swap",
                                "insertion",
                                "critical_block_resequence",
                                "expand_closure",
                            )
                            if item in self.semantics.allowed_interventions
                        ),
                    )
                )

        # FJSP/HFSP routing signal: an operation on a relatively loaded
        # resource with at least one eligible lower-load alternative.
        if problem.kind in {"FJSP", "HFSP"}:
            resource_load = {
                resource_id: sum(item.end - item.start for item in assignments)
                / max(
                    1.0,
                    schedule.makespan * resource_map[resource_id].capacity,
                )
                for resource_id, assignments in by_resource.items()
            }
            for assignment in schedule.assignments:
                operation = operation_map[assignment.operation_id]
                if len(operation.modes) < 2:
                    continue
                current_resources = selected_resources[operation.id]
                current_load = max(
                    (resource_load.get(item, 0.0) for item in current_resources),
                    default=0.0,
                )
                alternatives = [
                    mode
                    for mode in operation.modes
                    if mode.id != assignment.mode_id
                    and max(
                        (resource_load.get(item, 0.0) for item in mode.resources),
                        default=0.0,
                    )
                    + 1e-9
                    < current_load
                ]
                if not alternatives:
                    continue
                diagnostic = DiagnosticPoint(
                    id=f"imbalance:{operation.id}:{assignment.mode_id}",
                    type="flexible_resource_imbalance",
                    location=(operation.id,),
                    resource_id=current_resources[0] if current_resources else None,
                    window=(float(assignment.start), float(assignment.end)),
                    magnitude=current_load
                    - min(
                        max(
                            (
                                resource_load.get(resource, 0.0)
                                for resource in mode.resources
                            ),
                            default=0.0,
                        )
                        for mode in alternatives
                    ),
                    evidence={
                        "problemFamily": problem.kind,
                        "alternativeModes": [mode.id for mode in alternatives],
                        "semanticPriority": 0 if problem.kind == "FJSP" else 1,
                    },
                )
                responsible = ResponsiblePoint(
                    operation_id=operation.id,
                    decision_type=DecisionType.RESOURCE,
                    reason="eligible alternative has lower observed resource-family load",
                    modifiability=min(1.0, 0.25 + 0.25 * len(operation.modes)),
                )
                closure = self.closure_predictor.predict(
                    problem=problem,
                    schedule=schedule,
                    diagnostic=diagnostic,
                    responsible=responsible,
                    level=closure_level,
                )
                raw.append(
                    CausalInterventionPoint(
                        id=_id(
                            {
                                "diagnostic": diagnostic.id,
                                "responsible": operation.id,
                                "closure": closure.operation_ids,
                            }
                        ),
                        diagnostic=diagnostic,
                        responsible=responsible,
                        causal_path=CausalPath(
                            nodes=(operation.id,),
                            edge_types=(),
                            explanation="route choice changes load and downstream timing",
                        ),
                        closure=closure,
                        recommended_operators=tuple(
                            item
                            for item in (
                                "machine_reassignment",
                                "insertion",
                                "expand_closure",
                            )
                            if item in self.semantics.allowed_interventions
                        ),
                    )
                )

        # High-wait recall prevents the framework from depending on sink gaps alone.
        for node in graph.nodes:
            if node.type != NodeType.OPERATION:
                continue
            wait = float(node.features.get("wait", 0))
            if wait <= 0:
                continue
            operation = operation_map[node.id]
            diagnostic = DiagnosticPoint(
                id=f"wait:{node.id}",
                type="high_wait",
                location=(node.id,),
                magnitude=wait / getattr(problem, "time_scale", 1),
                evidence={
                    "wait": wait,
                    "problemFamily": problem.kind,
                    "semanticPriority": 4,
                },
            )
            responsible = ResponsiblePoint(
                operation_id=node.id,
                decision_type=(
                    DecisionType.RESOURCE if len(operation.modes) > 1 else DecisionType.SEQUENCE
                ),
                reason="operation carries observed waiting and remains locally modifiable",
                modifiability=min(1.0, 0.25 + 0.25 * len(operation.modes)),
            )
            closure = self.closure_predictor.predict(
                problem=problem,
                schedule=schedule,
                diagnostic=diagnostic,
                responsible=responsible,
                level=closure_level,
            )
            raw.append(
                CausalInterventionPoint(
                    id=_id({"diagnostic": diagnostic.id, "closure": closure.operation_ids}),
                    diagnostic=diagnostic,
                    responsible=responsible,
                    causal_path=CausalPath(
                        nodes=(node.id,),
                        edge_types=(),
                        explanation="observed wait is used as a recall signal; intervention must verify causality",
                    ),
                    closure=closure,
                    recommended_operators=tuple(
                        item
                        for item in (
                            "machine_reassignment",
                            "insertion",
                            "critical_block_resequence",
                            "expand_closure",
                        )
                        if item in self.semantics.allowed_interventions
                    ),
                )
            )
        unique = {candidate.id: candidate for candidate in raw}
        return self.ranker.rank(unique.values())[:top_k]
