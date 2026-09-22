from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from .ir import ParameterSymbol, SymbolicExpression
from .models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    SchedulingGraph,
    SymbolicConstraint,
)


def _symbol(name: str, *indices: str) -> SymbolicExpression:
    return SymbolicExpression(op="symbol", symbol=name, indices=tuple(indices))


def _constant(value: float | int | bool | str) -> SymbolicExpression:
    return SymbolicExpression(op="constant", value=value)


def _expr(op: str, *args: SymbolicExpression) -> SymbolicExpression:
    return SymbolicExpression(op=op, args=tuple(args))  # type: ignore[arg-type]


def _opaque(rendered: str) -> SymbolicExpression:
    return SymbolicExpression(op="opaque", value=rendered)


def _edge_id(
    edge_type: EdgeType,
    source: str,
    target: str,
    discriminator: str = "",
) -> str:
    raw = f"{edge_type.value}|{source}|{target}|{discriminator}"
    return "edge:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _constraint_edge_type(kind: str) -> EdgeType:
    return {
        "setup": EdgeType.SETUP,
        "blocking": EdgeType.BLOCKING,
        "no_wait": EdgeType.NO_WAIT,
        "transport": EdgeType.TRANSPORT,
        "time_window": EdgeType.TIME_WINDOW,
        "calendar": EdgeType.CALENDAR,
        "capacity": EdgeType.CAPACITY,
        "binding": EdgeType.BINDING,
    }.get(kind, EdgeType.CONSTRAINT_SCOPE)


_CANONICAL_SYMBOLS = (
    ParameterSymbol(name="S", latex="S_o", meaning="operation start time", unit="time"),
    ParameterSymbol(name="C", latex="C_o", meaning="operation completion time", unit="time"),
    ParameterSymbol(name="p", latex="p_{ok}", meaning="mode-dependent processing time", unit="time"),
    ParameterSymbol(name="x", latex="x_{ok}", meaning="mode-selection indicator", domain="binary"),
    ParameterSymbol(name="y", latex="y_{uvr}", meaning="resource-order orientation indicator", domain="binary"),
    ParameterSymbol(name="r", latex="r_o", meaning="release time", unit="time"),
    ParameterSymbol(name="cap", latex="cap_r", meaning="resource capacity", domain="integer", unit="capacity"),
    ParameterSymbol(name="M", latex="M", meaning="valid big-M bound", unit="time"),
)


def _selected_resources(problem: Any, schedule: Any) -> dict[str, tuple[str, ...]]:
    mode_map = problem.mode_map()
    return {
        assignment.operation_id: tuple(mode_map[assignment.mode_id][1].resources)
        for assignment in schedule.assignments
    }


def _add_explicit_constraints(
    *,
    problem: Any,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    constraints: list[SymbolicConstraint],
    symbols: dict[str, ParameterSymbol],
    views: dict[str, set[str]],
) -> None:
    known_nodes = {node.id for node in nodes}
    for spec in getattr(problem, "constraints", ()):
        node_id = f"constraint:{spec.id}"
        rendered = spec.rendered_expression or (
            f"{spec.kind}({', '.join(spec.scope)}; "
            f"{json.dumps(spec.parameters, ensure_ascii=False, sort_keys=True)})"
        )
        expression = spec.expression or _opaque(rendered)
        representation_status = (
            "explicit_ast" if spec.expression is not None else "parameterized_opaque"
        )
        for symbol in spec.parameter_symbols:
            symbols.setdefault(symbol.name, symbol)
        symbolic = SymbolicConstraint(
            id=spec.id,
            kind=spec.kind,
            scope=tuple(spec.scope),
            expression=expression,
            rendered_expression=rendered,
            parameter_symbols=tuple(spec.parameter_symbols),
            parameter_bindings=dict(spec.parameters),
            hard=spec.hard,
            active=spec.active,
            activation_condition=spec.activation_condition,
            encoded_by=spec.encoded_by,
            provenance="problem_ir",
            representation_status=representation_status,
        )
        constraints.append(symbolic)
        nodes.append(
            GraphNode(
                id=node_id,
                type=NodeType.CONSTRAINT,
                features={
                    "constraint_id": spec.id,
                    "kind": spec.kind,
                    "hard": spec.hard,
                    "encoded_by": spec.encoded_by,
                    "parameters": spec.parameters,
                    "parameter_symbols": [
                        item.model_dump(mode="json") for item in spec.parameter_symbols
                    ],
                    "expression": expression.model_dump(mode="json"),
                    "rendered_expression": rendered,
                    "activation_condition": spec.activation_condition,
                    "representation_status": representation_status,
                },
                active=spec.active,
                provenance="problem_ir",
            )
        )
        views["constraints"].add(node_id)
        relation = _constraint_edge_type(spec.kind)
        for target in spec.scope:
            if target not in known_nodes:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(relation, node_id, target, spec.id),
                    source=node_id,
                    target=target,
                    type=relation,
                    features={
                        "relation": "applies_to",
                        "constraint_kind": spec.kind,
                    },
                    active=spec.active,
                    constraint_id=spec.id,
                    provenance="problem_ir",
                )
            )


def _add_dynamic_events(
    *,
    problem: Any,
    schedule: Any,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    views: dict[str, set[str]],
) -> None:
    known_nodes = {node.id for node in nodes}
    state_time = int(schedule.metadata.get("state_time", schedule.makespan))
    for event in getattr(problem, "events", ()):
        node_id = f"event:{event.id}"
        active = bool(event.active and event.time <= state_time)
        nodes.append(
            GraphNode(
                id=node_id,
                type=NodeType.EVENT,
                features={
                    "event_id": event.id,
                    "kind": event.kind,
                    "time": event.time,
                    "parameters": event.parameters,
                    "state_time": state_time,
                },
                active=active,
                provenance="problem_ir",
            )
        )
        views["events"].add(node_id)
        relation = (
            EdgeType.DEACTIVATES
            if event.kind == "resource_breakdown"
            else EdgeType.ACTIVATES
            if event.kind in {"resource_recovery", "job_arrival"}
            else EdgeType.EVENT_AFFECTS
        )
        for target in event.scope:
            if target not in known_nodes:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(relation, node_id, target, event.id),
                    source=node_id,
                    target=target,
                    type=relation,
                    features={"event_kind": event.kind, "event_time": event.time},
                    active=active,
                    provenance="problem_ir",
                )
            )


def _add_objectives_and_decisions(
    *,
    problem: Any,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    views: dict[str, set[str]],
) -> None:
    for objective in problem.objective:
        objective_id = f"objective:{objective.name}"
        nodes.append(
            GraphNode(
                id=objective_id,
                type=NodeType.OBJECTIVE,
                features={
                    "name": objective.name,
                    "sense": objective.sense,
                    "weight": objective.weight,
                    "tolerance": objective.tolerance,
                },
                provenance="problem_ir",
            )
        )
        views["objectives"].add(objective_id)
        if objective.name == "makespan":
            targets = (f"sink:{problem.id}",)
        elif "tardiness" in objective.name or "flow" in objective.name:
            targets = tuple(job.id for job in problem.jobs)
        else:
            targets = tuple(operation.id for operation in problem.operations)
        for target in targets:
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.OBJECTIVE_DEPENDS_ON,
                        objective_id,
                        target,
                    ),
                    source=objective_id,
                    target=target,
                    type=EdgeType.OBJECTIVE_DEPENDS_ON,
                    provenance="problem_ir",
                )
            )

    for operation in problem.operations:
        decision_id = f"decision:mode:{operation.id}"
        nodes.append(
            GraphNode(
                id=decision_id,
                type=NodeType.DECISION,
                features={
                    "decision_type": "resource_or_mode_assignment",
                    "operation_id": operation.id,
                    "alternative_count": len(operation.modes),
                },
                provenance="derived",
            )
        )
        views["decisions"].add(decision_id)
        for target in (operation.id, *(mode.id for mode in operation.modes)):
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.DECISION_CONTROLS,
                        decision_id,
                        target,
                    ),
                    source=decision_id,
                    target=target,
                    type=EdgeType.DECISION_CONTROLS,
                    provenance="derived",
                )
            )
    for resource in problem.resources:
        decision_id = f"decision:sequence:{resource.id}"
        nodes.append(
            GraphNode(
                id=decision_id,
                type=NodeType.DECISION,
                features={
                    "decision_type": "resource_sequence",
                    "resource_id": resource.id,
                },
                provenance="derived",
            )
        )
        views["decisions"].add(decision_id)
        edges.append(
            GraphEdge(
                id=_edge_id(
                    EdgeType.DECISION_CONTROLS,
                    decision_id,
                    resource.id,
                ),
                source=decision_id,
                target=resource.id,
                type=EdgeType.DECISION_CONTROLS,
                provenance="derived",
            )
        )

    policy = problem.metadata.get("solver_policy")
    if policy is not None:
        policy_id = "solver_policy:active"
        nodes.append(
            GraphNode(
                id=policy_id,
                type=NodeType.SOLVER_POLICY,
                features={
                    "policy": policy
                    if isinstance(policy, (dict, list, str, int, float, bool))
                    else repr(policy)
                },
                provenance="project_evidence",
            )
        )
        views["decisions"].add(policy_id)
        for decision_id in tuple(views["decisions"]):
            if decision_id == policy_id:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.PROJECT_BINDING,
                        policy_id,
                        decision_id,
                        "governs",
                    ),
                    source=policy_id,
                    target=decision_id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "governs"},
                    provenance="project_evidence",
                )
            )


def _convert_formula_expr(value: Any) -> SymbolicExpression:
    op = value.op.value if hasattr(value.op, "value") else str(value.op)
    if op == "symbol":
        return _symbol(str(value.symbol))
    if op == "constant":
        return _constant(value.value)
    mapping = {
        "subtract": "subtract",
        "divide": "divide",
        "add": "add",
        "multiply": "multiply",
        "sum": "sum",
        "max": "max",
        "min": "min",
        "mean": "mean",
        "std": "std",
        "cv": "cv",
        "abs": "abs",
    }
    return _expr(mapping[op], *(_convert_formula_expr(item) for item in value.args))


def _add_diagnostic_evidence(
    *,
    diagnostic_artifacts: Any,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    constraints: list[SymbolicConstraint],
    symbols: dict[str, ParameterSymbol],
    views: dict[str, set[str]],
) -> None:
    profile = diagnostic_artifacts.diagnostic_semantics_profile
    program = diagnostic_artifacts.measurement_program
    for definition in profile.symbol_table:
        symbols.setdefault(
            definition.symbol,
            ParameterSymbol(
                name=definition.symbol,
                latex=definition.latex,
                meaning=definition.meaning,
                unit=definition.unit,
            ),
        )
    for group in (
        *profile.entity_groups,
        *profile.resource_comparison_groups,
    ):
        members = tuple(
            getattr(group, "resolved_member_ids", ())
            or getattr(group, "resolved_resource_ids", ())
        )
        nodes.append(
            GraphNode(
                id=group.group_id,
                type=NodeType.ENTITY_GROUP,
                features=group.model_dump(mode="json"),
                provenance="project_evidence",
            )
        )
        views["evidence"].add(group.group_id)
        known_nodes = {node.id for node in nodes}
        for member in members:
            if member not in known_nodes:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.GROUP_MEMBERSHIP,
                        group.group_id,
                        member,
                    ),
                    source=group.group_id,
                    target=member,
                    type=EdgeType.GROUP_MEMBERSHIP,
                    provenance="project_evidence",
                )
            )
    for item in profile.constraint_instances:
        node_id = f"constraint:{item.constraint_id}"
        nodes.append(
            GraphNode(
                id=node_id,
                type=NodeType.CONSTRAINT,
                features=item.model_dump(mode="json"),
                provenance="semantic_prior",
            )
        )
        views["evidence"].add(node_id)
        constraints.append(
            SymbolicConstraint(
                id=item.constraint_id,
                kind=item.kind,
                scope=tuple(item.entity_group_ids),
                expression=_opaque(item.exact_statement),
                rendered_expression=item.exact_statement,
                parameter_symbols=tuple(
                    symbols[symbol]
                    for symbol in item.parameter_symbols
                    if symbol in symbols
                ),
                hard=item.hard,
                encoded_by="derived",
                provenance="semantic_prior",
                representation_status="parameterized_opaque",
            )
        )
        for group_id in item.entity_group_ids:
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.CONSTRAINT_SCOPE,
                        node_id,
                        group_id,
                        item.constraint_id,
                    ),
                    source=node_id,
                    target=group_id,
                    type=EdgeType.CONSTRAINT_SCOPE,
                    constraint_id=item.constraint_id,
                    provenance="semantic_prior",
                )
            )
    for decision in profile.decisions:
        nodes.append(
            GraphNode(
                id=decision.decision_id,
                type=NodeType.DECISION,
                features=decision.model_dump(mode="json"),
                provenance="semantic_prior",
            )
        )
        views["evidence"].add(decision.decision_id)
        for group_id in decision.affected_group_ids:
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.DECISION_CONTROLS,
                        decision.decision_id,
                        group_id,
                    ),
                    source=decision.decision_id,
                    target=group_id,
                    type=EdgeType.DECISION_CONTROLS,
                    provenance="semantic_prior",
                )
            )
    for seed in profile.structural_pressure_seeds:
        nodes.append(
            GraphNode(
                id=seed.seed_id,
                type=NodeType.STRUCTURAL_PRESSURE,
                features=seed.model_dump(mode="json"),
                provenance="semantic_prior",
            )
        )
        views["evidence"].add(seed.seed_id)
        targets = (
            *seed.exact_entity_ids,
            *seed.entity_group_ids,
            *(f"constraint:{item}" for item in seed.constraint_ids),
        )
        known_nodes = {node.id for node in nodes}
        for target in targets:
            if target not in known_nodes:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.STRUCTURAL_EXPOSURE,
                        seed.seed_id,
                        target,
                    ),
                    source=seed.seed_id,
                    target=target,
                    type=EdgeType.STRUCTURAL_EXPOSURE,
                    provenance="semantic_prior",
                )
            )
    for target in program.targets:
        expression = _convert_formula_expr(target.formula)
        nodes.append(
            GraphNode(
                id=target.target_id,
                type=NodeType.MEASUREMENT,
                features={
                    **target.model_dump(mode="json", exclude={"formula"}),
                    "formula": expression.model_dump(mode="json"),
                },
                provenance="semantic_prior",
            )
        )
        views["evidence"].add(target.target_id)
        known_nodes = {node.id for node in nodes}
        for bound in (*target.entity_group_ids, *target.exact_entity_ids):
            if bound not in known_nodes:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(EdgeType.MEASURES, target.target_id, bound),
                    source=target.target_id,
                    target=bound,
                    type=EdgeType.MEASURES,
                    provenance="semantic_prior",
                )
            )


def _add_project_constraint_evidence(
    *,
    project_constraint_graph: Any,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    views: dict[str, set[str]],
) -> None:
    """Merge the audited semantic graph as a non-causal evidence overlay."""

    existing = {node.id for node in nodes}
    id_map: dict[str, str] = {}
    node_type_map = {
        "Job": NodeType.JOB,
        "Operation": NodeType.OPERATION,
        "ProcessingMode": NodeType.MODE,
        "Resource": NodeType.RESOURCE,
        "Constraint": NodeType.CONSTRAINT,
        "Objective": NodeType.OBJECTIVE,
        "Decision": NodeType.DECISION,
        "CodeEvidence": NodeType.EVIDENCE,
    }
    resolvable = {"Job", "Operation", "ProcessingMode", "Resource"}
    for item in project_constraint_graph.nodes:
        target_id = (
            item.label
            if item.node_type in resolvable and item.label in existing
            else item.id
        )
        id_map[item.id] = target_id
        if target_id in existing:
            continue
        nodes.append(
            GraphNode(
                id=target_id,
                type=node_type_map.get(item.node_type, NodeType.EVIDENCE),
                features={
                    "semantic_node_type": item.node_type,
                    "label": item.label,
                    **item.attributes,
                },
                provenance="project_evidence",
            )
        )
        existing.add(target_id)
        views["evidence"].add(target_id)
    relation_map = {
        "PRECEDES": EdgeType.PRECEDENCE,
        "HAS_MODE": EdgeType.HAS_MODE,
        "REQUIRES": EdgeType.REQUIRES_RESOURCE,
        "APPLIES_TO": EdgeType.CONSTRAINT_SCOPE,
        "EVIDENCED_BY": EdgeType.EVIDENCED_BY,
    }
    for item in project_constraint_graph.edges:
        source = id_map.get(item.source, item.source)
        target = id_map.get(item.target, item.target)
        if source not in existing or target not in existing:
            continue
        relation = relation_map.get(item.relation, EdgeType.PROJECT_BINDING)
        edges.append(
            GraphEdge(
                id=_edge_id(relation, source, target, item.relation),
                source=source,
                target=target,
                type=relation,
                features={
                    "semantic_relation": item.relation,
                    **item.attributes,
                },
                provenance="project_evidence",
            )
        )


def build_utseg(
    problem: Any,
    schedule: Any,
    *,
    project_id: str,
    diagnostic_artifacts: Any | None = None,
    project_constraint_graph: Any | None = None,
) -> SchedulingGraph:
    """Build the unified typed symbolic evidence graph (UTSEG).

    The result is simultaneously a generalized disjunctive graph of the
    current schedule, a parameterized constraint graph and an event/state
    graph.  Semantic-prior nodes are optional and never upgraded to identified
    causal edges.
    """

    operation_map = problem.operation_map()
    assignment_map = schedule.assignment_map()
    resource_map = problem.resource_map()
    selected_resources = _selected_resources(problem, schedule)
    successors: dict[str, list[str]] = defaultdict(list)
    for operation in problem.operations:
        for predecessor in operation.predecessors:
            successors[predecessor].append(operation.id)

    by_job: dict[str, list[Any]] = defaultdict(list)
    by_resource: dict[str, list[Any]] = defaultdict(list)
    for operation in problem.operations:
        by_job[operation.job_id].append(operation)
    for assignment in schedule.assignments:
        for resource in selected_resources[assignment.operation_id]:
            by_resource[resource].append(assignment)
    for items in by_job.values():
        items.sort(key=lambda operation: (operation.index, operation.id))
    for items in by_resource.values():
        items.sort(key=lambda assignment: (assignment.start, assignment.end, assignment.operation_id))
    resource_utilization = {
        resource_id: sum(item.end - item.start for item in assignments)
        / max(1.0, schedule.makespan * resource_map[resource_id].capacity)
        for resource_id, assignments in by_resource.items()
    }

    predecessor_end: dict[str, int] = {}
    resource_previous: dict[tuple[str, str], Any] = {}
    resource_next: dict[tuple[str, str], Any] = {}
    resource_idle_before: dict[str, int] = defaultdict(int)
    resource_idle_after: dict[str, int] = defaultdict(int)
    for resource_id, assignments in by_resource.items():
        for index, assignment in enumerate(assignments):
            if index:
                previous = assignments[index - 1]
                resource_previous[(resource_id, assignment.operation_id)] = previous
                resource_idle_before[assignment.operation_id] += max(0, assignment.start - previous.end)
            if index + 1 < len(assignments):
                following = assignments[index + 1]
                resource_next[(resource_id, assignment.operation_id)] = following
                resource_idle_after[assignment.operation_id] += max(0, following.start - assignment.end)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    constraints: list[SymbolicConstraint] = []
    symbols: dict[str, ParameterSymbol] = {
        item.name: item for item in _CANONICAL_SYMBOLS
    }
    views: dict[str, set[str]] = defaultdict(set)
    makespan = float(schedule.makespan)
    source_id = f"source:{problem.id}"
    sink_id = f"sink:{problem.id}"
    nodes.extend(
        (
            GraphNode(
                id=source_id,
                type=NodeType.SOURCE,
                features={"time": 0},
                provenance="derived",
            ),
            GraphNode(
                id=sink_id,
                type=NodeType.SINK,
                features={"time": schedule.makespan},
                provenance="derived",
            ),
        )
    )
    views["topology"].update((source_id, sink_id))

    for factory in getattr(problem, "factories", ()):
        nodes.append(
            GraphNode(
                id=factory.id,
                type=NodeType.FACTORY,
                features={
                    "name": factory.name,
                    "calendar": factory.calendar,
                    "location": factory.location,
                    "metadata": factory.metadata,
                },
                provenance="problem_ir",
            )
        )
        views["topology"].add(factory.id)

    for operation in problem.operations:
        assignment = assignment_map[operation.id]
        predecessor_end[operation.id] = max(
            (assignment_map[item].end for item in operation.predecessors),
            default=operation.release,
        )
        wait = max(0, assignment.start - predecessor_end[operation.id])
        remaining = sum(
            assignment_map[item.id].end - assignment_map[item.id].start
            for item in by_job[operation.job_id]
            if item.index >= operation.index
        )
        available_modes = len(operation.modes)
        _, selected_mode = problem.mode_map()[assignment.mode_id]
        utilization = max(
            (
                resource_utilization.get(resource, 0.0)
                for resource in selected_resources[operation.id]
            ),
            default=0.0,
        )
        slack = max(0.0, makespan - assignment.end)
        nodes.append(
            GraphNode(
                id=operation.id,
                type=NodeType.OPERATION,
                features={
                    "job_id": operation.job_id,
                    "stage": operation.index,
                    "start": assignment.start,
                    "end": assignment.end,
                    "duration": assignment.end - assignment.start,
                    "arrival": predecessor_end[operation.id],
                    "wait": wait,
                    "remaining_operations": sum(
                        item.index >= operation.index for item in by_job[operation.job_id]
                    ),
                    "stage_id": operation.stage_id,
                    "remaining_processing": remaining,
                    "release": operation.release,
                    "slack": slack,
                    "slack_to_makespan": slack,
                    "criticality": 1.0 - slack / max(1.0, makespan),
                    "is_terminal": not successors[operation.id],
                    "available_modes": available_modes,
                    "resource_idle_before": resource_idle_before[operation.id],
                    "resource_idle_after": resource_idle_after[operation.id],
                    "modifiable": available_modes > 1 or bool(resource_idle_before[operation.id]),
                    "modifiability": min(
                        1.0,
                        0.25 * available_modes
                        + resource_idle_before[operation.id] / max(1.0, makespan),
                    ),
                    "resource_utilization": utilization,
                    "risk": float(bool(problem.choice_links)),
                    "cost": selected_mode.cost,
                },
                provenance="schedule",
            )
        )
        views["schedule_state"].add(operation.id)
        if not operation.predecessors:
            edges.append(
                GraphEdge(
                    id=_edge_id(EdgeType.SOURCE_LINK, source_id, operation.id),
                    source=source_id,
                    target=operation.id,
                    type=EdgeType.SOURCE_LINK,
                    provenance="derived",
                )
            )
        if not successors[operation.id]:
            edges.append(
                GraphEdge(
                    id=_edge_id(EdgeType.SINK_LINK, operation.id, sink_id),
                    source=operation.id,
                    target=sink_id,
                    type=EdgeType.SINK_LINK,
                    provenance="derived",
                )
            )
        for predecessor in operation.predecessors:
            delay = max(0, assignment.start - assignment_map[predecessor].end)
            edges.append(
                GraphEdge(
                    source=predecessor,
                    target=operation.id,
                    type=EdgeType.PRECEDENCE,
                    features={"delay": delay, "relation_class": "conjunctive"},
                    constraint_id=f"precedence:{predecessor}->{operation.id}",
                    provenance="problem_ir",
                )
            )
            constraints.append(
                SymbolicConstraint(
                    id=f"precedence:{predecessor}->{operation.id}",
                    kind="precedence",
                    scope=(predecessor, operation.id),
                    expression=_expr(
                        "greater_equal",
                        _symbol("S", operation.id),
                        _symbol("C", predecessor),
                    ),
                    rendered_expression=(
                        f"S_{{{operation.id}}} ≥ C_{{{predecessor}}}"
                    ),
                    encoded_by="core",
                    provenance="problem_ir",
                )
            )
            edges.append(
                GraphEdge(
                    source=predecessor,
                    target=operation.id,
                    type=EdgeType.TEMPORAL_CAUSAL,
                    features={"arrival_delay": delay},
                )
            )
            if delay > 0:
                edges.append(
                    GraphEdge(
                        source=predecessor,
                        target=operation.id,
                        type=EdgeType.WAIT_PROPAGATION,
                        features={"wait": delay},
                        provenance="schedule",
                    )
                )

        release = max(operation.release, problem.job_map()[operation.job_id].release)
        constraints.append(
            SymbolicConstraint(
                id=f"release:{operation.id}",
                kind="release",
                scope=(operation.id,),
                expression=_expr(
                    "greater_equal",
                    _symbol("S", operation.id),
                    _symbol("r", operation.id),
                ),
                rendered_expression=f"S_{{{operation.id}}} ≥ r_{{{operation.id}}}",
                parameter_bindings={"r": release},
                encoded_by="core",
                provenance="problem_ir",
            )
        )
        for mode in operation.modes:
            nodes.append(
                GraphNode(
                    id=mode.id,
                    type=NodeType.MODE,
                    features={
                        "duration": mode.duration,
                        "setup_family": mode.setup_family,
                        "route_id": mode.route_id,
                        "cost": mode.cost,
                        "energy": mode.energy,
                        "selected": mode.id == assignment.mode_id,
                        "metadata": mode.metadata,
                    },
                    active=mode.id == assignment.mode_id,
                    provenance="problem_ir",
                )
            )
            views["alternatives"].add(mode.id)
            edges.append(
                GraphEdge(
                    id=_edge_id(EdgeType.HAS_MODE, operation.id, mode.id),
                    source=operation.id,
                    target=mode.id,
                    type=EdgeType.HAS_MODE,
                    features={"selected": mode.id == assignment.mode_id},
                    active=mode.id == assignment.mode_id,
                    provenance="problem_ir",
                )
            )
            for resource_id in mode.resources:
                edges.append(
                    GraphEdge(
                        id=_edge_id(EdgeType.REQUIRES_RESOURCE, mode.id, resource_id),
                        source=mode.id,
                        target=resource_id,
                        type=EdgeType.REQUIRES_RESOURCE,
                        features={"selected": mode.id == assignment.mode_id},
                        active=mode.id == assignment.mode_id,
                        provenance="problem_ir",
                    )
                )
            if len(mode.resources) > 1:
                for index, left_resource in enumerate(mode.resources):
                    for right_resource in mode.resources[index + 1 :]:
                        edges.append(
                            GraphEdge(
                                id=_edge_id(
                                    EdgeType.MULTI_RESOURCE_SYNC,
                                    left_resource,
                                    right_resource,
                                    mode.id,
                                ),
                                source=left_resource,
                                target=right_resource,
                                type=EdgeType.MULTI_RESOURCE_SYNC,
                                features={
                                    "mode": mode.id,
                                    "operation": operation.id,
                                    "selected": mode.id == assignment.mode_id,
                                },
                                directed=False,
                                active=mode.id == assignment.mode_id,
                                provenance="problem_ir",
                            )
                        )
            if mode.route_id:
                route_id = f"route:{mode.route_id}"
                if not any(node.id == route_id for node in nodes):
                    nodes.append(
                        GraphNode(
                            id=route_id,
                            type=NodeType.ROUTE,
                            features={"route_id": mode.route_id},
                            provenance="problem_ir",
                        )
                    )
                    views["topology"].add(route_id)
                edges.append(
                    GraphEdge(
                        id=_edge_id(EdgeType.PROJECT_BINDING, mode.id, route_id, "route"),
                        source=mode.id,
                        target=route_id,
                        type=EdgeType.PROJECT_BINDING,
                        features={"relation": "uses_route"},
                        active=mode.id == assignment.mode_id,
                        provenance="problem_ir",
                    )
                )
        constraints.append(
            SymbolicConstraint(
                id=f"mode_selection:{operation.id}",
                kind="mode_selection",
                scope=(operation.id, *(mode.id for mode in operation.modes)),
                expression=_expr("equal", _expr("sum", _opaque(",".join(mode.id for mode in operation.modes))), _constant(1)),
                rendered_expression=f"Σ_{{k∈K_{{{operation.id}}}}} x_{{{operation.id},k}} = 1",
                parameter_bindings={"eligible_modes": [mode.id for mode in operation.modes]},
                encoded_by="core",
                provenance="problem_ir",
                representation_status="canonical",
            )
        )

    for job_id, operations in sorted(by_job.items()):
        completion = max(assignment_map[item.id].end for item in operations)
        nodes.append(
            GraphNode(
                id=job_id,
                type=NodeType.JOB,
                features={
                    "operation_count": len(operations),
                    "completion": completion,
                    "flow_time": completion - min(item.release for item in operations),
                },
                provenance="schedule",
            )
        )
        views["schedule_state"].add(job_id)
        for operation in operations:
            edges.append(
                GraphEdge(
                    source=job_id,
                    target=operation.id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "contains"},
                )
            )
            edges.append(
                GraphEdge(
                    source=operation.id,
                    target=job_id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "member_of"},
                )
            )

    by_stage: dict[str, list[Any]] = defaultdict(list)
    for operation in problem.operations:
        stage_key = operation.stage_id or f"operation-index:{operation.index}"
        by_stage[stage_key].append(operation)
    for stage, operations in sorted(by_stage.items()):
        stage_id = f"stage:{stage}"
        nodes.append(
            GraphNode(
                id=stage_id,
                type=NodeType.STAGE,
                features={
                    "stage": stage,
                    "explicit_stage": all(item.stage_id is not None for item in operations),
                    "operation_count": len(operations),
                    "mean_wait": sum(
                        float(
                            next(
                                node
                                for node in nodes
                                if node.id == operation.id
                            ).features.get("wait", 0.0)
                        )
                        for operation in operations
                    )
                    / max(1, len(operations)),
                },
                provenance="derived",
            )
        )
        views["topology"].add(stage_id)
        for operation in operations:
            edges.append(
                GraphEdge(
                    source=stage_id,
                    target=operation.id,
                    type=EdgeType.PROJECT_BINDING,
                    features={"relation": "stage_contains"},
                )
            )

    for resource_id, resource in sorted(resource_map.items()):
        assignments = by_resource.get(resource_id, [])
        busy = sum(item.end - item.start for item in assignments)
        available = max(1.0, makespan * resource.capacity)
        resource_type = {
            "vehicle": NodeType.VEHICLE,
            "agv": NodeType.VEHICLE,
            "buffer": NodeType.BUFFER,
        }.get(str(resource.family).casefold(), NodeType.RESOURCE)
        nodes.append(
            GraphNode(
                id=resource_id,
                type=resource_type,
                features={
                    "capacity": resource.capacity,
                    "load": busy,
                    "queue_length": len(assignments),
                    "utilization": busy / available,
                    "idle": max(0.0, available - busy),
                    "tags": ",".join(resource.tags),
                    "family": resource.family,
                    "calendar": resource.calendar,
                    "factory_id": resource.factory_id,
                },
                provenance="schedule",
            )
        )
        views["schedule_state"].add(resource_id)
        if resource.factory_id:
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.FACTORY_MEMBERSHIP,
                        resource.factory_id,
                        resource_id,
                    ),
                    source=resource.factory_id,
                    target=resource_id,
                    type=EdgeType.FACTORY_MEMBERSHIP,
                    provenance="problem_ir",
                )
            )
        eligible_operations = [
            operation
            for operation in problem.operations
            if any(
                resource_id in mode.resources for mode in operation.modes
            )
        ]
        selected_ids = {item.operation_id for item in assignments}
        for operation in eligible_operations:
            edges.append(
                GraphEdge(
                    source=operation.id,
                    target=resource_id,
                    type=EdgeType.ELIGIBILITY,
                    features={"selected": operation.id in selected_ids},
                    provenance="problem_ir",
                )
            )
            edges.append(
                GraphEdge(
                    source=resource_id,
                    target=operation.id,
                    type=EdgeType.ELIGIBILITY,
                    features={
                        "selected": operation.id in selected_ids,
                        "direction": "resource_to_operation",
                    },
                    provenance="problem_ir",
                )
            )
        for index, left in enumerate(eligible_operations):
            for right in eligible_operations[index + 1 :]:
                edges.append(
                    GraphEdge(
                        source=left.id,
                        target=right.id,
                        type=EdgeType.RESOURCE_COMPETITION,
                        features={
                            "resource": resource_id,
                            "relation_class": "disjunctive",
                            "selected_left": left.id in selected_ids,
                            "selected_right": right.id in selected_ids,
                        },
                        directed=False,
                        active=left.id in selected_ids and right.id in selected_ids,
                        constraint_id=f"competition:{resource_id}:{left.id}:{right.id}",
                        provenance="problem_ir",
                    )
                )
                if resource.capacity == 1:
                    constraints.append(
                        SymbolicConstraint(
                            id=f"competition:{resource_id}:{left.id}:{right.id}",
                            kind="disjunctive_resource_conflict",
                            scope=(left.id, right.id, resource_id),
                            expression=_expr(
                                "or",
                                _expr(
                                    "greater_equal",
                                    _symbol("S", right.id),
                                    _symbol("C", left.id),
                                ),
                                _expr(
                                    "greater_equal",
                                    _symbol("S", left.id),
                                    _symbol("C", right.id),
                                ),
                            ),
                            rendered_expression=(
                                f"(S_{{{right.id}}} ≥ C_{{{left.id}}}) ∨ "
                                f"(S_{{{left.id}}} ≥ C_{{{right.id}}})"
                            ),
                            parameter_bindings={"resource": resource_id},
                            encoded_by="core",
                            provenance="problem_ir",
                        )
                    )
        if resource.capacity == 1:
            for left, right in zip(assignments, assignments[1:]):
                edges.append(
                    GraphEdge(
                        source=left.operation_id,
                        target=right.operation_id,
                        type=EdgeType.RESOURCE_SEQUENCE,
                        features={
                            "gap": max(0, right.start - left.end),
                            "resource": resource_id,
                            "relation_class": "oriented_disjunctive",
                        },
                        constraint_id=f"competition:{resource_id}:{min(left.operation_id, right.operation_id)}:{max(left.operation_id, right.operation_id)}",
                        provenance="schedule",
                    )
                )
                if left.end == right.start and right.end == schedule.makespan:
                    edges.append(
                        GraphEdge(
                            source=left.operation_id,
                            target=right.operation_id,
                            type=EdgeType.CRITICAL_PATH,
                            features={"resource": resource_id},
                            provenance="derived",
                        )
                    )
        else:
            constraints.append(
                SymbolicConstraint(
                    id=f"capacity:{resource_id}",
                    kind="cumulative_capacity",
                    scope=(resource_id, *(item.operation_id for item in assignments)),
                    expression=_opaque(
                        f"Σ_o 1[S_o ≤ t < C_o ∧ o uses {resource_id}] ≤ cap_{{{resource_id}}}"
                    ),
                    rendered_expression=(
                        f"Σ_o 1[S_o ≤ t < C_o ∧ o uses {resource_id}] "
                        f"≤ cap_{{{resource_id}}}, ∀t"
                    ),
                    parameter_bindings={"cap": resource.capacity},
                    encoded_by="core",
                    provenance="problem_ir",
                    representation_status="parameterized_opaque",
                )
            )

    for link in problem.choice_links:
        for left, right in zip(link.operation_ids, link.operation_ids[1:]):
            edges.append(
                GraphEdge(
                    source=left,
                    target=right,
                    type=EdgeType.PROJECT_BINDING,
                    features={"binding": link.id},
                    constraint_id=f"choice_binding:{link.id}",
                    provenance="problem_ir",
                )
            )

    _add_explicit_constraints(
        problem=problem,
        nodes=nodes,
        edges=edges,
        constraints=constraints,
        symbols=symbols,
        views=views,
    )
    _add_dynamic_events(
        problem=problem,
        schedule=schedule,
        nodes=nodes,
        edges=edges,
        views=views,
    )
    _add_objectives_and_decisions(
        problem=problem,
        nodes=nodes,
        edges=edges,
        views=views,
    )
    if diagnostic_artifacts is not None:
        _add_diagnostic_evidence(
            diagnostic_artifacts=diagnostic_artifacts,
            nodes=nodes,
            edges=edges,
            constraints=constraints,
            symbols=symbols,
            views=views,
        )
    if project_constraint_graph is not None:
        _add_project_constraint_evidence(
            project_constraint_graph=project_constraint_graph,
            nodes=nodes,
            edges=edges,
            views=views,
        )

    existing_node_ids = {node.id for node in nodes}
    for constraint in constraints:
        node_id = f"constraint:{constraint.id}"
        if node_id not in existing_node_ids:
            nodes.append(
                GraphNode(
                    id=node_id,
                    type=NodeType.CONSTRAINT,
                    features={
                        "constraint_id": constraint.id,
                        "kind": constraint.kind,
                        "hard": constraint.hard,
                        "encoded_by": constraint.encoded_by,
                        "expression": constraint.expression.model_dump(mode="json"),
                        "rendered_expression": constraint.rendered_expression,
                        "parameter_bindings": constraint.parameter_bindings,
                        "activation_condition": constraint.activation_condition,
                        "representation_status": constraint.representation_status,
                    },
                    active=constraint.active,
                    provenance=constraint.provenance,
                )
            )
            existing_node_ids.add(node_id)
        views["constraints"].add(node_id)
        for target in constraint.scope:
            if target not in existing_node_ids:
                continue
            edges.append(
                GraphEdge(
                    id=_edge_id(
                        EdgeType.CONSTRAINT_SCOPE,
                        node_id,
                        target,
                        constraint.id,
                    ),
                    source=node_id,
                    target=target,
                    type=EdgeType.CONSTRAINT_SCOPE,
                    active=constraint.active,
                    constraint_id=constraint.id,
                    provenance=constraint.provenance,
                )
            )

    # Stable de-duplication is important because a semantic constraint and a
    # core IR relation may cite the same entities without being the same fact.
    node_map: dict[str, GraphNode] = {}
    for node in nodes:
        node_map.setdefault(node.id, node)
    edge_map: dict[str, GraphEdge] = {}
    for edge in edges:
        stable_id = edge.id or _edge_id(
            edge.type,
            edge.source,
            edge.target,
            edge.constraint_id or "",
        )
        edge_map[stable_id] = (
            edge if edge.id is not None else edge.model_copy(update={"id": stable_id})
        )
    constraint_map = {item.id: item for item in constraints}

    return SchedulingGraph(
        project_id=project_id,
        problem_id=problem.id,
        problem_family=problem.kind,
        nodes=tuple(node_map.values()),
        edges=tuple(edge_map.values()),
        constraints=tuple(constraint_map.values()),
        symbols=tuple(symbols.values()),
        views={key: tuple(sorted(value)) for key, value in sorted(views.items())},
        objective=makespan / getattr(problem, "time_scale", 1),
        metadata={
            "graphKind": "unified_typed_symbolic_evidence_graph",
            "graphAcronym": "UTSEG",
            "timeScale": getattr(problem, "time_scale", 1),
            "operationCount": len(problem.operations),
            "resourceCount": len(problem.resources),
            "constraintCount": len(constraint_map),
            "eventCount": len(getattr(problem, "events", ())),
            "environment": getattr(problem, "environment", "static"),
            "stateTime": int(schedule.metadata.get("state_time", schedule.makespan)),
            "evidenceSemantics": "observational_and_semantic_prior_not_identified_causality",
            "sharedCausalSkeleton": ["A", "L", "W", "T", "D", "J"],
        },
    )


def build_scheduling_graph(
    problem: Any,
    schedule: Any,
    *,
    project_id: str,
    diagnostic_artifacts: Any | None = None,
    project_constraint_graph: Any | None = None,
) -> SchedulingGraph:
    """Backward-compatible name for :func:`build_utseg`."""

    return build_utseg(
        problem,
        schedule,
        project_id=project_id,
        diagnostic_artifacts=diagnostic_artifacts,
        project_constraint_graph=project_constraint_graph,
    )
