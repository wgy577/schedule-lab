"""Build an evidence-linked scheduling graph from semantic findings and the IR."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from .ir import Problem
from .semantic_graph_models import (
    ProjectConstraintGraph,
    SemanticGraphEdge,
    SemanticGraphNode,
)


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._:-" else "_" for char in value)


class _GraphBuilder:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.nodes: dict[str, SemanticGraphNode] = {}
        self.edges: dict[tuple[str, str, str], SemanticGraphEdge] = {}

    def node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        **attributes: Any,
    ) -> str:
        self.nodes[node_id] = SemanticGraphNode(
            id=node_id,
            node_type=node_type,
            label=label,
            attributes=attributes,
        )
        return node_id

    def edge(
        self,
        source: str,
        relation: str,
        target: str,
        **attributes: Any,
    ) -> None:
        key = (source, relation, target)
        self.edges[key] = SemanticGraphEdge(
            source=source,
            relation=relation,
            target=target,
            attributes=attributes,
        )

    def evidence(self, finding_id: str, citations: Iterable[Any]) -> None:
        for citation in citations:
            raw = f"{citation.file}|{citation.symbol or ''}|{citation.detail}"
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
            evidence_id = f"evidence:{digest}"
            self.node(
                evidence_id,
                "CodeEvidence",
                citation.symbol or citation.file,
                file=citation.file,
                symbol=citation.symbol,
                detail=citation.detail,
                graph_layer="evidence_overlay",
            )
            self.edge(
                finding_id,
                "EVIDENCED_BY",
                evidence_id,
                relation_basis="explicit_citation",
            )


def _add_semantic_analysis(builder: _GraphBuilder, analysis: Any) -> None:
    project = f"project:{_safe(builder.project_id)}"
    builder.node(
        project,
        "Project",
        builder.project_id,
        project_type=str(analysis.project_type),
        summary=analysis.summary,
        overall_confidence=str(analysis.overall_confidence),
        graph_layer="scheduling_core",
    )
    for family in analysis.problem_families:
        family_value = family.value if hasattr(family, "value") else str(family)
        family_id = f"family:{family_value}"
        builder.node(
            family_id,
            "ProblemFamily",
            family_value,
            graph_layer="scheduling_core",
        )
        builder.edge(project, "CLASSIFIED_AS", family_id, relation_basis="llm_finding")

    groups = (
        ("environments", "Environment", "HAS_ENVIRONMENT", "context_overlay"),
        ("objectives", "Objective", "OPTIMIZES", "scheduling_core"),
        ("constraints", "Constraint", "SUBJECT_TO", "scheduling_core"),
        ("decisions", "Decision", "ALLOWS_DECISION", "scheduling_core"),
        ("oracles", "Oracle", "VALIDATED_BY", "validation_overlay"),
    )
    for field, node_type, relation, graph_layer in groups:
        for index, finding in enumerate(getattr(analysis, field)):
            raw_id = getattr(finding, "id", f"{field}_{index + 1}")
            finding_id = f"semantic:{raw_id}"
            attributes = finding.model_dump(mode="json", exclude={"evidence"})
            label = attributes.get("statement") or raw_id
            builder.node(
                finding_id,
                node_type,
                label,
                graph_layer=graph_layer,
                **attributes,
            )
            builder.edge(project, relation, finding_id, relation_basis="llm_finding")
            builder.evidence(finding_id, getattr(finding, "evidence", ()))

    for index, unknown in enumerate(analysis.unknowns):
        unknown_id = f"semantic:unknown_{index + 1}"
        builder.node(
            unknown_id,
            "Unknown",
            unknown.question,
            **unknown.model_dump(mode="json"),
            graph_layer="audit_overlay",
        )
        builder.edge(project, "HAS_UNKNOWN", unknown_id, relation_basis="llm_finding")


def _add_problem_ir(builder: _GraphBuilder, problem: Problem) -> None:
    project = f"project:{_safe(builder.project_id)}"
    problem_id = f"ir:problem:{_safe(problem.id)}"
    builder.node(
        problem_id,
        "SchedulingInstance",
        problem.id,
        kind=problem.kind,
        graph_layer="scheduling_core",
    )
    builder.edge(project, "HAS_INSTANCE", problem_id, relation_basis="ir")

    for job in problem.jobs:
        node_id = f"ir:job:{_safe(job.id)}"
        builder.node(
            node_id,
            "Job",
            job.id,
            release=job.release,
            due=job.due,
            weight=job.weight,
            metadata=job.metadata,
            graph_layer="scheduling_core",
        )
        builder.edge(problem_id, "HAS_JOB", node_id, relation_basis="ir")

    for resource in problem.resources:
        node_id = f"ir:resource:{_safe(resource.id)}"
        builder.node(
            node_id,
            "Resource",
            resource.name,
            capacity=resource.capacity,
            family=resource.family,
            calendar=resource.calendar,
            location=resource.location,
            tags=resource.tags,
            graph_layer="scheduling_core",
        )
        builder.edge(problem_id, "HAS_RESOURCE", node_id, relation_basis="ir")

    for operation in problem.operations:
        operation_id = f"ir:operation:{_safe(operation.id)}"
        builder.node(
            operation_id,
            "Operation",
            operation.id,
            index=operation.index,
            release=operation.release,
            due=operation.due,
            metadata=operation.metadata,
            graph_layer="scheduling_core",
        )
        builder.edge(
            f"ir:job:{_safe(operation.job_id)}",
            "CONTAINS",
            operation_id,
            relation_basis="ir",
        )
        for predecessor in operation.predecessors:
            builder.edge(
                f"ir:operation:{_safe(predecessor)}",
                "PRECEDES",
                operation_id,
                relation_basis="ir",
            )
        for mode in operation.modes:
            mode_id = f"ir:mode:{_safe(mode.id)}"
            builder.node(
                mode_id,
                "ProcessingMode",
                mode.id,
                duration=mode.duration,
                setup_family=mode.setup_family,
                route_id=mode.route_id,
                cost=mode.cost,
                energy=mode.energy,
                metadata=mode.metadata,
                graph_layer="scheduling_core",
            )
            builder.edge(operation_id, "HAS_MODE", mode_id, relation_basis="ir")
            for resource in mode.resources:
                builder.edge(
                    mode_id,
                    "REQUIRES",
                    f"ir:resource:{_safe(resource)}",
                    relation_basis="ir",
                )

    for constraint in problem.constraints:
        constraint_id = f"ir:constraint:{_safe(constraint.id)}"
        builder.node(
            constraint_id,
            "Constraint",
            constraint.id,
            kind=constraint.kind,
            parameters=constraint.parameters,
            encoded_by=constraint.encoded_by,
            graph_layer="scheduling_core",
        )
        builder.edge(problem_id, "SUBJECT_TO", constraint_id, relation_basis="ir")
        for scoped_id in constraint.scope:
            candidates = (
                f"ir:operation:{_safe(scoped_id)}",
                f"ir:resource:{_safe(scoped_id)}",
                f"ir:job:{_safe(scoped_id)}",
                f"ir:mode:{_safe(scoped_id)}",
            )
            target = next((item for item in candidates if item in builder.nodes), None)
            if target:
                builder.edge(
                    constraint_id,
                    "APPLIES_TO",
                    target,
                    relation_basis="explicit_ir_scope",
                )

    for objective in problem.objective:
        objective_id = f"ir:objective:{_safe(objective.name)}"
        builder.node(
            objective_id,
            "Objective",
            objective.name,
            sense=objective.sense,
            tolerance=objective.tolerance,
            weight=objective.weight,
            graph_layer="scheduling_core",
        )
        builder.edge(problem_id, "OPTIMIZES", objective_id, relation_basis="ir")

    for link in problem.choice_links:
        link_id = f"ir:choice:{_safe(link.id)}"
        builder.node(
            link_id,
            "ChoiceBinding",
            link.id,
            mode_keys=link.mode_keys,
            graph_layer="scheduling_core",
        )
        builder.edge(problem_id, "HAS_CHOICE_BINDING", link_id, relation_basis="ir")
        for operation_id in link.operation_ids:
            builder.edge(
                link_id,
                "BINDS",
                f"ir:operation:{_safe(operation_id)}",
                relation_basis="ir",
            )


def build_project_constraint_graph(
    analysis: Any,
    *,
    project_id: str,
    problem: Problem | None = None,
    impact_report: Any | None = None,
) -> ProjectConstraintGraph:
    """Create a deterministic graph without inventing unsupported causal edges."""

    builder = _GraphBuilder(project_id)
    _add_semantic_analysis(builder, analysis)
    if problem is not None:
        _add_problem_ir(builder, problem)
    if impact_report is not None:
        primary_objectives = [
            node.id for node in builder.nodes.values() if node.node_type == "Objective"
        ]
        for target in impact_report.secondary_targets:
            target_id = f"secondary:{target.id}"
            builder.node(
                target_id,
                "SecondaryTarget",
                target.statement,
                graph_layer="scheduling_core",
                **target.model_dump(mode="json"),
            )
            for constraint_id in target.source_constraint_ids:
                semantic_constraint = f"semantic:{constraint_id}"
                if semantic_constraint in builder.nodes:
                    builder.edge(
                        semantic_constraint,
                        "AFFECTS_SECONDARY_TARGET",
                        target_id,
                        relation_basis="structured_llm_prior",
                    )
            for objective_id in primary_objectives:
                builder.edge(
                    target_id,
                    "MEDIATES_OBJECTIVE",
                    objective_id,
                    relation_basis="structured_llm_prior",
                )
        for target in impact_report.diagnostic_targets:
            target_id = f"diagnostic:{target.id}"
            builder.node(
                target_id,
                "DiagnosticTarget",
                target.statement,
                graph_layer="validation_overlay",
                **target.model_dump(mode="json"),
            )
            for constraint_id in target.source_constraint_ids:
                semantic_constraint = f"semantic:{constraint_id}"
                if semantic_constraint in builder.nodes:
                    builder.edge(
                        semantic_constraint,
                        "CHECKED_BY_DIAGNOSTIC",
                        target_id,
                        relation_basis="structured_llm_prior",
                    )
    views: dict[str, list[str]] = {}
    for node in builder.nodes.values():
        layer = str(node.attributes.get("graph_layer", "audit_overlay"))
        views.setdefault(layer, []).append(node.id)
    return ProjectConstraintGraph(
        project_id=project_id,
        nodes=tuple(sorted(builder.nodes.values(), key=lambda item: item.id)),
        edges=tuple(
            sorted(
                builder.edges.values(),
                key=lambda item: (item.source, item.relation, item.target),
            )
        ),
        views={
            key: tuple(sorted(value))
            for key, value in sorted(views.items())
        },
        metadata={
            "graph_kind": "evidence_linked_scheduling_constraint_graph",
            "contains_ir": problem is not None,
            "contains_secondary_targets": bool(
                impact_report is not None and impact_report.secondary_targets
            ),
            "contains_diagnostic_targets": bool(
                impact_report is not None and impact_report.diagnostic_targets
            ),
            "inference_policy": "explicit_findings_and_ir_only",
        },
    )
