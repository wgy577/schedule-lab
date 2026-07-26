# TEST-TAGS: modules=B,D; capabilities=semantic_graph,taxonomy,variant_head_router; level=integration; cost=low
from pathlib import Path

from causal_schedule_lab.benchmarks import example_problems
from causal_schedule_lab.llm_semantics import (
    Confidence,
    ConstraintFinding,
    ConstraintKind,
    ConstraintScope,
    DecisionFinding,
    DecisionKind,
    EnvironmentFinding,
    EnvironmentKind,
    EvidenceCitation,
    ObjectiveFinding,
    ObjectiveKind,
    ObjectiveSense,
    OracleFinding,
    OracleKind,
    ProblemFamily,
    ProjectType,
    SemanticAnalysis,
)
from causal_schedule_lab.semantic_graph import build_project_constraint_graph
from causal_schedule_lab.semantic_agent import retrieve_long_term_context
from causal_schedule_lab.semantic_knowledge import SchedulingKnowledgeBase
from causal_schedule_lab.storage import HierarchicalMemory
from causal_schedule_lab.taxonomy import SchedulingTaxonomy
from causal_schedule_lab.taxonomy import TaxonomyNode
from pydantic import ValidationError
import pytest


def _analysis() -> SemanticAnalysis:
    evidence = (
        EvidenceCitation(
            file="env.py",
            symbol="step",
            detail="The environment checks machine availability.",
        ),
    )
    return SemanticAnalysis(
        schema_version="1.0",
        language="zh-CN",
        summary="FJSP environment",
        project_type=ProjectType.SCHEDULING_OPTIMIZATION,
        problem_families=(ProblemFamily.FJSP,),
        environments=(
            EnvironmentFinding(
                kind=EnvironmentKind.DETERMINISTIC,
                statement="Deterministic environment",
                evidence=evidence,
                confidence=Confidence.HIGH,
            ),
        ),
        objectives=(
            ObjectiveFinding(
                id="objective_1",
                kind=ObjectiveKind.MAKESPAN,
                sense=ObjectiveSense.MINIMIZE,
                priority=1,
                statement="Minimize makespan",
                evidence=evidence,
                confidence=Confidence.HIGH,
            ),
        ),
        constraints=(
            ConstraintFinding(
                id="constraint_1",
                kind=ConstraintKind.RESOURCE_ELIGIBILITY,
                scope=ConstraintScope.OPERATION,
                hard=True,
                statement="Each operation has eligible machines",
                evidence=evidence,
                confidence=Confidence.HIGH,
            ),
        ),
        decisions=(
            DecisionFinding(
                id="decision_1",
                kind=DecisionKind.ASSIGN_RESOURCE,
                modifiable=True,
                statement="Choose an eligible machine",
                evidence=evidence,
                confidence=Confidence.HIGH,
            ),
        ),
        oracles=(
            OracleFinding(
                id="oracle_1",
                kind=OracleKind.STATIC_VALIDATOR,
                required=True,
                statement="Validate feasibility",
                evidence=evidence,
                confidence=Confidence.MEDIUM,
            ),
        ),
        allowed_interventions=(DecisionKind.ASSIGN_RESOURCE,),
        overall_confidence=Confidence.HIGH,
    )


def test_project_constraint_graph_combines_semantics_evidence_and_ir() -> None:
    graph = build_project_constraint_graph(
        _analysis(),
        project_id="demo",
        problem=example_problems()["fjsp"],
    )
    node_types = {item.node_type for item in graph.nodes}
    relations = {item.relation for item in graph.edges}

    assert {
        "Project",
        "ProblemFamily",
        "Environment",
        "Objective",
        "Constraint",
        "Decision",
        "Oracle",
        "CodeEvidence",
        "Job",
        "Operation",
        "ProcessingMode",
        "Resource",
    } <= node_types
    assert {
        "CLASSIFIED_AS",
        "EVIDENCED_BY",
        "PRECEDES",
        "HAS_MODE",
        "REQUIRES",
    } <= relations
    assert graph.metadata["inference_policy"] == "explicit_findings_and_ir_only"
    assert "scheduling_core" in graph.views
    assert "evidence_overlay" in graph.views
    assert set(graph.views["scheduling_core"]).isdisjoint(
        graph.views["evidence_overlay"]
    )


def test_taxonomy_resolves_root_family_and_variant_without_repetition() -> None:
    taxonomy = SchedulingTaxonomy.from_knowledge_base(
        SchedulingKnowledgeBase.load()
    )
    profile = taxonomy.resolve("partial_total_fjsp")

    assert profile.lineage == (
        "taxonomy:shop_scheduling",
        "family:FJSP",
        "variant:partial_total_fjsp",
    )
    assert "entities" in profile.resolved
    assert "core_constraints" in profile.resolved
    assert "variant_features" in profile.resolved
    variant_delta = profile.contributions["variant:partial_total_fjsp"]
    assert "core_constraints" not in variant_delta
    assert taxonomy.children("taxonomy:shop_scheduling")


def test_sql_memory_persists_and_resolves_taxonomy_tree(tmp_path: Path) -> None:
    memory = HierarchicalMemory.open(tmp_path / "memory.sqlite3")
    memory.ingest_curated_seed()

    profile = memory.resolve_taxonomy("partial flexibility", scope="FJSP")
    root = memory.store.get_nodes(("taxonomy:shop_scheduling",))[0]
    variant = memory.store.get_nodes(("variant:partial_total_fjsp",))[0]

    assert profile.lineage == (
        "taxonomy:shop_scheduling",
        "family:FJSP",
        "variant:partial_total_fjsp",
    )
    assert root.metadata["taxonomy_level"] == 1
    assert variant.metadata["taxonomy_level"] == 3
    assert "core_constraints" not in variant.metadata["local_delta"]


def test_taxonomy_rejects_summary_placeholder_branches() -> None:
    with pytest.raises(ValidationError):
        TaxonomyNode(
            id="variant:other_variants",
            parent_id="family:FJSP",
            level=3,
            node_type="Variant",
            name="其他变体",
        )


def test_agent_memory_context_contains_resolved_taxonomy(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    memory = HierarchicalMemory.open(database)
    memory.ingest_curated_seed()

    context = retrieve_long_term_context(
        database,
        "This repository implements a static FJSP environment.",
    )

    assert context is not None
    assert len(context.taxonomy_profiles) == 1
    assert context.taxonomy_profiles[0].lineage == (
        "taxonomy:shop_scheduling",
        "family:FJSP",
        "variant:fjsp_classic",
    )
    assert context.variant_head_selections[0].fallback_to_classic is True
    assert context.variant_head_selections[0].selected_node_id == (
        "variant:fjsp_classic"
    )


def test_agent_memory_routes_to_one_salient_variant(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    memory = HierarchicalMemory.open(database)
    memory.ingest_curated_seed()

    context = retrieve_long_term_context(
        database,
        "A FJSP with AGV transport time and collision avoidance.",
    )

    assert context is not None
    assert len(context.taxonomy_profiles) == 1
    assert context.taxonomy_profiles[0].node_id == "variant:fjsp_transport"
    assert context.variant_head_selections[0].selected_node_id == (
        "variant:fjsp_transport"
    )
    assert context.variant_head_selections[0].fallback_to_classic is False
    assert "agv" in {
        item.casefold()
        for item in context.variant_head_selections[0].matched_heads
    }
