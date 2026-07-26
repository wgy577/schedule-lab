# TEST-TAGS: modules=B,D,H; capabilities=long_term_memory,mechanism_metrics,posterior; level=integration; cost=low
from pathlib import Path

from causal_schedule_lab.benchmarks import example_problems
from causal_schedule_lab.mechanisms import (
    FactorControllability,
    MechanismLink,
    estimate_effect_posterior,
    measure_mechanisms,
    qualify_factor,
)
from causal_schedule_lab.solvers.dispatching import solve_dispatching
from causal_schedule_lab.storage import HierarchicalMemory, InterventionRecord


def test_curated_seed_builds_graph_and_hierarchical_retrieval(tmp_path: Path) -> None:
    memory = HierarchicalMemory.open(tmp_path / "memory.sqlite3")
    report = memory.ingest_curated_seed()
    stats = memory.store.stats()

    assert report.changed_nodes > 50
    assert stats.nodes > 50
    assert stats.edges > 50
    assert stats.evidence_chunks > 50
    assert memory.store.resolve_alias("FJSP") == ("family:FJSP",)

    result = memory.retrieve(
        "AGV transport collision",
        family_hint="FJSP",
    )
    assert "family:FJSP" in result.expanded_ids
    assert any(
        "mobile_transport_coupling" in item.chunk.node_ids
        or "pattern:mobile_transport_coupling" in item.chunk.node_ids
        for item in result.evidence
    )


def test_seed_ingestion_is_idempotent(tmp_path: Path) -> None:
    memory = HierarchicalMemory.open(tmp_path / "memory.sqlite3")
    memory.ingest_curated_seed()
    before = memory.store.stats()
    second = memory.ingest_curated_seed()
    after = memory.store.stats()

    assert second.changed_nodes == 0
    assert second.changed_edges == 0
    assert second.changed_evidence == 0
    assert before == after


def test_external_document_stays_proposed_and_is_retrievable(tmp_path: Path) -> None:
    document = tmp_path / "review.md"
    document.write_text(
        "Flexible job shop scheduling integrates AGV routing.\n\n"
        "This paragraph discusses transport synchronization delays.",
        encoding="utf-8",
    )
    memory = HierarchicalMemory.open(tmp_path / "memory.sqlite3")
    report = memory.ingest_external_document(
        document,
        title="Unreviewed survey",
        scope="FJSP",
    )
    result = memory.retrieve("transport synchronization", family_hint="FJSP")

    assert report.changed_evidence == 1
    assert result.evidence[0].chunk.review_status == "proposed"
    assert "Unreviewed survey" in result.evidence[0].chunk.title


def test_mechanism_measurements_are_deterministic_and_do_not_invent_metadata() -> None:
    problem = example_problems()["jsp"]
    schedule = solve_dispatching(problem)
    first = measure_mechanisms(problem, schedule)
    second = measure_mechanisms(problem, schedule)
    by_id = {item.mechanism_id: item for item in first}

    assert first == second
    assert by_id["critical_resource_internal_idle_time"].available
    assert by_id["realized_schedule_critical_path_length"].available
    assert by_id["operation_ready_to_start_waiting_time"].available
    assert not by_id["total_sequence_dependent_setup_time"].available
    assert not by_id["transport_induced_waiting_time"].available


def test_candidate_variation_and_control_are_hard_eligibility_gates() -> None:
    fixed = qualify_factor(
        [0, 0, 0],
        controllability=FactorControllability.DIRECT,
        mechanism_link=MechanismLink.DIRECT,
    )
    hypothetical = qualify_factor(
        [4, 3, 2],
        controllability=FactorControllability.HYPOTHESIZED_INDIRECT,
        mechanism_link=MechanismLink.MEDIATED,
    )
    verified = qualify_factor(
        [4, 3, 2],
        controllability=FactorControllability.VERIFIED_INDIRECT,
        mechanism_link=MechanismLink.MEDIATED,
        replay_count=3,
        replay_consistency=1.0,
    )

    assert not fixed.eligible
    assert not hypothetical.eligible
    assert verified.eligible


def test_intervention_memory_and_conservative_posterior(tmp_path: Path) -> None:
    memory = HierarchicalMemory.open(tmp_path / "memory.sqlite3")
    rows = [
        InterventionRecord(
            experiment_id=f"e{index}",
            problem_family="JSP",
            instance_hash="instance",
            incumbent_hash="incumbent",
            state_hash="state",
            operator_id="swap",
            factor_id="machine_sequence",
            mechanism_id="critical_resource_internal_idle_time",
            mechanism_delta=mechanism_delta,
            objective_gain=gain,
            valid=valid,
            validation_fidelity="full",
            runtime_seconds=2.0,
            token_cost=0,
        )
        for index, (mechanism_delta, gain, valid) in enumerate(
            [(-2.0, 4.0, True), (-1.0, 2.0, True), (0.0, 0.0, False)]
        )
    ]
    for row in rows:
        assert memory.store.record_intervention(row)
        assert not memory.store.record_intervention(row)

    stored = memory.store.intervention_records(
        problem_family="JSP",
        mechanism_id="critical_resource_internal_idle_time",
        full_only=True,
    )
    estimate = estimate_effect_posterior(stored, scope_level="instance")

    assert len(stored) == 3
    assert estimate.observations == 3
    assert estimate.full_observations == 3
    assert 0 < estimate.valid_probability < 1
    assert estimate.objective_gain_mean == 3.0
    assert estimate.conservative_gain > 0
    assert estimate.acquisition > 0
