# TEST-TAGS: modules=B,D; capabilities=knowledge_retrieval,constraint_impact,secondary_metric_catalog; level=integration; cost=low
import json
from pathlib import Path

import pytest

from causal_schedule_lab.constraint_impact import (
    assess_constraint_impacts_with_llm,
)
from causal_schedule_lab.llm_semantics import SemanticAnalysis
from causal_schedule_lab.providers.base import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from causal_schedule_lab.semantic_knowledge import SchedulingKnowledgeBase
from causal_schedule_lab.semantic_knowledge import compact_knowledge_context
from causal_schedule_lab.secondary_metric_knowledge import (
    SecondaryMetricKnowledgeBase,
)


class FakeProvider:
    name = "fake"
    model = "smart-critic"

    def __init__(self, payload: dict):
        self.payload = payload
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=json.dumps(self.payload, ensure_ascii=False),
            provider=self.name,
            requested_model=self.model,
            response_model=self.model,
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )


def _analysis() -> SemanticAnalysis:
    evidence = [
        {
            "file": "env.py",
            "symbol": "step",
            "detail": "代码固定首工序释放时间为零并执行最早可行插入。",
        }
    ]
    return SemanticAnalysis.model_validate(
        {
            "schema_version": "1.0",
            "language": "zh-CN",
            "summary": "标准静态零释放时间 JSSP，最小化 makespan。",
            "project_type": "scheduling_optimization",
            "problem_families": ["JSSP"],
            "environments": [
                {
                    "kind": "static",
                    "statement": "全部实例预先给定。",
                    "evidence": evidence,
                    "confidence": "high",
                }
            ],
            "objectives": [
                {
                    "id": "objective_1",
                    "kind": "makespan",
                    "sense": "minimize",
                    "priority": 1,
                    "statement": "最小化最大完工时间。",
                    "evidence": evidence,
                    "confidence": "high",
                }
            ],
            "constraints": [
                {
                    "id": "constraint_1",
                    "kind": "release_time",
                    "scope": "job",
                    "hard": True,
                    "statement": "所有作业释放时间固定为零。",
                    "evidence": evidence,
                    "confidence": "high",
                }
            ],
            "decisions": [
                {
                    "id": "decision_1",
                    "kind": "select",
                    "modifiable": True,
                    "statement": "选择下一个 eligible operation。",
                    "evidence": evidence,
                    "confidence": "high",
                }
            ],
            "oracles": [],
            "allowed_interventions": ["select"],
            "unknowns": [],
            "overall_confidence": "high",
        }
    )


def _fjsp_analysis(*, integrated_transport: bool = False) -> SemanticAnalysis:
    payload = _analysis().model_dump(mode="json")
    payload["summary"] = (
        "Distributed dynamic FJSP with AGV transport, interfactory transfer and "
        "rescheduling."
        if integrated_transport
        else "Classic static FJSP with machine eligibility and alternative processing times."
    )
    payload["problem_families"] = ["FJSP"]
    payload["constraints"] = [
        {
            "id": "constraint_1",
            "kind": "resource_eligibility",
            "scope": "operation",
            "hard": True,
            "statement": "Each operation selects one eligible machine.",
            "evidence": payload["constraints"][0]["evidence"],
            "confidence": "high",
        },
        {
            "id": "constraint_2",
            "kind": "transport" if integrated_transport else "no_overlap",
            "scope": "route" if integrated_transport else "resource",
            "hard": True,
            "statement": (
                "AGV transport and interfactory transfer synchronize operation readiness."
                if integrated_transport
                else "Operations assigned to one machine cannot overlap."
            ),
            "evidence": payload["constraints"][0]["evidence"],
            "confidence": "high",
        },
    ]
    payload["decisions"] = [
        {
            "id": "decision_1",
            "kind": "assign_resource",
            "modifiable": True,
            "statement": "Choose an eligible machine assignment.",
            "evidence": payload["decisions"][0]["evidence"],
            "confidence": "high",
        },
        {
            "id": "decision_2",
            "kind": "route" if integrated_transport else "sequence",
            "modifiable": True,
            "statement": (
                "Choose AGV and interfactory routes."
                if integrated_transport
                else "Sequence operations assigned to each machine."
            ),
            "evidence": payload["decisions"][0]["evidence"],
            "confidence": "high",
        },
    ]
    payload["allowed_interventions"] = [
        "assign_resource",
        "route" if integrated_transport else "sequence",
    ]
    return SemanticAnalysis.model_validate(payload)


def _impact_payload(*, must_validate: bool = True) -> dict:
    return {
        "schema_version": "1.0",
        "status": "pending_human_review",
        "project_summary": "标准零释放时间 JSSP。",
        "impacts": [
            {
                "constraint_id": "constraint_1",
                "role": "fixed_instance_structure",
                "context_scope": "classical_family",
                "feasibility_criticality": "high",
                "decision_leverage": "none",
                "objective_sensitivity": "very_low",
                "candidate_discrimination": "none",
                "controllability": "fixed",
                "variation_across_candidates": "constant_across_candidates",
                "validation": "required" if must_validate else "not_required",
                "optimization_attention": "exclude",
                "diagnosis_attention": "deprioritize",
                "rationale": "本项目所有作业释放时间固定为零，且不是策略决策；动态到达变体中权重会升高。",
                "knowledge_refs": ["JSP", "jsp_release_dates_dynamic"],
                "confidence": "high",
            }
        ],
        "excluded_non_scheduling_items": ["OR-Tools 对比运行时间"],
        "unresolved_questions": [],
        "overall_confidence": "high",
    }


def test_metric_recall_does_not_activate_negated_variants_from_summary() -> None:
    analysis = _fjsp_analysis()
    payload = analysis.model_dump(mode="json")
    payload["summary"] += " No setup, blocking, transport, or maintenance is implemented."
    analysis = SemanticAnalysis.model_validate(payload)

    knowledge = SecondaryMetricKnowledgeBase.load()
    context = knowledge.context_from_analysis(analysis)
    assert "machine_eligibility" in context.variant_heads
    assert "setup" not in context.variant_heads
    assert "blocking" not in context.variant_heads
    assert "transport" not in context.variant_heads

    recall = knowledge.recall(context)
    assert all(
        "variant_head:setup" not in option.matched_views
        for option in recall.options
    )


def test_exact_retrieval_distinguishes_jsp_from_fjsp() -> None:
    knowledge = SchedulingKnowledgeBase.load()
    jsp_hits = knowledge.retrieve("standard JSSP with disjunctive graph")
    assert jsp_hits[0].family == "JSP"
    assert all(item.family != "FJSP" for item in jsp_hits)

    fjsp_hits = knowledge.retrieve(
        "Flexible job shop FJSP with AGV transport and machine eligibility"
    )
    assert fjsp_hits[0].family == "FJSP"
    assert "fjsp_transport" in {
        item.id for item in fjsp_hits[0].matched_variants
    }

    assert SchedulingKnowledgeBase.load().retrieve(
        "generic neural network batch_size=32"
    ) == ()


def test_knowledge_retrieval_does_not_promote_negated_variants() -> None:
    knowledge = SchedulingKnowledgeBase.load()
    hits = knowledge.retrieve(
        "Classic FJSP; no setup, transport, maintenance, or worker constraint is implemented.",
        family_hints=("FJSP",),
    )
    assert hits[0].family == "FJSP"
    assert hits[0].matched_variants == ()
    compact = compact_knowledge_context(hits)
    assert "fjsp_setup_maintenance" not in compact
    assert "fjsp_transport" not in compact
    assert "fjsp_dual_resource" not in compact

    conditioned, patterns = knowledge.retrieve_conditioned_on_analysis(
        _fjsp_analysis()
    )
    assert conditioned[0].matched_variants == ()
    assert patterns == ()


def test_conditioned_knowledge_uses_structured_variant_constraint() -> None:
    payload = _fjsp_analysis().model_dump(mode="json")
    payload["constraints"][1]["kind"] = "setup_time"
    payload["constraints"][1]["statement"] = "Sequence-dependent setup is enforced."
    analysis = SemanticAnalysis.model_validate(payload)
    hits, patterns = SchedulingKnowledgeBase.load().retrieve_conditioned_on_analysis(
        analysis
    )
    assert {item.id for item in hits[0].matched_variants} == {
        "fjsp_setup_maintenance"
    }
    assert {item.pattern_id for item in patterns} == {
        "sequence_dependent_changeover"
    }


def test_engineering_pattern_retrieval_is_cross_family_and_auditable() -> None:
    knowledge = SchedulingKnowledgeBase.load()
    hits = knowledge.retrieve_engineering_patterns(
        "FJSP integrates AGV routing, collision avoidance and charging.",
        family_hints=("FJSP",),
    )
    assert hits[0].pattern_id == "mobile_transport_coupling"
    assert "AGV" in hits[0].matched_terms
    assert hits[0].source_urls


def test_constraint_impact_keeps_hard_guard_but_can_assign_zero_leverage() -> None:
    analysis = _analysis()
    knowledge = SchedulingKnowledgeBase.load()
    hits = knowledge.retrieve(
        analysis.summary,
        family_hints=("JSSP",),
    )
    provider = FakeProvider(_impact_payload())
    report = assess_constraint_impacts_with_llm(
        analysis,
        hits,
        provider=provider,
    )
    impact = report.impacts[0]
    assert impact.validation.value == "required"
    assert impact.decision_leverage.value == "none"
    assert impact.decision_leverage_score == 0.0
    assert impact.optimization_attention.value == "exclude"
    assert provider.requests[0].reasoning_effort == "high"
    assert "低优化杠杆绝不等于删除" in provider.requests[0].system


def test_constraint_impact_program_owns_hard_constraint_validation() -> None:
    analysis = _analysis()
    hits = SchedulingKnowledgeBase.load().retrieve(
        analysis.summary,
        family_hints=("JSSP",),
    )
    report = assess_constraint_impacts_with_llm(
        analysis,
        hits,
        provider=FakeProvider(_impact_payload(must_validate=False)),
    )
    assert report.impacts[0].validation.value == "required"
    assert report.safety_overrides == (
        "impacts[constraint_1].validation=not_required->required",
    )


def test_constraint_impact_retains_unknown_fields_without_runtime_control() -> None:
    analysis = _analysis()
    payload = _impact_payload()
    payload["diagnostic_targets"] = [
        {
            "id": "diagnostic_1",
            "kind": "validator_coverage",
            "statement": "Check validator coverage.",
            "check": "Compare constraints against validator checks.",
            "rationale": "Keep validation separate from optimization.",
            "statement_note": "",
        }
    ]
    report = assess_constraint_impacts_with_llm(
        analysis,
        SchedulingKnowledgeBase.load().retrieve(
            analysis.summary, family_hints=("JSSP",)
        ),
        provider=FakeProvider(payload),
    )
    assert report.schema_normalizations == (
        "diagnostic_targets[0].statement_note",
    )
    assert report.schema_extensions[0].path == (
        "diagnostic_targets[0].statement_note"
    )
    assert report.schema_extensions[0].value == ""

    payload["diagnostic_targets"][0]["statement_note"] = "substantive extra"
    payload["model_commentary"] = {"recall_policy": "keep uncertain targets"}
    report = assess_constraint_impacts_with_llm(
        analysis,
        SchedulingKnowledgeBase.load().retrieve(
            analysis.summary, family_hints=("JSSP",)
        ),
        provider=FakeProvider(payload),
    )
    assert report.schema_normalizations == ()
    assert {item.path for item in report.schema_extensions} == {
        "model_commentary",
        "diagnostic_targets[0].statement_note",
    }
    assert next(
        item.value
        for item in report.schema_extensions
        if item.path == "diagnostic_targets[0].statement_note"
    ) == "substantive extra"


def test_secondary_metric_definition_is_owned_by_code_not_llm() -> None:
    analysis = _analysis()
    payload = _impact_payload()
    payload["secondary_targets"] = [
        {
            "id": "secondary_1",
            "catalog_candidate_id": "maximum_machine_workload",
            "desired_direction": "decrease",
            "relation_to_primary_objective": "mediated",
            "controllability": "indirect",
            "availability": "computable",
            "source_constraint_ids": ["constraint_1"],
            "rationale": "选择固定目录中的标准机器工作负荷指标。",
            "confidence": "medium",
        }
    ]
    report = assess_constraint_impacts_with_llm(
        analysis,
        SchedulingKnowledgeBase.load().retrieve(
            analysis.summary, family_hints=("JSSP",)
        ),
        provider=FakeProvider(payload),
    )

    target = report.secondary_targets[0]
    assert target.catalog_candidate_id == "maximum_machine_workload"
    assert target.statement == "最大机器工作负荷"
    assert target.unit == "time"
    assert target.measurement_definition == "max_m sum_{o assigned m}(p_o + setup_o)"
    assert target.availability.value == "needs_adapter"
    assert target.confidence == "medium"


def test_open_world_metric_proposal_is_grounded_and_does_not_bypass_catalog() -> None:
    analysis = _analysis()
    payload = _impact_payload()
    payload["novel_secondary_targets"] = [
        {
            "id": "novel_secondary_1",
            "provisional_name": "解码器回填机会损失时间",
            "measurement_definition": (
                "sum over operations of earliest_insert_start minus chosen_append_start"
            ),
            "unit": "time",
            "measurement_scope": "construction_trajectory",
            "variation_across_candidates": "decision_dependent",
            "candidate_variation_basis": "不同动作序列产生不同的可回填机会与损失。",
            "intervention_handle": "改变解码器插入规则或动作序列。",
            "desired_direction": "decrease",
            "relation_to_primary_objective": "mediated",
            "controllability": "indirect",
            "availability": "needs_adapter",
            "source_constraint_ids": ["constraint_1"],
            "evidence_refs": ["env.py::step"],
            "catalog_gap_reason": "现有目录没有表达该解码器特有的回填机会损失。",
            "rationale": "该损失可能延长后续工序的就绪和完工时间。",
            "confidence": "medium",
            "promotion_status": "proposed",
        },
        {
            "id": "novel_secondary_2",
            "provisional_name": "最大机器工作负荷",
            "measurement_definition": "max machine assigned processing time",
            "unit": "time",
            "measurement_scope": "candidate_schedule",
            "variation_across_candidates": "decision_dependent",
            "candidate_variation_basis": "不同机器分配产生不同最大负荷。",
            "intervention_handle": "改变工序机器分配。",
            "desired_direction": "decrease",
            "relation_to_primary_objective": "mediated",
            "controllability": "indirect",
            "availability": "needs_adapter",
            "source_constraint_ids": ["constraint_1"],
            "evidence_refs": ["env.py::step"],
            "catalog_gap_reason": "重复项，用于验证去重门。",
            "rationale": "目录已经存在，因此不应晋级为 novel。",
            "confidence": "low",
            "promotion_status": "proposed",
        },
        {
            "id": "novel_secondary_3",
            "provisional_name": "无证据候选",
            "measurement_definition": "sum of an unavailable signal",
            "unit": "time",
            "measurement_scope": "candidate_schedule",
            "variation_across_candidates": "mixed",
            "candidate_variation_basis": "假定候选值不同，但证据缺失。",
            "intervention_handle": "未知排程决策。",
            "desired_direction": "decrease",
            "relation_to_primary_objective": "unknown",
            "controllability": "unknown",
            "availability": "unknown",
            "source_constraint_ids": ["constraint_1"],
            "evidence_refs": ["missing.py::ghost"],
            "catalog_gap_reason": "测试无效证据不会拖垮整个报告。",
            "rationale": "证据不存在，应单独拒绝。",
            "confidence": "low",
            "promotion_status": "proposed",
        },
    ]
    payload["novel_diagnostic_targets"] = [
        {
            "id": "novel_diagnostic_1",
            "provisional_name": "归一化资格集合一致性",
            "check": "比较归一化前后每个工序的 eligible machine 布尔集合。",
            "source_constraint_ids": ["constraint_1"],
            "evidence_refs": ["env.py::step"],
            "catalog_gap_reason": "固定诊断枚举没有该实现特有的一致性检查。",
            "rationale": "不一致会使两个求解器比较不同的问题实例。",
            "confidence": "high",
            "promotion_status": "proposed",
        }
    ]
    provider = FakeProvider(payload)
    report = assess_constraint_impacts_with_llm(
        analysis,
        SchedulingKnowledgeBase.load().retrieve(
            analysis.summary, family_hints=("JSSP",)
        ),
        provider=provider,
    )

    assert [item.id for item in report.novel_secondary_targets] == [
        "novel_secondary_1"
    ]
    assert report.novel_secondary_targets[0].promotion_status == "proposed"
    assert [item.id for item in report.novel_diagnostic_targets] == [
        "novel_diagnostic_1"
    ]
    assert len(report.novel_candidate_rejections) == 2
    assert any("exact_name_duplicate_of_catalog" in item for item in report.novel_candidate_rejections)
    assert any("unknown_evidence" in item for item in report.novel_candidate_rejections)
    assert "novel_secondary_targets" in provider.requests[0].system
    assert "env.py::step" in provider.requests[0].user


def test_packaged_knowledge_file_exists() -> None:
    knowledge_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "causal_schedule_lab"
        / "knowledge"
        / "scheduling_families.json"
    )
    assert knowledge_path.exists()


def test_incremental_secondary_metric_knowledge_is_complete_and_proposed() -> None:
    knowledge = SecondaryMetricKnowledgeBase.load()

    assert len(knowledge.metrics) == 85
    assert len(knowledge.diagnostics) == 36
    assert len(knowledge.memberships) == 1056
    assert len(knowledge.computability_requirements) == 85
    assert len(knowledge.reference_cases) == 21
    assert all(item.promotion_status == "proposed" for item in knowledge.metrics)
    assert knowledge.materialization_audit["all_checks_passed"] is True
    assert knowledge.metric("batch_formation_wait_time") is not None
    assert knowledge.metric("interfactory_transfer_wait_time") is not None


def test_large_catalog_contains_all_legacy_active_metric_kinds() -> None:
    knowledge = SecondaryMetricKnowledgeBase.load()
    assert len(knowledge.legacy_bindings) == 11
    assert all(
        knowledge.metric(item["candidate_id"]) is not None
        for item in knowledge.legacy_bindings
    )


def test_deterministic_recall_is_bounded_and_auditable() -> None:
    knowledge = SecondaryMetricKnowledgeBase.load()
    first = knowledge.recall_for_analysis(_analysis())
    second = knowledge.recall_for_analysis(_analysis())

    assert first == second
    assert first.catalog_metric_count == 85
    assert len(first.options) <= 20
    assert all(len(batch) <= 12 for batch in first.batches)
    assert first.prompt_character_estimate < 6000
    assert {item.candidate_id for item in first.options} >= {
        "maximum_machine_workload",
        "critical_resource_internal_idle_time",
    }


def test_fjsp_recall_uses_variant_gates_without_sending_whole_catalog() -> None:
    knowledge = SecondaryMetricKnowledgeBase.load()
    classic = knowledge.recall_for_analysis(_fjsp_analysis())
    integrated = knowledge.recall_for_analysis(
        _fjsp_analysis(integrated_transport=True)
    )
    classic_ids = {item.candidate_id for item in classic.options}
    integrated_ids = {item.candidate_id for item in integrated.options}

    assert len(classic.options) <= 20
    assert len(integrated.options) <= 20
    assert classic_ids >= {
        "assignment_processing_penalty",
        "scarce_eligibility_load",
        "maximum_machine_workload",
    }
    assert "interfactory_transfer_wait_time" not in classic_ids
    assert integrated_ids & {
        "interfactory_transfer_wait_time",
        "transport_induced_waiting_time",
        "machine_vehicle_sync_wait",
    }


def test_round4_contextual_roles_and_computability_boundaries_are_preserved() -> None:
    knowledge = SecondaryMetricKnowledgeBase.load()

    no_wait = knowledge.memberships_for(
        candidate_id="interoperation_wait_total",
        view_id="variant_head:no_wait",
    )
    assert len(no_wait) == 1
    assert no_wait[0].contextual_role == "diagnostic"
    assert no_wait[0].contextual_direction == "target_zero_constraint_check"

    no_idle = knowledge.memberships_for(
        candidate_id="machine_idle_time_total",
        view_id="variant_head:no_idle",
    )
    assert len(no_idle) == 1
    assert no_idle[0].contextual_role == "diagnostic"
    assert no_idle[0].contextual_direction == "target_zero_constraint_check"

    cvar = knowledge.computability("makespan_cvar")
    assert cvar is not None
    requirements = cvar.model_extra["computable_candidate_requirements"]
    assert requirements["currently_computable"] == (
        "unknown_until_real_project_IR_is_checked"
    )
    assert "scenario_set" in requirements["required_comparators"]


def test_round4_reference_cases_are_design_labels_not_blind_truth() -> None:
    knowledge = SecondaryMetricKnowledgeBase.load()
    case = knowledge.reference_case("R4_dynamic_rescheduling")

    assert case is not None
    assert case.reference_label_status == "design_reference_not_independent_truth"
    assert knowledge.capability_group("cap_transport_agv_events") is not None
