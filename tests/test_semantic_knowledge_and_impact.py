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
    assert provider.requests[0].reasoning_effort == "max"
    assert "低优化杠杆绝不等于删除" in provider.requests[0].system


def test_constraint_impact_rejects_removing_hard_constraint_from_validation() -> None:
    analysis = _analysis()
    hits = SchedulingKnowledgeBase.load().retrieve(
        analysis.summary,
        family_hints=("JSSP",),
    )
    with pytest.raises(ValueError, match="cannot be removed"):
        assess_constraint_impacts_with_llm(
            analysis,
            hits,
            provider=FakeProvider(_impact_payload(must_validate=False)),
        )


def test_packaged_knowledge_file_exists() -> None:
    knowledge_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "causal_schedule_lab"
        / "knowledge"
        / "scheduling_families.json"
    )
    assert knowledge_path.exists()
