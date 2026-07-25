"""Context-conditioned constraint impact assessment with an independent LLM critic."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .providers.base import ModelProvider, ModelRequest, TokenUsage
from .semantic_knowledge import (
    EngineeringPatternHit,
    KnowledgeHit,
    compact_knowledge_context,
)

if TYPE_CHECKING:
    from .llm_semantics import SemanticAnalysis


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ConstraintRole(StrEnum):
    FEASIBILITY_GUARD = "feasibility_guard"
    DECISION_LEVER = "decision_lever"
    OBJECTIVE_DRIVER = "objective_driver"
    STATE_PARAMETER = "state_parameter"
    FIXED_INSTANCE_STRUCTURE = "fixed_instance_structure"
    EVALUATION_ONLY = "evaluation_only"
    IMPLEMENTATION_ONLY = "implementation_only"
    UNKNOWN = "unknown"


class ContextScope(StrEnum):
    CLASSICAL_FAMILY = "classical_family"
    COMMON_VARIANT = "common_variant"
    PROJECT_SPECIFIC = "project_specific"
    EXPERIMENT_ONLY = "experiment_only"
    UNKNOWN = "unknown"


class ImpactLevel(StrEnum):
    NONE = "none"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class Controllability(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    FIXED = "fixed"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class CandidateVariation(StrEnum):
    CONSTANT = "constant_across_candidates"
    DECISION_DEPENDENT = "decision_dependent"
    EXOGENOUS = "exogenous"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AttentionChoice(StrEnum):
    REQUIRED = "required"
    CONSIDER = "consider"
    DEPRIORITIZE = "deprioritize"
    EXCLUDE = "exclude"
    PENDING_REVIEW = "pending_review"


class ValidationChoice(StrEnum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    PENDING_REVIEW = "pending_review"


class ConstraintImpactChoice(FrozenModel):
    constraint_id: str
    role: ConstraintRole
    context_scope: ContextScope
    feasibility_criticality: ImpactLevel
    decision_leverage: ImpactLevel
    objective_sensitivity: ImpactLevel
    candidate_discrimination: ImpactLevel
    controllability: Controllability
    variation_across_candidates: CandidateVariation
    validation: ValidationChoice
    optimization_attention: AttentionChoice
    diagnosis_attention: AttentionChoice
    rationale: str = Field(min_length=1, max_length=1600)
    knowledge_refs: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"]


class ConstraintImpact(ConstraintImpactChoice):
    feasibility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_leverage_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    objective_sensitivity_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    candidate_discrimination_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class ConstraintImpactReport(FrozenModel):
    schema_version: Literal["1.0"]
    status: Literal["pending_human_review"]
    project_summary: str
    impacts: tuple[ConstraintImpact, ...]
    excluded_non_scheduling_items: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    overall_confidence: Literal["high", "medium", "low"]
    provider: str
    requested_model: str
    response_model: str | None = None
    usage: TokenUsage
    latency_seconds: float = Field(ge=0.0)

class _CriticPayload(FrozenModel):
    schema_version: Literal["1.0"]
    status: Literal["pending_human_review"]
    project_summary: str
    impacts: tuple[ConstraintImpactChoice, ...]
    excluded_non_scheduling_items: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    overall_confidence: Literal["high", "medium", "low"]


_SYSTEM_PROMPT = """\
你是调度语义 Critic。你不负责再次摘抄所有约束，而是判断已提取内容对当前项目的
调度改进是否具有实际杠杆。

必须区分：
- feasibility_criticality：从 none/very_low/low/medium/high/critical/unknown 选择；
- decision_leverage：从同一档位选择；
- objective_sensitivity：从同一档位选择；
- candidate_discrimination：从同一档位选择。

严格规则：
1. 权重必须结合当前问题族、目标、直接决策和项目证据，不能把文献中的典型权重
   当作当前项目结论。
2. 所有 hard=true 的项目约束必须 validation=required；低优化杠杆绝不等于删除
   可行性检查。
3. 固定、对所有候选相同、且不受决策改变的参数可以具有很低 decision_leverage
   和 candidate_discrimination，但要说明何种变体下权重会升高。
4. 论文对比算法、训练硬件、benchmark、运行时间和作者性能主张属于
   evaluation_only，不得冒充调度硬约束。
5. 经典问题族知识只是检索先验，不是项目证据。项目是否具备某个变体必须由传入的
   SemanticAnalysis 及其证据支持。
6. 每条输入 constraint 必须且只能返回一次；不能创造 constraint_id。
7. 所有分类和影响档位只能从 JSON Schema 枚举选择；不要自行输出小数权重。
8. 输出严格 JSON，不要 Markdown。
"""


def _critic_prompt(
    analysis: "SemanticAnalysis",
    knowledge_hits: tuple[KnowledgeHit, ...],
    engineering_hits: tuple[EngineeringPatternHit, ...],
) -> str:
    constraints = [
        {
            "id": item.id,
            "kind": item.kind,
            "scope": item.scope,
            "hard": item.hard,
            "statement": item.statement,
            "confidence": item.confidence,
            "evidence": [
                citation.model_dump(mode="json")
                for citation in item.evidence
            ],
        }
        for item in analysis.constraints
    ]
    context = {
        "summary": analysis.summary,
        "problem_families": analysis.problem_families,
        "environments": [
            {
                "kind": item.kind,
                "statement": item.statement,
            }
            for item in analysis.environments
        ],
        "objectives": [
            {
                "id": item.id,
                "kind": item.kind,
                "sense": item.sense,
                "priority": item.priority,
                "statement": item.statement,
            }
            for item in analysis.objectives
        ],
        "decisions": [
            {
                "id": item.id,
                "kind": item.kind,
                "modifiable": item.modifiable,
                "statement": item.statement,
            }
            for item in analysis.decisions
        ],
        "constraints": constraints,
    }
    template = {
        "schema_version": "1.0",
        "status": "pending_human_review",
        "project_summary": "概括权重所依赖的项目条件",
        "impacts": [
            {
                "constraint_id": "constraint_1",
                "role": "feasibility_guard",
                "context_scope": "classical_family",
                "feasibility_criticality": "critical",
                "decision_leverage": "medium",
                "objective_sensitivity": "medium",
                "candidate_discrimination": "medium",
                "controllability": "indirect",
                "variation_across_candidates": "decision_dependent",
                "validation": "required",
                "optimization_attention": "consider",
                "diagnosis_attention": "consider",
                "rationale": "结合当前项目说明理由",
                "knowledge_refs": ["JSP"],
                "confidence": "medium",
            }
        ],
        "excluded_non_scheduling_items": [],
        "unresolved_questions": [],
        "overall_confidence": "medium",
    }
    return (
        "请对下面的项目约束做情境化影响分析。\n\n"
        "项目语义：\n"
        + json.dumps(context, ensure_ascii=False, indent=2, default=str)
        + "\n\n检索到的调度知识先验：\n"
        + compact_knowledge_context(knowledge_hits, engineering_hits)
        + "\n\n完整输出模板：\n"
        + json.dumps(template, ensure_ascii=False, indent=2)
        + "\n\nJSON Schema：\n"
        + json.dumps(
            _CriticPayload.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("constraint impact response must be a JSON object")
    return value


def assess_constraint_impacts_with_llm(
    analysis: "SemanticAnalysis",
    knowledge_hits: tuple[KnowledgeHit, ...],
    *,
    provider: ModelProvider,
    engineering_hits: tuple[EngineeringPatternHit, ...] = (),
    max_output_tokens: int = 12000,
    thinking_mode: Literal["enabled", "disabled", "adaptive"] | None = "enabled",
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "max",
) -> ConstraintImpactReport:
    response = provider.complete(
        ModelRequest(
            system=_SYSTEM_PROMPT,
            user=_critic_prompt(
                analysis,
                knowledge_hits,
                engineering_hits,
            ),
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            require_json=True,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            metadata={"task": "constraint_impact_critic"},
        )
    )
    try:
        payload = _CriticPayload.model_validate(
            _extract_json_object(response.content)
        )
    except (json.JSONDecodeError, ValueError, ValidationError) as error:
        raise ValueError(
            f"constraint impact critic failed schema validation: {error}"
        ) from error

    expected = {item.id: item for item in analysis.constraints}
    actual_ids = [item.constraint_id for item in payload.impacts]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected):
        raise ValueError(
            "constraint impact critic must return each input constraint exactly once"
        )
    for impact in payload.impacts:
        if (
            expected[impact.constraint_id].hard
            and impact.validation != ValidationChoice.REQUIRED
        ):
            raise ValueError(
                f"hard constraint {impact.constraint_id} cannot be removed from validation"
            )
    scores = {
        ImpactLevel.NONE: 0.0,
        ImpactLevel.VERY_LOW: 0.1,
        ImpactLevel.LOW: 0.25,
        ImpactLevel.MEDIUM: 0.5,
        ImpactLevel.HIGH: 0.75,
        ImpactLevel.CRITICAL: 1.0,
        ImpactLevel.UNKNOWN: None,
    }
    return ConstraintImpactReport(
        **payload.model_dump(mode="python", exclude={"impacts"}),
        impacts=tuple(
            ConstraintImpact(
                **item.model_dump(mode="python"),
                feasibility_score=scores[item.feasibility_criticality],
                decision_leverage_score=scores[item.decision_leverage],
                objective_sensitivity_score=scores[
                    item.objective_sensitivity
                ],
                candidate_discrimination_score=scores[
                    item.candidate_discrimination
                ],
            )
            for item in payload.impacts
        ),
        provider=response.provider,
        requested_model=response.requested_model,
        response_model=response.response_model,
        usage=response.usage,
        latency_seconds=response.latency_seconds,
    )
