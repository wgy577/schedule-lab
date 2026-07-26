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
from .secondary_metric_knowledge import (
    MetricRecallResult,
    SecondaryMetricKnowledgeBase,
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


class SecondaryTargetKind(StrEnum):
    CRITICAL_RESOURCE_INTERNAL_IDLE_TIME = "critical_resource_internal_idle_time"
    OPERATION_READY_TO_START_WAITING_TIME = "operation_ready_to_start_waiting_time"
    RESOURCE_UTILIZATION_COEFFICIENT_OF_VARIATION = (
        "resource_utilization_coefficient_of_variation"
    )
    TOTAL_SEQUENCE_DEPENDENT_SETUP_TIME = "total_sequence_dependent_setup_time"
    TRANSPORT_INDUCED_WAITING_TIME = "transport_induced_waiting_time"
    TOTAL_BLOCKING_TIME = "total_blocking_time"
    RESOURCE_CAPACITY_IDLE_RATE = "resource_capacity_idle_rate"
    MACHINE_SELECTION_PROCESSING_TIME_INCREMENT = (
        "machine_selection_processing_time_increment"
    )
    MAXIMUM_MACHINE_WORKLOAD = "maximum_machine_workload"
    LOW_FLEXIBILITY_OPERATION_MACHINE_LOAD = (
        "low_flexibility_operation_machine_load"
    )
    RESTRICTED_DECODER_MAKESPAN_GAP = "restricted_decoder_makespan_gap"
    OTHER = "other"
    UNKNOWN = "unknown"


class DiagnosticTargetKind(StrEnum):
    ELIGIBILITY_INTEGRITY = "eligibility_integrity"
    OBJECTIVE_LOWER_BOUND_FIDELITY = "objective_lower_bound_fidelity"
    ORACLE_REFERENCE_VALIDITY = "oracle_reference_validity"
    VALIDATOR_COVERAGE = "validator_coverage"
    OTHER = "other"
    UNKNOWN = "unknown"


class ObjectiveRelation(StrEnum):
    DIRECT = "direct"
    MEDIATED = "mediated"
    HYPOTHESIZED = "hypothesized"
    UNKNOWN = "unknown"


class TargetAvailability(StrEnum):
    COMPUTABLE = "computable"
    NEEDS_ADAPTER = "needs_adapter"
    NEEDS_ORACLE = "needs_oracle"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


_SECONDARY_METRIC_CATALOG: dict[
    SecondaryTargetKind, tuple[str, str, str]
] = {
    SecondaryTargetKind.CRITICAL_RESOURCE_INTERNAL_IDLE_TIME: (
        "关键资源内部空闲时间",
        "关键资源上按开始时间排序后，相邻工序之间 max(0, start[i+1]-end[i]) 的总和；不含首工序开始前和末工序结束后的空闲。",
        "time",
    ),
    SecondaryTargetKind.OPERATION_READY_TO_START_WAITING_TIME: (
        "工序就绪后等待时间",
        "各工序 start[o]-max(release[o], max end[pred]) 的非负值总和。",
        "time",
    ),
    SecondaryTargetKind.RESOURCE_UTILIZATION_COEFFICIENT_OF_VARIATION: (
        "资源利用率变异系数",
        "同类已使用资源利用率的样本标准差除以均值；异构资源必须按 resource family 分层计算。",
        "coefficient_of_variation",
    ),
    SecondaryTargetKind.TOTAL_SEQUENCE_DEPENDENT_SETUP_TIME: (
        "序列相关准备时间总和",
        "候选日程中由相邻加工顺序触发的 setup/changeover 时间总和。",
        "time",
    ),
    SecondaryTargetKind.TRANSPORT_INDUCED_WAITING_TIME: (
        "运输引起的等待时间",
        "候选日程中由运输、车辆等待和运输同步造成的等待时间总和。",
        "time",
    ),
    SecondaryTargetKind.TOTAL_BLOCKING_TIME: (
        "总阻塞时间",
        "工件完成当前工序后，占用上游资源等待下游资源或缓冲可用的时间总和。",
        "time",
    ),
    SecondaryTargetKind.RESOURCE_CAPACITY_IDLE_RATE: (
        "资源容量空闲率",
        "(makespan×总资源容量-忙碌容量时间)/(makespan×总资源容量)，仅比较同实例同资源集合。",
        "fraction",
    ),
    SecondaryTargetKind.MACHINE_SELECTION_PROCESSING_TIME_INCREMENT: (
        "机器选择加工时间增量",
        "各工序已选机器加工时间减去该工序候选机器最短加工时间，再对全部工序求和。",
        "time",
    ),
    SecondaryTargetKind.MAXIMUM_MACHINE_WORKLOAD: (
        "最大机器工作负荷",
        "对每台机器累计分配给它的工序加工时间，再取所有机器中的最大值。",
        "time",
    ),
    SecondaryTargetKind.LOW_FLEXIBILITY_OPERATION_MACHINE_LOAD: (
        "低柔性工序机器负荷",
        "以候选机器数倒数为工序权重，累计到已选机器后取最大加权工作负荷。",
        "weighted_time",
    ),
    SecondaryTargetKind.RESTRICTED_DECODER_MAKESPAN_GAP: (
        "受限解码规则完工期差值",
        "同实例、同预算配对实验中，受限解码器 makespan 减去完整可行插入解码器 makespan。",
        "time",
    ),
    SecondaryTargetKind.OTHER: (
        "待定义次级指标",
        "当前目录无法表达；必须进入人工审核并新增具有固定计算口径的指标后才能参与优化。",
        "project_defined",
    ),
    SecondaryTargetKind.UNKNOWN: (
        "未知次级指标",
        "现有证据不足，暂不参与候选排序或优化。",
        "unknown",
    ),
}


class _SecondaryTargetPayload(FrozenModel):
    """LLM selects a metric; code owns its name, definition and unit."""

    id: str = Field(pattern=r"^secondary_[0-9]+$")
    catalog_candidate_id: str
    kind: SecondaryTargetKind | None = None
    desired_direction: Literal["decrease", "increase", "stabilize"]
    relation_to_primary_objective: ObjectiveRelation
    controllability: Controllability
    availability: TargetAvailability
    source_constraint_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=1400)
    confidence: Literal["high", "medium", "low"] | None = None


class SecondaryTargetChoice(FrozenModel):
    id: str = Field(pattern=r"^secondary_[0-9]+$")
    catalog_candidate_id: str
    kind: SecondaryTargetKind
    statement: str = Field(min_length=1, max_length=1000)
    measurement_definition: str = Field(min_length=1, max_length=1400)
    unit: str = Field(min_length=1, max_length=120)
    desired_direction: Literal["decrease", "increase", "stabilize"]
    relation_to_primary_objective: ObjectiveRelation
    controllability: Controllability
    availability: TargetAvailability
    source_constraint_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=1400)
    confidence: Literal["high", "medium", "low"] | None = None


class NovelSecondaryTargetProposal(FrozenModel):
    """Open-world metric proposal grounded in current project code evidence."""

    id: str = Field(pattern=r"^novel_secondary_[0-9]+$")
    provisional_name: str = Field(min_length=2, max_length=160)
    measurement_definition: str = Field(min_length=5, max_length=1400)
    unit: str = Field(min_length=1, max_length=120)
    measurement_scope: Literal["candidate_schedule", "construction_trajectory"]
    variation_across_candidates: Literal["decision_dependent", "mixed"]
    candidate_variation_basis: str = Field(min_length=5, max_length=1000)
    intervention_handle: str = Field(min_length=3, max_length=1000)
    desired_direction: Literal["decrease", "increase", "stabilize"]
    relation_to_primary_objective: ObjectiveRelation
    controllability: Controllability
    availability: TargetAvailability
    source_constraint_ids: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    catalog_gap_reason: str = Field(min_length=5, max_length=1000)
    rationale: str = Field(min_length=5, max_length=1400)
    confidence: Literal["high", "medium", "low"]
    promotion_status: Literal["proposed"] = "proposed"


class NovelDiagnosticProposal(FrozenModel):
    """Open-world validation or implementation-risk proposal."""

    id: str = Field(pattern=r"^novel_diagnostic_[0-9]+$")
    provisional_name: str = Field(min_length=2, max_length=160)
    check: str = Field(min_length=5, max_length=1400)
    source_constraint_ids: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    catalog_gap_reason: str = Field(min_length=5, max_length=1000)
    rationale: str = Field(min_length=5, max_length=1400)
    confidence: Literal["high", "medium", "low"]
    promotion_status: Literal["proposed"] = "proposed"


class DiagnosticTargetChoice(FrozenModel):
    id: str = Field(pattern=r"^diagnostic_[0-9]+$")
    kind: DiagnosticTargetKind
    statement: str = Field(min_length=1, max_length=1000)
    check: str = Field(min_length=1, max_length=1200)
    source_constraint_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=1400)


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
    secondary_targets: tuple[SecondaryTargetChoice, ...] = ()
    novel_secondary_targets: tuple[NovelSecondaryTargetProposal, ...] = ()
    novel_diagnostic_targets: tuple[NovelDiagnosticProposal, ...] = ()
    novel_candidate_rejections: tuple[str, ...] = ()
    diagnostic_targets: tuple[DiagnosticTargetChoice, ...] = ()
    excluded_non_scheduling_items: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    overall_confidence: Literal["high", "medium", "low"]
    assessment_kind: Literal["structured_llm_prior"] = "structured_llm_prior"
    empirically_validated: bool = False
    provider: str
    requested_model: str
    response_model: str | None = None
    usage: TokenUsage
    latency_seconds: float = Field(ge=0.0)
    metric_recall: MetricRecallResult
    schema_normalizations: tuple[str, ...] = ()
    schema_extensions: tuple["SchemaExtension", ...] = ()
    safety_overrides: tuple[str, ...] = ()


class SchemaExtension(FrozenModel):
    """Non-contract model output retained for audit without controlling runtime."""

    path: str
    value: Any

class _CriticPayload(FrozenModel):
    schema_version: Literal["1.0"]
    status: Literal["pending_human_review"]
    project_summary: str
    impacts: tuple[ConstraintImpactChoice, ...]
    secondary_targets: tuple[_SecondaryTargetPayload, ...] = ()
    novel_secondary_targets: tuple[NovelSecondaryTargetProposal, ...] = ()
    novel_diagnostic_targets: tuple[NovelDiagnosticProposal, ...] = ()
    diagnostic_targets: tuple[DiagnosticTargetChoice, ...] = ()
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
9. 本步骤只生成结构化专家先验，不是实验测得的因果效应。不得声称已经通过候选
   扰动、反事实、统计检验或 Oracle 回放验证。
10. 区分“约束是否满足”和“约束活跃程度/松弛量/等待机制”。所有可行候选都满足
    同一硬约束时，不能仅因该约束重要就声称其满足状态能区分候选；必须说明真正
    变化的是哪项决策、松弛量或中间机制。
11. 在 secondary_targets 中显式列出 makespan 之前可测量、可比较的次级目标或
    中间机制。catalog_candidate_id 必须从程序召回的候选清单中选择；不得创造 ID。
    次级目标不是最终原因，不得在本步骤声称已经找到真实影响因子。
12. “分配机器”“排列同机工序”是决策维度，不是次级目标；“机器分配质量”、
    “排序质量”“效率”等没有计算定义的宽泛短语禁止作为次级目标。机器分配需要
    拆成标准调度量，例如机器选择加工时间增量、最大机器工作负荷、低柔性工序负荷。
13. secondary_targets 必须是对一个候选日程或成对实验可计算的标量，并给出明确
    增减方向。资格集合是否正确、下界是否可靠、Oracle 是否有效等只用于校验，
    必须放入 diagnostic_targets，不得混入优化次级目标。
14. secondary target 的名称、measurement_definition 和 unit 由程序端目录固定；
    你只选择 catalog_candidate_id、方向、关系、可控性、可用性、来源约束和理由。
    kind 可省略，由程序根据兼容映射补全。禁止创造“质量、
    效率、压力、损失、拥塞、平衡程度”等自由指标。关键路径长度若与当前唯一主目标
    makespan 数值等价，只能作为目标分解信息，不得重复列为次级目标。
15. 若程序召回候选清单为空，secondary_targets 必须返回空数组；不得用自由文本补题。
16. 本轮采用高召回策略：不要人为限制 secondary_targets 数量。候选清单中只要与
    当前项目的 makespan 改进存在合理直接或间接关系就应保留；证据较弱时返回
    confidence=low，而不是为了显得精确而省略。后续程序和实验会负责筛除假阳性。
17. confidence 表示你对“该候选与当前项目相关”的认识置信度，不代表真实因果效应
    大小。允许 high/medium/low，低置信候选仍可返回。
18. 题库不是封闭世界。如果代码证据揭示了候选清单无法表达、但对当前主目标可能
    有关且能定义为标量的新机制，写入 novel_secondary_targets。必须给出明确测量定义、
    单位、方向、source_constraint_ids、evidence_refs、目录缺口理由和置信度；禁止用
    “质量、效率、复杂度、压力”等无计算口径的宽泛名称。
19. evidence_refs 必须逐字使用项目语义中提供的 `file::symbol` 键。若现有目录候选
    已能表达同一测量，不要重复提案。novel 项只进入 proposed 审核池，不代表已被采用。
    没有真正的目录外候选时必须返回空数组，不要为了填模板而创造新指标。
20. 资格/表示一致性、下界可靠性、Oracle 有效性、Validator 覆盖、训练或实现风险等
    不用于优化排序的代码特有发现，必须写入 novel_diagnostic_targets，而不是
    novel_secondary_targets。新诊断同样必须引用 source_constraint_ids 和 evidence_refs。
21. novel secondary 必须在同一实例的不同候选排程或不同构造轨迹之间发生变化，并
    填写 measurement_scope、variation_across_candidates、candidate_variation_basis 和
    intervention_handle。实例常量、实现配置差异或只用于验证可信度的量必须走诊断通道。
"""


def _critic_prompt(
    analysis: "SemanticAnalysis",
    knowledge_hits: tuple[KnowledgeHit, ...],
    engineering_hits: tuple[EngineeringPatternHit, ...],
    metric_recall: MetricRecallResult,
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
                {
                    **citation.model_dump(mode="json"),
                    "evidence_ref": (
                        f"{citation.file}::{citation.symbol or '-'}"
                    ),
                }
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
        "secondary_targets": [
            {
                "id": "secondary_1",
                "catalog_candidate_id": (
                    metric_recall.options[0].candidate_id
                    if metric_recall.options
                    else "none_of_above"
                ),
                "desired_direction": "decrease",
                "relation_to_primary_objective": "mediated",
                "controllability": "indirect",
                "availability": "needs_adapter",
                "source_constraint_ids": ["constraint_4"],
                "rationale": "由机器分配和同机排序共同影响，仍需候选实验校准",
                "confidence": "medium",
            }
        ],
        "novel_secondary_targets": [
            {
                "id": "novel_secondary_1",
                "provisional_name": "代码特有的可测量机制名称",
                "measurement_definition": "由代码可观测量构成的标量公式或确定性计算规则",
                "unit": "time",
                "measurement_scope": "construction_trajectory",
                "variation_across_candidates": "decision_dependent",
                "candidate_variation_basis": "说明同一实例的不同候选为何会产生不同数值",
                "intervention_handle": "说明哪项排程或解码决策可以改变该数值",
                "desired_direction": "decrease",
                "relation_to_primary_objective": "mediated",
                "controllability": "indirect",
                "availability": "needs_adapter",
                "source_constraint_ids": ["constraint_4"],
                "evidence_refs": ["path/to/file.py::function_name"],
                "catalog_gap_reason": "说明为什么候选目录中的指标不能表达该机制",
                "rationale": "说明它与当前项目主目标的关系",
                "confidence": "low",
                "promotion_status": "proposed",
            }
        ],
        "novel_diagnostic_targets": [
            {
                "id": "novel_diagnostic_1",
                "provisional_name": "代码特有的一致性或验证检查",
                "check": "可执行或可人工复核的检查方法",
                "source_constraint_ids": ["constraint_2"],
                "evidence_refs": ["path/to/file.py::function_name"],
                "catalog_gap_reason": "说明为什么固定诊断枚举无法准确表达",
                "rationale": "说明不检查会怎样污染候选比较或可行性",
                "confidence": "low",
                "promotion_status": "proposed",
            }
        ],
        "diagnostic_targets": [
            {
                "id": "diagnostic_1",
                "kind": "eligibility_integrity",
                "statement": "所有已选机器都属于对应工序的资格集合",
                "check": "逐操作检查 selected_machine in eligible_machines",
                "source_constraint_ids": ["constraint_2"],
                "rationale": "这是可行性/解析完整性校验，不是待最小化指标",
            }
        ],
        "excluded_non_scheduling_items": [],
        "unresolved_questions": [],
        "overall_confidence": "medium",
    }
    return (
        "请对下面的项目约束做情境化影响分析。\n\n"
        "项目语义：\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        + "\n\n检索到的调度知识先验：\n"
        + compact_knowledge_context(knowledge_hits, engineering_hits)
        + "\n\n程序确定性召回的次级指标候选（只能从这些 candidate_id 中选择）：\n"
        + json.dumps(
            [
                {
                    "candidate_id": item.candidate_id,
                    "name": item.canonical_name,
                    "role": item.canonical_role,
                    "formula": item.formula,
                    "unit": item.unit,
                    "catalog_direction": item.desired_direction,
                    "availability": item.availability,
                    "matched_views": item.matched_views,
                    "missing_ir_fields": item.missing_ir_fields,
                    "evidence_grade": item.evidence_grade,
                }
                for item in metric_recall.options
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n召回审计：\n"
        + json.dumps(
            {
                "catalog_version": metric_recall.catalog_version,
                "catalog_metric_count": metric_recall.catalog_metric_count,
                "eligible_before_limit": metric_recall.eligible_before_limit,
                "prompt_option_count": len(metric_recall.options),
                "batches": metric_recall.batches,
                "prompt_character_estimate": metric_recall.prompt_character_estimate,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n完整输出模板：\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\n\nJSON Schema：\n"
        + json.dumps(
            _CriticPayload.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
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


def _separate_unknown_fields(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[SchemaExtension, ...]]:
    """Separate non-contract fields while retaining their values for audit.

    Known fields remain strict. Unknown fields cannot control runtime behavior, but a
    useful model-side annotation should not invalidate the complete response.
    """

    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    paths: list[str] = []
    extensions: list[SchemaExtension] = []
    root_allowed = set(_CriticPayload.model_fields)
    for key in tuple(normalized):
        if key in root_allowed:
            continue
        value = normalized.pop(key)
        path = key
        extensions.append(SchemaExtension(path=path, value=value))
        if value in (None, "", [], {}):
            paths.append(path)
    targets = (
        ("impacts", ConstraintImpactChoice),
        ("secondary_targets", _SecondaryTargetPayload),
        ("novel_secondary_targets", NovelSecondaryTargetProposal),
        ("novel_diagnostic_targets", NovelDiagnosticProposal),
        ("diagnostic_targets", DiagnosticTargetChoice),
    )
    for field_name, model in targets:
        items = normalized.get(field_name, [])
        if not isinstance(items, list):
            continue
        allowed = set(model.model_fields)
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for key in tuple(item):
                if key in allowed:
                    continue
                value = item.pop(key)
                path = f"{field_name}[{index}].{key}"
                extensions.append(SchemaExtension(path=path, value=value))
                if value in (None, "", [], {}):
                    paths.append(path)
    return normalized, tuple(paths), tuple(extensions)


def assess_constraint_impacts_with_llm(
    analysis: "SemanticAnalysis",
    knowledge_hits: tuple[KnowledgeHit, ...],
    *,
    provider: ModelProvider,
    engineering_hits: tuple[EngineeringPatternHit, ...] = (),
    metric_knowledge: SecondaryMetricKnowledgeBase | None = None,
    max_output_tokens: int = 12000,
    thinking_mode: Literal["enabled", "disabled", "adaptive"] | None = "enabled",
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "high",
) -> ConstraintImpactReport:
    metric_knowledge = metric_knowledge or SecondaryMetricKnowledgeBase.load()
    metric_recall = metric_knowledge.recall_for_analysis(analysis)
    response = provider.complete(
        ModelRequest(
            system=_SYSTEM_PROMPT,
            user=_critic_prompt(
                analysis,
                knowledge_hits,
                engineering_hits,
                metric_recall,
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
        raw_payload, schema_normalizations, schema_extensions = _separate_unknown_fields(
            _extract_json_object(response.content)
        )
        payload = _CriticPayload.model_validate(
            raw_payload
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
    safety_overrides: list[str] = []
    safe_impacts: list[ConstraintImpactChoice] = []
    for impact in payload.impacts:
        if expected[impact.constraint_id].hard and (
            impact.validation != ValidationChoice.REQUIRED
        ):
            safety_overrides.append(
                f"impacts[{impact.constraint_id}].validation="
                f"{impact.validation.value}->required"
            )
            impact = impact.model_copy(
                update={"validation": ValidationChoice.REQUIRED}
            )
        safe_impacts.append(impact)
    allowed_candidate_ids = {item.candidate_id for item in metric_recall.options}
    selected_candidate_ids = [
        item.catalog_candidate_id for item in payload.secondary_targets
    ]
    if len(selected_candidate_ids) != len(set(selected_candidate_ids)):
        raise ValueError("secondary target candidate IDs must be unique")
    for target in payload.secondary_targets:
        if target.catalog_candidate_id not in allowed_candidate_ids:
            raise ValueError(
                "secondary target was not in deterministic recall options: "
                + target.catalog_candidate_id
            )
        unknown_constraints = set(target.source_constraint_ids) - set(expected)
        if unknown_constraints:
            raise ValueError(
                "secondary target references unknown constraints: "
                + ", ".join(sorted(unknown_constraints))
            )
    allowed_evidence_refs = {
        f"{citation.file}::{citation.symbol or '-'}"
        for constraint in analysis.constraints
        for citation in constraint.evidence
    }
    catalog_names = {
        " ".join(item.canonical_name.casefold().split())
        for item in metric_knowledge.metrics
    }
    diagnostic_catalog_names = {
        " ".join(item.canonical_name.casefold().split())
        for item in metric_knowledge.diagnostics
    }
    valid_novel_targets: list[NovelSecondaryTargetProposal] = []
    valid_novel_diagnostics: list[NovelDiagnosticProposal] = []
    novel_candidate_rejections: list[str] = []
    seen_novel_names: set[str] = set()
    for target in payload.novel_secondary_targets:
        normalized_name = " ".join(target.provisional_name.casefold().split())
        reasons: list[str] = []
        unknown_constraints = set(target.source_constraint_ids) - set(expected)
        unknown_evidence = set(target.evidence_refs) - allowed_evidence_refs
        if normalized_name in catalog_names:
            reasons.append("exact_name_duplicate_of_catalog")
        if normalized_name in seen_novel_names:
            reasons.append("duplicate_novel_name")
        if unknown_constraints:
            reasons.append(
                "unknown_constraints=" + ",".join(sorted(unknown_constraints))
            )
        if unknown_evidence:
            reasons.append(
                "unknown_evidence=" + ",".join(sorted(unknown_evidence))
            )
        if reasons:
            novel_candidate_rejections.append(
                f"{target.id}:" + ";".join(reasons)
            )
            continue
        seen_novel_names.add(normalized_name)
        valid_novel_targets.append(target)
    for target in payload.novel_diagnostic_targets:
        normalized_name = " ".join(target.provisional_name.casefold().split())
        reasons = []
        unknown_constraints = set(target.source_constraint_ids) - set(expected)
        unknown_evidence = set(target.evidence_refs) - allowed_evidence_refs
        if normalized_name in diagnostic_catalog_names:
            reasons.append("exact_name_duplicate_of_diagnostic_catalog")
        if normalized_name in seen_novel_names:
            reasons.append("duplicate_novel_name")
        if unknown_constraints:
            reasons.append(
                "unknown_constraints=" + ",".join(sorted(unknown_constraints))
            )
        if unknown_evidence:
            reasons.append(
                "unknown_evidence=" + ",".join(sorted(unknown_evidence))
            )
        if reasons:
            novel_candidate_rejections.append(
                f"{target.id}:" + ";".join(reasons)
            )
            continue
        seen_novel_names.add(normalized_name)
        valid_novel_diagnostics.append(target)
    for target in payload.diagnostic_targets:
        unknown_constraints = set(target.source_constraint_ids) - set(expected)
        if unknown_constraints:
            raise ValueError(
                "diagnostic target references unknown constraints: "
                + ", ".join(sorted(unknown_constraints))
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
    option_by_id = {item.candidate_id: item for item in metric_recall.options}
    legacy_kind_by_id = {
        item["candidate_id"]: item["legacy_kind"]
        for item in metric_knowledge.legacy_bindings
    }
    return ConstraintImpactReport(
        **payload.model_dump(
            mode="python",
            exclude={
                "impacts",
                "secondary_targets",
                "novel_secondary_targets",
                "novel_diagnostic_targets",
            },
        ),
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
            for item in safe_impacts
        ),
        secondary_targets=tuple(
            SecondaryTargetChoice(
                **item.model_dump(
                    mode="python",
                    exclude={"kind", "availability", "desired_direction"},
                ),
                kind=(
                    SecondaryTargetKind(legacy_kind_by_id[item.catalog_candidate_id])
                    if item.catalog_candidate_id in legacy_kind_by_id
                    else SecondaryTargetKind.OTHER
                ),
                statement=option_by_id[item.catalog_candidate_id].canonical_name,
                measurement_definition=(
                    option_by_id[item.catalog_candidate_id].formula
                    or "目录未提供可执行公式；保持 proposed，需补计算器。"
                ),
                unit=option_by_id[item.catalog_candidate_id].unit or "unknown",
                availability=(
                    TargetAvailability.NEEDS_ADAPTER
                    if option_by_id[item.catalog_candidate_id].missing_ir_fields
                    and item.availability == TargetAvailability.COMPUTABLE
                    else item.availability
                ),
                desired_direction=(
                    option_by_id[item.catalog_candidate_id].desired_direction
                    if option_by_id[item.catalog_candidate_id].desired_direction
                    in {"decrease", "increase", "stabilize"}
                    else item.desired_direction
                ),
            )
            for item in payload.secondary_targets
        ),
        novel_secondary_targets=tuple(valid_novel_targets),
        novel_diagnostic_targets=tuple(valid_novel_diagnostics),
        novel_candidate_rejections=tuple(novel_candidate_rejections),
        provider=response.provider,
        requested_model=response.requested_model,
        response_model=response.response_model,
        usage=response.usage,
        latency_seconds=response.latency_seconds,
        metric_recall=metric_recall,
        schema_normalizations=schema_normalizations,
        schema_extensions=schema_extensions,
        safety_overrides=tuple(safety_overrides),
    )
