#!/usr/bin/env python3
"""Materialize Round 4 knowledge artifacts from the real Round 3 catalogs.

This script is intentionally deterministic.  It does not call an LLM, does not
promote evidence status, and never invents candidate identifiers.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "src"
    / "causal_schedule_lab"
    / "knowledge"
    / "secondary_metrics"
    / "round3"
)
OUT = SOURCE.parent / "round4"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


metric_path = SOURCE / "candidate_metrics_round2.jsonl"
diagnostic_path = SOURCE / "candidate_diagnostics_round2.jsonl"
membership_path = SOURCE / "candidate_view_memberships.jsonl"
gap_path = SOURCE / "coverage_gap_queue.json"
registry_path = SOURCE / "view_registry.json"
test_path = SOURCE / "round3_recall_test_cases.jsonl"

metrics = load_jsonl(metric_path)
diagnostics = load_jsonl(diagnostic_path)
memberships = load_jsonl(membership_path)
gap_doc = json.loads(gap_path.read_text(encoding="utf-8"))
registry = json.loads(registry_path.read_text(encoding="utf-8"))
round3_cases = load_jsonl(test_path)

metric_by_id = {row["metric_id"]: row for row in metrics}
diagnostic_by_id = {row["diagnostic_id"]: row for row in diagnostics}
metric_ids = set(metric_by_id)
diagnostic_ids = set(diagnostic_by_id)
all_known_ids = metric_ids | diagnostic_ids

if len(metrics) != 55 or len(metric_by_id) != 55:
    raise ValueError("Round 3 canonical metric catalog must contain 55 unique IDs")
if len(diagnostics) != 25 or len(diagnostic_by_id) != 25:
    raise ValueError("Round 3 diagnostic catalog must contain 25 unique IDs")
if len(memberships) != 623:
    raise ValueError("Round 3 membership relation must contain 623 rows")
if len(gap_doc["queue"]) != 44:
    raise ValueError("Round 3 gap queue must contain 44 rows")
if any(row["candidate_id"] not in metric_ids for row in memberships):
    raise ValueError("Membership references a non-canonical candidate ID")

source_manifest = {
    path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in [metric_path, diagnostic_path, membership_path, gap_path, registry_path, test_path]
}


# ---------------------------------------------------------------------------
# 1. Gap capability grouping
# ---------------------------------------------------------------------------

group_specs: list[dict[str, Any]] = [
    {
        "capability_group_id": "cap_core_schedule_timing",
        "name": "基础工序时间、等待与约束耦合事件",
        "metric_families": ["time_flow_waiting", "no_wait_no_idle"],
        "diagnostic_ids": ["event_log_completeness", "input_data_coverage", "formula_executability"],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["classic", "no_wait", "no_idle"],
        "required_ir_field_groups": [
            "operation identity and precedence",
            "processing start/end and predecessor completion",
            "machine sequence and machine-ready timestamps",
            "explicit no-wait/no-idle flags where applicable",
        ],
        "blocked_computations": [
            "缺少时间戳语义时不能分解作业等待、机器等待与同步等待",
            "缺少 no-wait/no-idle 标志时不能把 target-zero 诊断与普通次级指标区分",
        ],
        "evidence_needed": ["schedule event semantics", "no-wait/no-idle formal definitions"],
        "priority": "P0",
        "uncertainty": "medium",
    },
    {
        "capability_group_id": "cap_schedule_dag_critical_workload",
        "name": "调度 DAG、关键结构、柔性与瓶颈负荷",
        "metric_families": ["critical_slack", "critical_block", "bottleneck_workload", "stage_sync", "flexibility_routing"],
        "diagnostic_ids": ["same_instance_variation", "controllability_trace", "objective_path_completeness"],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["classic", "machine_eligibility"],
        "required_ir_field_groups": [
            "complete schedule DAG including machine and variant-specific edges",
            "parallel critical paths and slack semantics",
            "machine/stage workload and eligibility matrices",
            "neighborhood and validator traces for move-count diagnostics",
        ],
        "blocked_computations": [
            "DAG 缺边时关键路径、关键块和 slack 均不可信",
            "没有 eligibility 与可选加工时间时不能计算机器选择机会损失",
        ],
        "evidence_needed": ["disjunctive graph definitions", "critical-block neighborhoods", "flexibility and workload definitions"],
        "priority": "P0",
        "uncertainty": "medium_high",
    },
    {
        "capability_group_id": "cap_setup_batch_events",
        "name": "换型、工艺族切换与批处理事件",
        "metric_families": ["setup_batch"],
        "diagnostic_ids": ["event_log_completeness", "formula_executability", "direction_sign_stability"],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["setup", "sequence_dependent_setup", "family_setup", "batching"],
        "required_ir_field_groups": ["machine sequence", "job/operation family", "setup matrix", "setup start/end and anticipatory semantics"],
        "blocked_computations": ["只有 setup duration 时不能判断重叠与等待", "没有 family 或 predecessor 语义时不能计算切换次数"],
        "evidence_needed": ["sequence-dependent setup definitions", "anticipatory setup semantics", "batch/setup interaction"],
        "priority": "P1",
        "uncertainty": "medium",
    },
    {
        "capability_group_id": "cap_blocking_buffer_events",
        "name": "Blocking、有限缓冲与占机传播事件",
        "metric_families": ["blocking_buffer"],
        "diagnostic_ids": ["event_log_completeness", "validator_constraint_coverage", "variant_signature_completeness"],
        "applicable_problem_families": ["FSP", "HFSP", "JSP", "FJSP"],
        "applicable_variant_heads": ["blocking", "limited_buffer", "zero_buffer"],
        "required_ir_field_groups": ["buffer identity/capacity", "buffer enter/leave", "processing finish vs machine release", "blocking dependency edges"],
        "blocked_computations": ["混淆 processing finish 与 machine release 会系统性低估 blocking", "没有 buffer events 不能从 idle 反推占用面积"],
        "evidence_needed": ["blocking flow-shop event semantics", "finite-buffer models", "constraint validator coverage"],
        "priority": "P0",
        "uncertainty": "high",
    },
    {
        "capability_group_id": "cap_transport_agv_events",
        "name": "运输、AGV 路由与机车同步事件",
        "metric_families": ["transport_agv"],
        "diagnostic_ids": ["event_log_completeness", "validator_constraint_coverage", "oracle_determinism"],
        "applicable_problem_families": ["FJSP", "HFSP", "FSP", "JSP"],
        "applicable_variant_heads": ["transport", "AGV", "path_conflict", "routing"],
        "required_ir_field_groups": ["vehicle assignment", "origin/destination and route", "loaded/empty state", "pickup/delivery/conflict timestamps"],
        "blocked_computations": ["只有距离矩阵时不能计算实际空驶和冲突等待", "没有 route trace 时不能区分空驶与重定位"],
        "evidence_needed": ["integrated machine-vehicle scheduling", "conflict-free routing", "transport event trace"],
        "priority": "P1",
        "uncertainty": "medium_high",
    },
    {
        "capability_group_id": "cap_auxiliary_resource_events",
        "name": "人员、工具、维护与其他辅助资源事件",
        "metric_families": ["auxiliary_resource"],
        "diagnostic_ids": ["variant_signature_completeness", "event_log_completeness", "controllability_trace"],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["worker", "tool", "maintenance", "auxiliary_resource", "shift"],
        "required_ir_field_groups": ["resource identity/type", "eligibility/skill", "request-ready-start-release", "calendar and maintenance windows"],
        "blocked_computations": ["只有最终 assignment 时不能识别同步等待", "维护重叠属于状态因子，不能默认解释为单调次级目标"],
        "evidence_needed": ["dual-resource scheduling", "tool-constrained scheduling", "maintenance integration"],
        "priority": "P1",
        "uncertainty": "medium_high",
    },
    {
        "capability_group_id": "cap_stochastic_robustness",
        "name": "随机情景、尾部风险与鲁棒性",
        "metric_families": ["stochastic_robustness"],
        "diagnostic_ids": ["simulation_replication_precision", "statistical_precision", "direction_sign_stability"],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["stochastic", "robust", "tail_risk", "service_level"],
        "required_ir_field_groups": ["scenario identity", "probability/sample weight", "scenario outcomes", "risk threshold and random seed"],
        "blocked_computations": ["单个 schedule 不能计算方差、CVaR 或服务水平", "没有概率时只能报告样本统计而非分布期望"],
        "evidence_needed": ["stochastic scheduling", "scenario-based optimization", "risk measure definitions"],
        "priority": "P1",
        "uncertainty": "high",
    },
    {
        "capability_group_id": "cap_dynamic_rescheduling",
        "name": "动态事件、重调度稳定性与恢复",
        "metric_families": ["rescheduling_stability"],
        "diagnostic_ids": ["variant_signature_completeness", "paired_intervention_validity", "event_log_completeness"],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["dynamic", "rescheduling", "machine_breakdown", "new_job_arrival", "frozen_zone"],
        "required_ir_field_groups": ["baseline/revised schedule IDs", "operation correspondence", "disruption timestamp/type", "frozen-zone policy"],
        "blocked_computations": ["没有 baseline/revised 对应关系不能计算稳定性", "没有 frozen-zone 规则不能判断冻结区违例"],
        "evidence_needed": ["predictive-reactive scheduling", "schedule stability", "disruption management"],
        "priority": "P0",
        "uncertainty": "medium",
    },
    {
        "capability_group_id": "cap_solver_decoder_repair",
        "name": "求解器、解码器、修复与验证轨迹",
        "metric_families": ["decoder_solver"],
        "diagnostic_ids": [
            "formula_executability", "validator_constraint_coverage", "oracle_determinism",
            "oracle_version_consistency", "metric_gaming_sensitivity", "calculator_cost_budget",
        ],
        "applicable_problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "applicable_variant_heads": ["decoder", "solver", "repair", "validator", "schedule_generation"],
        "required_ir_field_groups": ["raw proposal", "decoded/repaired schedule", "reference comparator", "move/validator trace and version/seed"],
        "blocked_computations": ["只有最终 schedule 时不能计算 decoder regret 或 repair delay", "feasible move count 依赖明确邻域而非排程本体"],
        "evidence_needed": ["schedule generation schemes", "decoder/repair definitions", "solver reproducibility"],
        "priority": "P0",
        "uncertainty": "medium",
    },
]


def gap_groups(gap: dict[str, Any]) -> list[str]:
    text = " ".join([gap["gap_id"], *gap.get("view_ids", [])]).lower()
    groups: list[str] = []
    tests = [
        ("cap_setup_batch_events", ["setup"]),
        ("cap_blocking_buffer_events", ["blocking", "limited_buffer"]),
        ("cap_transport_agv_events", ["transport", "agv"]),
        ("cap_auxiliary_resource_events", ["worker", "tool", "maintenance", "auxiliary_resource"]),
        ("cap_stochastic_robustness", ["stochastic", "robust"]),
        ("cap_dynamic_rescheduling", ["dynamic", "rescheduling", "stability", "lifecycle:execution"]),
        ("cap_solver_decoder_repair", ["decoder", "solver", "algorithm_behavior", "lifecycle:validation"]),
        ("cap_schedule_dag_critical_workload", ["critical_structure", "bottleneck_workload", "flexibility", "lifecycle:planning"]),
        ("cap_core_schedule_timing", ["classic", "no_wait", "no_idle", "mechanism:waiting", "mechanism:idle", "lifecycle:scheduling"]),
    ]
    for group_id, tokens in tests:
        if any(token in text for token in tokens):
            groups.append(group_id)
    return groups


gap_by_id = {gap["gap_id"]: gap for gap in gap_doc["queue"]}
assigned_by_group: dict[str, list[str]] = defaultdict(list)
assigned_by_gap: dict[str, list[str]] = {}
for gap in gap_doc["queue"]:
    groups = gap_groups(gap)
    assigned_by_gap[gap["gap_id"]] = groups
    for group_id in groups:
        assigned_by_group[group_id].append(gap["gap_id"])

for spec in group_specs:
    families = set(spec.pop("metric_families"))
    spec["related_gap_ids"] = sorted(assigned_by_group[spec["capability_group_id"]])
    spec["candidate_ids"] = sorted(
        row["metric_id"] for row in metrics if row.get("metric_family") in families
    )
    spec["available_diagnostic_ids"] = sorted(spec.pop("diagnostic_ids"))
    spec["status"] = "knowledge_group_materialized_not_runtime_validated"

unassigned_gaps = sorted(gap_id for gap_id, groups in assigned_by_gap.items() if not groups)
multi_assigned_gaps = {gap_id: groups for gap_id, groups in assigned_by_gap.items() if len(groups) > 1}
group_candidate_union = {candidate_id for spec in group_specs for candidate_id in spec["candidate_ids"]}

gap_output = {
    "schema_version": "round4.0",
    "artifact_status": "knowledge_layer_materialized_not_runtime_validated",
    "source_manifest": source_manifest,
    "grouping_policy": {
        "candidate_identity": "candidate_ids reference the 55-member canonical metric catalog only",
        "diagnostic_identity": "available_diagnostic_ids reference the separate 25-member diagnostic catalog",
        "overlap": "a gap may belong to multiple capability groups when mechanisms intersect",
        "promotion_rule": "no proposed candidate is promoted by grouping",
    },
    "capability_groups": group_specs,
    "coverage_audit": {
        "input_gap_count": len(gap_doc["queue"]),
        "assigned_gap_count": len(gap_doc["queue"]) - len(unassigned_gaps),
        "unassigned_gap_ids": unassigned_gaps,
        "multi_assigned_gap_count": len(multi_assigned_gaps),
        "multi_assigned_gap_ids": multi_assigned_gaps,
        "canonical_candidate_count": len(metric_ids),
        "candidate_ids_covered_by_groups": len(group_candidate_union),
        "candidate_ids_not_grouped": sorted(metric_ids - group_candidate_union),
    },
}
write_json(OUT / "gap_capability_groups.json", gap_output)


# ---------------------------------------------------------------------------
# 2. All 623 membership recommendations
# ---------------------------------------------------------------------------

axis_policy = {
    "problem_family": ("all", "hard", "defer_to_family_consistency_check"),
    "variant_head": ("any", "soft", "retain_as_adjacent_with_lower_priority"),
    "mechanism": ("any", "soft", "retain_with_lower_priority"),
    "decision": ("any", "mixed", "retain_semantically_but_mark_not_controllable"),
    "role": ("any", "hard_when_role_requested", "exclude_when_query_is_role_restricted"),
    "lifecycle": ("any", "soft", "retain_with_lower_priority"),
    "evidence": ("threshold", "soft_rerank", "retain_with_evidence_warning"),
}


def stable_profile_id(activation: dict[str, Any]) -> str:
    canonical = json.dumps(activation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "condition_profile_" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


profile_counts = Counter(stable_profile_id(row["activation_conditions"]) for row in memberships)
canonical_roles = {row["metric_id"]: row["candidate_role"] for row in metrics}
membership_out: list[dict[str, Any]] = []
for row in memberships:
    axis, value = row["view_id"].split(":", 1)
    match_mode, hard_or_soft, missing_behavior = axis_policy[axis]
    activation = row["activation_conditions"]
    profile_id = stable_profile_id(activation)
    flags = ["computability_gate_separated_from_semantic_view_match"]
    if profile_counts[profile_id] > 1:
        flags.append("activation_profile_repeated_reuse_profile_id")
    if len(activation.get("problem_families", [])) > 2:
        flags.append("broad_family_activation_review")
    if row["contextual_role"] != canonical_roles[row["candidate_id"]]:
        flags.append("contextual_role_differs_from_canonical")
    if canonical_roles[row["candidate_id"]] == "unresolved":
        flags.append("unresolved_candidate_human_review_required")
    if not activation.get("required_ir_fields"):
        flags.append("no_required_ir_fields_declared")
    membership_out.append({
        "candidate_id": row["candidate_id"],
        "view_id": row["view_id"],
        "condition_profile_id": profile_id,
        "original_condition_summary": {
            "problem_families": activation.get("problem_families", []),
            "required_variant_heads": activation.get("required_variant_heads", []),
            "excluded_variant_heads": activation.get("excluded_variant_heads", []),
            "required_decisions": activation.get("required_decisions", []),
            "required_ir_field_count": len(activation.get("required_ir_fields", [])),
            "objective_context": activation.get("objective_context", []),
        },
        "recommended_condition": {
            "semantic_view_match": {
                "axis": axis,
                "match_mode": match_mode,
                "minimum_match_count": 1,
                "required_values": [value],
                "excluded_values": [],
                "hard_or_soft": hard_or_soft,
                "missing_field_behavior": missing_behavior,
            },
            "computability_gate": {
                "match_mode": "all",
                "required_ir_fields": activation.get("required_ir_fields", []),
                "hard_or_soft": "hard_for_computable_channel_only",
                "missing_field_behavior": "semantic_only",
            },
            "variant_applicability": {
                "match_mode": "all" if activation.get("required_variant_heads") else "not_required",
                "required_values": activation.get("required_variant_heads", []),
                "excluded_values": activation.get("excluded_variant_heads", []),
            },
            "decision_controllability": {
                "match_mode": "any" if activation.get("required_decisions") else "not_required",
                "required_values": activation.get("required_decisions", []),
                "missing_field_behavior": "retain_semantic_candidate_but_mark_not_currently_controllable",
            },
        },
        "contextual_role": row["contextual_role"],
        "contextual_direction": row["contextual_direction"],
        "priority_in_view": row["priority_in_view"],
        "reason": row["reason"],
        "review_flags": flags,
        "review_status": "rule_materialized_requires_runtime_and_human_validation",
    })

write_jsonl(OUT / "membership_condition_recommendations.jsonl", membership_out)


# ---------------------------------------------------------------------------
# 3. Computability requirements for every canonical candidate
# ---------------------------------------------------------------------------

family_event_semantics = {
    "time_flow_waiting": ["start/end timestamps share one time origin", "readiness timestamp is distinguished from actual start"],
    "critical_slack": ["schedule DAG is complete", "parallel critical paths use a declared tolerance"],
    "critical_block": ["machine arcs and zero-lag semantics are explicit", "critical-block definition and neighborhood are versioned"],
    "bottleneck_workload": ["processing/setup/resource-active time inclusions are declared", "capacity normalization is explicit"],
    "stage_sync": ["stage identity and stage-ready events are explicit"],
    "flexibility_routing": ["eligible alternatives and alternative processing times refer to the same operation"],
    "setup_batch": ["processing finish, setup start/end, and next processing start are distinct", "anticipatory/non-anticipatory setup semantics are declared"],
    "blocking_buffer": ["processing finish is distinct from machine release", "buffer enter/leave events and capacity semantics are explicit"],
    "no_wait_no_idle": ["constraint scope and target-zero semantics are explicit"],
    "transport_agv": ["loaded, empty, pickup, delivery, and conflict events are distinguished"],
    "auxiliary_resource": ["resource request, ready, acquisition, and release timestamps are distinguished"],
    "rescheduling_stability": ["baseline and revised operations have a stable correspondence", "disruption and frozen-zone semantics are versioned"],
    "stochastic_robustness": ["scenario identity, probability/sample weight, and random seed are retained"],
    "decoder_solver": ["proposal, decoded schedule, repaired schedule, comparator, and versions are retained separately"],
}


def comparator_requirements(row: dict[str, Any]) -> list[str]:
    family = row.get("metric_family")
    if family == "rescheduling_stability":
        return ["baseline_schedule", "revised_schedule", "operation_correspondence"]
    if family == "stochastic_robustness":
        return ["scenario_set", "scenario_weights_or_sampling_protocol"]
    if family == "decoder_solver":
        return ["raw_or_pre_repair_result", "decoded_or_repaired_result", "reference_comparator_definition"]
    if family in {"critical_slack", "critical_block"}:
        return ["declared_critical_path_tolerance_and_parallel_path_policy"]
    return []


computability_out: list[dict[str, Any]] = []
for row in metrics:
    required_fields = list(dict.fromkeys(row.get("required_ir_fields", row.get("calculator_inputs", []))))
    source_status = row.get("source_access_statuses", {})
    computability_out.append({
        "candidate_id": row["metric_id"],
        "canonical_name": row["canonical_name"],
        "canonical_role": row["candidate_role"],
        "promotion_status": row["promotion_status"],
        "semantic_candidate": {
            "status": "relevant_or_plausible_proposed",
            "problem_families": row.get("problem_family", []),
            "variant_heads": row.get("required_variant_heads", []),
            "mechanism_family": row.get("metric_family"),
            "objective_path": row.get("objective_path"),
            "counterexample_boundary": row.get("counterexample_boundary"),
            "evidence_grade": row.get("evidence_grade"),
            "source_access_statuses": source_status,
        },
        "computable_candidate_requirements": {
            "required_ir_fields": required_fields,
            "required_event_semantics": family_event_semantics.get(row.get("metric_family"), []),
            "required_comparators": comparator_requirements(row),
            "required_variant_heads": row.get("required_variant_heads", []),
            "required_decision_trace": row.get("required_decisions", []),
            "formula": row.get("project_canonical_formula", row.get("formula")),
            "formula_origin": row.get("formula_origin"),
            "unit": row.get("unit"),
            "missing_behavior": "semantic_only; " + row.get("missing_data_behavior", "return null and diagnostic missing_input"),
            "partial_metric_policy": "reject_unless_partial_quantity_has_a_distinct_declared_name",
            "currently_computable": "unknown_until_real_project_IR_is_checked",
        },
        "review_flags": (
            ["unresolved_definition_do_not_compute"] if row["candidate_role"] == "unresolved" else []
        ) + (
            ["no_complete_code_evidence"] if not row.get("code_evidence", {}).get("complete", False) else []
        ),
    })

write_jsonl(OUT / "computability_requirements.jsonl", computability_out)


# ---------------------------------------------------------------------------
# 4. Reference cases.  These are design labels, never independent truth.
# ---------------------------------------------------------------------------


def ref_case(
    case_id: str,
    label: str,
    profile: dict[str, Any],
    priority: list[str],
    acceptable: list[str],
    irrelevant: list[str],
    fallback: list[str],
    confidence: str,
    review: list[str],
) -> dict[str, Any]:
    return {
        "scenario_id": case_id,
        "label": label,
        "reference_label_status": "design_reference_not_independent_truth",
        "input_profile": profile,
        "priority_candidate_ids": priority,
        "acceptable_not_priority_candidate_ids": acceptable,
        "clearly_irrelevant_candidate_ids": irrelevant,
        "expected_fallbacks": fallback,
        "label_confidence": confidence,
        "human_review_items": review,
    }


reference_cases: list[dict[str, Any]] = []
for old in round3_cases:
    reference_cases.append(ref_case(
        "R4_" + old["scenario_id"],
        old["label"],
        {
            "problem_family": old["problem_family"],
            "variant_heads": old["variant_heads"],
            "mechanisms": old["mechanisms"],
            "modifiable_decisions": old["modifiable_decisions"],
            "available_ir_fields": old["available_ir_fields"],
        },
        old["known_relevant"],
        [cid for cid in old["retrieved_candidate_ids"] if cid not in old["known_relevant"] and cid not in old["irrelevant_candidate_ids"]],
        old["irrelevant_candidate_ids"],
        old["fallback_triggered"],
        "medium",
        ["Round 3 known_relevant 是架构验收标签，必须由独立 Critic/人工复核后才能用于评价"],
    ))

reference_cases.extend([
    ref_case(
        "R4_dynamic_rescheduling", "机器故障后的动态重调度",
        {"problem_family": "FJSP", "variant_heads": ["dynamic", "rescheduling"], "requested_roles": ["primary_objective", "diagnostic"]},
        ["start_time_deviation", "completion_time_deviation", "machine_sequence_inversion_count", "assignment_change_count", "recovery_duration", "frozen_zone_violation_count"],
        ["expected_relative_makespan_degradation"], ["empty_travel_time", "buffer_occupancy_area"],
        ["evidence_insufficient"], "high", ["确认 baseline/revised operation correspondence 与 frozen-zone 规则"],
    ),
    ref_case(
        "R4_setup_agv_worker", "Setup + AGV + Worker 组合变体",
        {"problem_family": "FJSP", "variant_heads": ["setup", "AGV", "worker"], "requested_roles": ["secondary_metric"]},
        ["total_setup_time", "critical_path_setup_time", "loaded_travel_time", "empty_travel_time", "pickup_waiting_time", "vehicle_conflict_delay", "machine_vehicle_sync_wait", "auxiliary_resource_wait_total"],
        ["maximum_machine_workload", "critical_block_count"], ["makespan_cvar", "start_time_deviation"],
        ["need_adjacent_view", "evidence_insufficient"], "medium", ["检查同步等待是否被 setup、运输和人员重复归因；按机制拆分批次"],
    ),
    ref_case(
        "R4_blocking_setup", "Blocking + Setup 组合变体",
        {"problem_family": "FSP", "variant_heads": ["blocking", "setup"], "requested_roles": ["secondary_metric", "experimental_feature"]},
        ["total_blocking_time", "maximum_blocking_chain_length", "buffer_occupancy_area", "minimum_buffer_headroom", "total_setup_time", "critical_path_setup_time"],
        ["stage_starvation_time", "machine_idle_time_total"], ["scarce_eligibility_load", "empty_travel_time"],
        ["evidence_insufficient"], "medium", ["确认 setup 与 blocking 时间是否重叠以及 machine release 语义"],
    ),
    ref_case(
        "R4_family_mismatch", "问题族标签与代码证据冲突",
        {"declared_problem_family": "JSP", "observed_features": ["eligible_machine_set", "processing_time_by_machine"], "requested_roles": ["diagnostic"]},
        ["variant_signature_completeness", "literature_code_data_consistency"],
        ["interoperation_wait_total", "critical_block_count"], ["makespan_cvar", "vehicle_conflict_delay"],
        ["need_adjacent_view"], "medium", ["以代码事实为主确认是否应识别为 FJSP；不得自动覆盖原标签"],
    ),
    ref_case(
        "R4_missing_transport_head", "代码含运输事件但变体标签缺失",
        {"problem_family": "FJSP", "declared_variant_heads": ["classic"], "observed_fields": ["vehicle_assignment", "travel_start", "travel_end"]},
        ["variant_signature_completeness", "loaded_travel_time", "empty_travel_time", "machine_vehicle_sync_wait"],
        ["vehicle_conflict_delay"], ["makespan_cvar", "frozen_zone_violation_count"],
        ["need_adjacent_view"], "medium", ["确认 vehicle 字段不是数据集附带但未参与约束的无效字段"],
    ),
    ref_case(
        "R4_missing_blocking_ir", "已声明 Blocking 但事件字段不足",
        {"problem_family": "FSP", "variant_heads": ["blocking"], "missing_ir_fields": ["machine_release", "buffer_enter", "buffer_leave"]},
        ["input_data_coverage", "event_log_completeness", "formula_executability"],
        ["total_blocking_time", "buffer_occupancy_area"], ["makespan_cvar", "empty_travel_time"],
        ["evidence_insufficient"], "high", ["阻止 computable channel 输出 blocking 数值；指标只保留在 semantic channel"],
    ),
    ref_case(
        "R4_unknown_charging_variant", "未注册的共享充电约束",
        {"problem_family": "FJSP", "unknown_features": ["mobile_robot", "shared_charger", "battery_state"]},
        ["variant_signature_completeness", "input_data_coverage"],
        ["vehicle_conflict_delay", "auxiliary_resource_wait_total", "maintenance_critical_overlap"], ["makespan_cvar"],
        ["none_of_above", "need_adjacent_view", "unresolved_new_candidate"], "low", ["不得自动创建正式候选；先建立新变体 head 和证据"],
    ),
    ref_case(
        "R4_secondary_only", "仅召回 makespan 上游次级指标",
        {"problem_family": "FJSP", "requested_roles": ["secondary_metric"], "objective_context": ["makespan_mechanism"]},
        ["interoperation_wait_total", "machine_queue_wait_total", "maximum_machine_workload", "machine_workload_cv", "assignment_processing_penalty", "scarce_eligibility_load"],
        ["total_setup_time", "loaded_travel_time"], ["makespan_cvar", "decoder_makespan_regret", "frozen_zone_violation_count"],
        ["evidence_insufficient"], "high", ["只有在对应 variant 激活时才加入 setup/transport 候选"],
    ),
    ref_case(
        "R4_diagnostic_only", "只审核 IR、公式、Oracle 与 Validator",
        {"requested_roles": ["diagnostic", "algorithm_diagnostic"], "lifecycle": ["validation"]},
        ["input_data_coverage", "formula_executability", "validator_constraint_coverage", "oracle_determinism", "oracle_version_consistency", "literature_code_data_consistency"],
        ["feasible_critical_move_count", "left_shift_opportunity_time", "decoder_makespan_regret", "feasibility_repair_delay"],
        ["maximum_machine_workload", "makespan_cvar"], ["need_adjacent_view"], "high", ["诊断目录与 55 个 canonical metric 目录必须保持来源区分"],
    ),
    ref_case(
        "R4_factor_only", "只召回影响因素与实验特征",
        {"requested_roles": ["factor", "experimental_feature"], "objective_context": ["mechanism_discovery"]},
        ["maintenance_critical_overlap"],
        ["machine_idle_time_total", "total_slack_sum", "critical_operation_ratio", "assignment_concentration_hhi"],
        ["makespan_cvar", "frozen_zone_violation_count"], ["need_adjacent_view"], "medium", ["不得把 secondary_metric 强行改为 factor；需结合次级目标逐层下钻"],
    ),
])

for case in reference_cases:
    refs = (
        case["priority_candidate_ids"]
        + case["acceptable_not_priority_candidate_ids"]
        + case["clearly_irrelevant_candidate_ids"]
    )
    unknown = sorted(set(refs) - all_known_ids)
    if unknown:
        raise ValueError(f"Reference case {case['scenario_id']} contains unknown IDs: {unknown}")

write_jsonl(OUT / "retrieval_reference_cases.jsonl", reference_cases)


# ---------------------------------------------------------------------------
# 5. Validation and report
# ---------------------------------------------------------------------------

membership_pairs = [(row["candidate_id"], row["view_id"]) for row in membership_out]
invalid_group_candidates = sorted(
    {cid for spec in group_specs for cid in spec["candidate_ids"]} - metric_ids
)
invalid_group_diagnostics = sorted(
    {cid for spec in group_specs for cid in spec["available_diagnostic_ids"]} - diagnostic_ids
)
invalid_case_ids = sorted({
    cid
    for case in reference_cases
    for field in ["priority_candidate_ids", "acceptable_not_priority_candidate_ids", "clearly_irrelevant_candidate_ids"]
    for cid in case[field]
    if cid not in all_known_ids
})

checks = {
    "canonical_metric_count_is_55": len(metrics) == 55 and len(metric_ids) == 55,
    "diagnostic_count_is_25": len(diagnostics) == 25 and len(diagnostic_ids) == 25,
    "membership_input_output_count_is_623": len(memberships) == len(membership_out) == 623,
    "membership_pairs_unique": len(membership_pairs) == len(set(membership_pairs)),
    "all_membership_candidate_ids_valid": all(row["candidate_id"] in metric_ids for row in membership_out),
    "all_44_gaps_assigned": len(unassigned_gaps) == 0 and len(assigned_by_gap) == 44,
    "all_55_candidates_have_computability_rows": len(computability_out) == 55 and {r["candidate_id"] for r in computability_out} == metric_ids,
    "group_candidate_references_valid": not invalid_group_candidates,
    "group_diagnostic_references_valid": not invalid_group_diagnostics,
    "reference_case_ids_valid": not invalid_case_ids,
    "no_status_promotion": all(row["promotion_status"] == "proposed" for row in metrics),
}

audit = {
    "schema_version": "round4.0",
    "checks": checks,
    "all_checks_passed": all(checks.values()),
    "counts": {
        "canonical_candidates": len(metrics),
        "diagnostics": len(diagnostics),
        "membership_recommendations": len(membership_out),
        "unique_condition_profiles": len(profile_counts),
        "coverage_gaps": len(gap_doc["queue"]),
        "capability_groups": len(group_specs),
        "computability_rows": len(computability_out),
        "reference_cases": len(reference_cases),
    },
    "invalid_references": {
        "group_candidates": invalid_group_candidates,
        "group_diagnostics": invalid_group_diagnostics,
        "reference_cases": invalid_case_ids,
    },
}
write_json(OUT / "round4_materialization_audit.json", audit)

report = f"""# Round 4 Knowledge Preparation Report

> 性质：知识层真实 ID 物化与规则整理  
> 状态：代码生成并通过结构校验；尚未接入运行时召回器，尚未做独立盲测  
> 数据来源：Round 3 的 55 个 canonical candidates、25 个 diagnostics、623 条 memberships、44 个 gaps

## 1. 本轮完成内容

本轮补齐了普通 GPT 因无法读取本地文件而未完成的工作。所有产物直接读取 Round 3 文件生成，未虚构 candidate ID，也未提升任何候选的证据状态。

- `gap_capability_groups.json`：把 44 个 gap 归入 9 个可重叠能力组。
- `membership_condition_recommendations.jsonl`：逐条覆盖全部 623 条 membership。
- `computability_requirements.jsonl`：为全部 55 个 canonical candidate 建立语义相关性与可计算性分离契约。
- `retrieval_reference_cases.jsonl`：保留 11 个 Round 3 架构验收场景，并补充 10 个组合、错误、缺字段、未知变体和角色定向场景，共 {len(reference_cases)} 个。
- `round4_materialization_audit.json`：记录数量、引用完整性和状态不晋升检查。

## 2. 关键数量核对

| 检查项 | 结果 |
|---|---:|
| canonical candidate | {len(metrics)} |
| diagnostic | {len(diagnostics)} |
| membership 输入/输出 | {len(memberships)} / {len(membership_out)} |
| 唯一条件 profile | {len(profile_counts)} |
| coverage gap 已分组 | {len(gap_doc['queue']) - len(unassigned_gaps)} / {len(gap_doc['queue'])} |
| capability group | {len(group_specs)} |
| computability rows | {len(computability_out)} |
| reference cases | {len(reference_cases)} |
| 全部自动检查通过 | {str(all(checks.values())).lower()} |

## 3. 关系条件的处理原则

每条 membership 被拆成四类条件：

1. `semantic_view_match`：判断候选是否与问题族、变体、机制、决策、角色或生命周期相关；
2. `computability_gate`：`required_ir_fields` 必须全部满足，才允许进入 computable channel；
3. `variant_applicability`：表达候选成立所需的变体；
4. `decision_controllability`：缺少可控决策时仍可保留语义候选，但标记为当前不可控。

重复激活条件通过 `condition_profile_id` 标识，后续程序可以抽成共享 profile。Round 4 不直接改写 Round 3 原始 membership。

## 4. 必须保留的限制

- 21 个参考场景是设计标签，不是独立真值。
- `currently_computable` 全部保持 `unknown_until_real_project_IR_is_checked`。
- `proposed` 不等于 `validated` 或 `causal`。
- diagnostic 目录与 55 个 canonical metric 目录保持身份隔离。
- 本轮没有实现真正的 deterministic retriever，也没有运行 Recall/Precision/NDCG 测试。
- 一个 gap 或候选允许出现在多个能力组，这是关系复用，不是实体复制。

## 5. 下一工程步骤

下一步应由 `causal_schedule_lab` 实现可执行召回器：读取项目画像，应用本轮条件，分别输出 semantic channel 与 computable channel，并通过独立 Critic/人工标签进行盲测。该步骤完成以前，不应继续宣称 Round 3 的 100% recall 是真实泛化结果。

## 6. 自动校验

详见 `round4_materialization_audit.json`。本报告生成时自动校验结果为：`all_checks_passed = {str(all(checks.values())).lower()}`。
"""
(OUT / "ROUND4_KNOWLEDGE_PREPARATION_REPORT.md").write_text(report, encoding="utf-8")

if not audit["all_checks_passed"]:
    raise SystemExit("Round 4 materialization audit failed")

print(json.dumps(audit, ensure_ascii=False, indent=2))
