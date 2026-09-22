"""Frozen, no-pickle SG-SCT v1 input compiler.

The compiler keeps three graph views separate:

* ``G_S`` is the heterogeneous scheduling context (job/operation/mode/resource).
* ``G_C`` is the realised directed operation execution graph.
* ``G_F`` is the compact operation-to-resource feasibility graph.

It deliberately does not materialise pairwise resource competition.  Eligibility
is represented by the bipartite ``G_F`` edges and realised contention by adjacent
machine-sequence edges in ``G_C``.  Unknown setup/calendar/event semantics are
never filled with synthetic values; they are reported in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .ir import Problem, Schedule


SCHEMA_NAME = "sg_sct_input_v1"
SCHEMA_VERSION = "1.2.1"
HISTORY_LIMIT = 8
APPEARANCE_RULES = tuple(f"A{i}" for i in range(1, 11))

# EdgeRole for the L4 shared-symptom cross connection (mirrors
# sg_sct_model_v1.EdgeRole.CAUSAL_CROSS_SYMPTOM).  Kept above the discrete
# role ids so `dial: edge_role != CAUSAL_FORBIDDEN` automatically includes it.
CROSS_SYMPTOM_ROLE = 5

NODE_TYPES = ("operation", "job", "resource", "mode")
GS_EDGE_TYPES = (
    "job_has_operation",
    "operation_of_job",
    "operation_has_mode",
    "mode_of_operation",
    "mode_requires_resource",
    "resource_supports_mode",
)
GC_EDGE_TYPES = ("precedence", "resource_sequence")
GF_EDGE_TYPES = ("eligible_resource",)
# Machine-as-hub: op -> resource bipartite edges let same-machine operations
# connect through the machine relay node (role 2, reverse-walkable).
MACHINE_HUB_EDGE_TYPE = "gc:machine"
# Rule-typed op-op edges: each appearance rule (A1..A10) gets its own edge type
# so the model learns a distinct edge embedding per rule ("每种规则不同表征").
RULE_EDGE_TYPES = tuple(f"gc:rule:{r}" for r in APPEARANCE_RULES)
# L4 shared-symptom edges: intra-block special edge + swap-feasible cross-block.
L4_EDGE_TYPES = ("block", "swap")
MODEL_EDGE_TYPES = tuple(
    [f"gs:{item}" for item in GS_EDGE_TYPES]      # 0-5
    + [f"gf:{item}" for item in GF_EDGE_TYPES]     # 6
    + [f"gc:{item}" for item in GC_EDGE_TYPES]     # 7-8
    + [MACHINE_HUB_EDGE_TYPE]                       # 9
    + list(RULE_EDGE_TYPES)                         # 10-19
    + [f"l4:{item}" for item in L4_EDGE_TYPES]      # 20-21
)
# Type-base offsets for the unified sparse edge assembly.
_GS_EDGE_COUNT = len(GS_EDGE_TYPES)
_GF_EDGE_COUNT = len(GF_EDGE_TYPES)
_GC_EDGE_COUNT = len(GC_EDGE_TYPES)
MACHINE_HUB_TYPE_INDEX = _GS_EDGE_COUNT + _GF_EDGE_COUNT + _GC_EDGE_COUNT
RULE_TYPE_BASE = MACHINE_HUB_TYPE_INDEX + 1
L4_TYPE_BASE = RULE_TYPE_BASE + len(RULE_EDGE_TYPES)

NODE_CONTINUOUS_FIELDS = (
    "job_weight",
    "resource_capacity",
    "mode_cost",
    "mode_energy",
    "resource_load",
    "resource_utilization",
    "constraint_est",
    "solver_decision_delay",
    "criticality",
)
NODE_TIME_FIELDS = (
    "start",
    "end",
    "duration",
    "operation_release",
    "job_release",
    "due",
    "constraint_est",
    "solver_decision_delay",
    "cp_slack",
)
NODE_CATEGORICAL_FIELDS = (
    "job_index",
    "operation_index",
    "resource_index",
    "mode_index_within_operation",
    "selected_mode",
    "selected_resource_index",
)
NODE_LABEL_FIELDS = (
    "strict_critical",
    "near_critical",
    "appearance_member",
) + APPEARANCE_RULES
TOKEN_CONTINUOUS_FIELDS = (
    "eligible_resource_count",
    "resource_sequence_position",
    "job_sequence_position",
    "constraint_est",
    "solver_decision_delay",
    "cp_slack",
    "criticality",
)
TOKEN_TIME_FIELDS = (
    "start",
    "end",
    "duration",
    "operation_release",
    "job_release",
    "due",
)
GC_EDGE_CONTINUOUS_FIELDS = (
    "actual_start_gap",
    "required_predecessor_duration",
    "constraint_slack",
    "actual_start_lag",
)
GF_EDGE_CONTINUOUS_FIELDS = ("processing_duration",)


@dataclass(frozen=True)
class SGSCTDataV1:
    """In-memory numeric tensors plus a JSON-safe audit manifest."""

    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value)
        if part
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _robust_normalize(
    raw: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Column-wise median/IQR normalization with explicit missingness.

    A zero-IQR column falls back to scaled MAD and finally to 1.0. Missing
    entries remain zero in the normalized tensor and are identified solely by
    ``observed``; no semantic value is imputed.
    """

    values = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(observed, dtype=np.bool_)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("raw and observed must be same-shape rank-2 arrays")
    normalized = np.zeros(values.shape, dtype=np.float32)
    centers = np.zeros(values.shape[1], dtype=np.float32)
    scales = np.ones(values.shape[1], dtype=np.float32)
    methods: list[str] = []
    for column in range(values.shape[1]):
        present = values[mask[:, column], column]
        if present.size == 0:
            methods.append("unobserved")
            continue
        center = float(np.median(present))
        q25, q75 = np.percentile(present, [25.0, 75.0])
        scale = float(q75 - q25)
        method = "iqr"
        if not math.isfinite(scale) or scale <= 0.0:
            scale = float(1.4826 * np.median(np.abs(present - center)))
            method = "scaled_mad"
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
            method = "unit_fallback"
        centers[column] = center
        scales[column] = scale
        normalized[mask[:, column], column] = (
            (present - center) / scale
        ).astype(np.float32)
        methods.append(method)
    return normalized, centers, scales, methods


def _topological_order(
    operation_count: int, edge_sources: Sequence[int], edge_targets: Sequence[int]
) -> list[int]:
    successors: list[list[int]] = [[] for _ in range(operation_count)]
    indegree = [0] * operation_count
    seen: set[tuple[int, int]] = set()
    for source, target in zip(edge_sources, edge_targets, strict=True):
        if (source, target) in seen:
            continue
        seen.add((source, target))
        successors[source].append(target)
        indegree[target] += 1
    ready = deque(index for index, degree in enumerate(indegree) if degree == 0)
    order: list[int] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for target in sorted(successors[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != operation_count:
        raise ValueError("realised execution graph G_C is cyclic")
    return order


def _appearance_payload(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnosis = payload.get("diagnosis", payload)
    if not isinstance(diagnosis, Mapping):
        raise ValueError("appearance payload diagnosis must be an object")
    raw_blocks = diagnosis.get("blocks", [])
    if not isinstance(raw_blocks, list):
        raise ValueError("appearance payload diagnosis.blocks must be a list")
    blocks: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_blocks):
        if not isinstance(entry, Mapping):
            raise ValueError(f"appearance block[{index}] must be an object")
        raw = entry.get("block", entry)
        if not isinstance(raw, Mapping):
            raise ValueError(f"appearance block[{index}].block must be an object")
        interval = raw.get("time_interval")
        if not (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in interval)
        ):
            raise ValueError(f"appearance block[{index}] has invalid time_interval")
        block_id = str(raw.get("block_id", ""))
        if not block_id:
            raise ValueError(f"appearance block[{index}] has no block_id")
        appearance_values = raw.get("appearance_values")
        if not isinstance(appearance_values, Mapping):
            appearance_values = {}
        specialness = entry.get("specialness")
        if not isinstance(specialness, Mapping):
            specialness = {}
        blocks.append(
            {
                "block_id": block_id,
                "operations": tuple(str(item) for item in raw.get("operations", ())),
                "rules": tuple(str(item) for item in raw.get("appearance_rules", ())),
                "interval": (float(interval[0]), float(interval[1])),
                "verification_status": str(raw.get("verification_status", "unknown")),
                "keep": str(entry.get("keep_or_prune", "unknown")),
                "h_score": float(entry.get("h_score", 0.0)),
                "priority": float(entry.get("priority", 0.0)),
                "prototype_match": float(entry.get("prototype_match", 0.0)),
                "appearance_values": {
                    str(key): float(value)
                    for key, value in appearance_values.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
                "specialness": {
                    str(key): float(value)
                    for key, value in specialness.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
                "evidence_ids": tuple(str(item) for item in raw.get("evidence_ids", ())),
            }
        )
    blocks.sort(key=lambda item: _natural_key(item["block_id"]))
    metadata = {
        "schema_version": diagnosis.get("schema_version"),
        "detector_version": diagnosis.get("detector_version"),
        "rule_catalog_version": diagnosis.get("rule_catalog_version"),
        "rule_snapshot_id": diagnosis.get("rule_snapshot_id"),
        "calibration_version": diagnosis.get("calibration_version"),
        "schedule_feasible": diagnosis.get("schedule_feasible"),
    }
    return blocks, metadata


# v2 §4 Appearance Query data contract.  Each retained appearance block is
# reduced to a primary-rule type id (int index into APPEARANCE_RULES) plus a
# fixed-dimension continuous feature vector.  The names double as the manifest's
# feature-column spec so the model can be built dimensionally from the bundle.
APPEARANCE_FEATURE_SPECS = (
    "appearance_score",   # primary-rule appearance value (bounded where possible)
    "h_score",            # symptom priority / strength from the detector
    "priority",           # block priority
    "prototype_match",    # similarity to an extreme prototype
    "block_size_frac",    # |block ops| / total ops
    "block_start_frac",   # block start / makespan
    "block_end_frac",     # block end / makespan
    "scarcity",
    "coverage",
    "environment_specificity",
    "load_rank",
    "flex_rate",
    "share_score",
    "underuse",
    "multi_job",
    "time_disadvantage",
    "flex_op",
)
_SPECIALNESS_DIMS = APPEARANCE_FEATURE_SPECS[7:]


def _appearance_type_index(block: Mapping[str, Any]) -> int:
    """Primary-rule type id (apex of the block's rule vector -> int index)."""
    rules = tuple(block.get("rules", ()))
    primary = rules[0] if rules else "A1"
    if primary not in APPEARANCE_RULES:
        raise ValueError(f"unsupported primary appearance rule: {primary!r}")
    return APPEARANCE_RULES.index(primary)


def _appearance_feature_vector(
    block: Mapping[str, Any], makespan: float, total_ops: int
) -> np.ndarray:
    """Fixed-dim normalized feature vector for one appearance block (v2 §4.1)."""
    rules = tuple(block.get("rules", ()))
    primary = rules[0] if rules else "A1"
    appearance_values = block.get("appearance_values", {})
    specialness = block.get("specialness", {})
    interval = block.get("interval", (0.0, 0.0))
    start, end = float(interval[0]), float(interval[1])
    size = max(1, len(block.get("operations", ())))
    appearance_score = float(appearance_values.get(primary, 0.0))
    fields = {
        "appearance_score": max(0.0, min(1.0, appearance_score)),
        "h_score": max(0.0, float(block.get("h_score", 0.0))),
        "priority": max(0.0, float(block.get("priority", 0.0))),
        "prototype_match": max(0.0, min(1.0, float(block.get("prototype_match", 0.0)))),
        "block_size_frac": size / max(1, total_ops),
        "block_start_frac": start / max(1.0, makespan),
        "block_end_frac": end / max(1.0, makespan),
    }
    for dim in _SPECIALNESS_DIMS:
        fields[dim] = max(0.0, min(1.0, float(specialness.get(dim, 0.0))))
    return np.asarray([fields[name] for name in APPEARANCE_FEATURE_SPECS], dtype=np.float32)


def compile_sg_sct_input_v1(
    problem: Problem,
    schedule: Schedule,
    appearance_payload: Mapping[str, Any],
    *,
    case_id: str,
    split_id: str | None = None,
    near_critical_threshold: float | None = None,
    near_critical_fraction: float = 0.05,
) -> SGSCTDataV1:
    """Compile canonical IR + verified appearance output into SG-SCT tensors."""

    if problem.id != schedule.problem_id:
        raise ValueError("problem and schedule IDs differ")
    if not case_id:
        raise ValueError("case_id is required")
    if not (0.0 <= near_critical_fraction <= 1.0):
        raise ValueError("near_critical_fraction must be in [0, 1]")

    operations = sorted(
        problem.operations,
        key=lambda item: (_natural_key(item.job_id), item.index, _natural_key(item.id)),
    )
    jobs = sorted(problem.jobs, key=lambda item: _natural_key(item.id))
    resources = sorted(problem.resources, key=lambda item: _natural_key(item.id))
    modes = sorted(
        ((operation, mode) for operation in operations for mode in operation.modes),
        key=lambda item: (_natural_key(item[0].id), _natural_key(item[1].id)),
    )
    assignments = schedule.assignment_map()
    if set(assignments) != {item.id for item in operations}:
        raise ValueError("SG-SCT v1 requires exactly one assignment for every operation")

    op_index = {item.id: index for index, item in enumerate(operations)}
    job_index = {item.id: index for index, item in enumerate(jobs)}
    resource_index = {item.id: index for index, item in enumerate(resources)}
    mode_index = {item[1].id: index for index, item in enumerate(modes)}
    selected_modes = {item.mode_id for item in assignments.values()}
    mode_owner = {mode.id: operation.id for operation, mode in modes}

    node_ids = (
        [f"operation:{item.id}" for item in operations]
        + [f"job:{item.id}" for item in jobs]
        + [f"resource:{item.id}" for item in resources]
        + [f"mode:{mode.id}" for _, mode in modes]
    )
    operation_offset = 0
    job_offset = len(operations)
    resource_offset = job_offset + len(jobs)
    mode_offset = resource_offset + len(resources)
    node_lookup = {node_id: index for index, node_id in enumerate(node_ids)}
    node_count = len(node_ids)

    # Compact G_S: explicit ownership/eligibility context and its reverse for
    # bidirectional contextual message passing. No resource competition pairs.
    gs_source: list[int] = []
    gs_target: list[int] = []
    gs_type: list[int] = []
    gs_ids: list[str] = []

    def add_gs(source: str, target: str, edge_type: str, edge_id: str) -> None:
        gs_source.append(node_lookup[source])
        gs_target.append(node_lookup[target])
        gs_type.append(GS_EDGE_TYPES.index(edge_type))
        gs_ids.append(edge_id)

    for operation in operations:
        op_node = f"operation:{operation.id}"
        job_node = f"job:{operation.job_id}"
        add_gs(job_node, op_node, "job_has_operation", f"gs:job_has_operation:{operation.job_id}->{operation.id}")
        add_gs(op_node, job_node, "operation_of_job", f"gs:operation_of_job:{operation.id}->{operation.job_id}")
        for mode in sorted(operation.modes, key=lambda item: _natural_key(item.id)):
            mode_node = f"mode:{mode.id}"
            add_gs(op_node, mode_node, "operation_has_mode", f"gs:operation_has_mode:{operation.id}->{mode.id}")
            add_gs(mode_node, op_node, "mode_of_operation", f"gs:mode_of_operation:{mode.id}->{operation.id}")
            for resource_id in sorted(mode.resources, key=_natural_key):
                resource_node = f"resource:{resource_id}"
                add_gs(mode_node, resource_node, "mode_requires_resource", f"gs:mode_requires_resource:{mode.id}->{resource_id}")
                add_gs(resource_node, mode_node, "resource_supports_mode", f"gs:resource_supports_mode:{resource_id}->{mode.id}")

    # Realised machine order is the only resource-contention relation in G_C.
    machine_predecessor: dict[str, str] = {}
    machine_position: dict[str, int] = {}
    resource_sequences: dict[str, list[str]] = {}
    for resource in resources:
        sequence = sorted(
            (
                assignment
                for assignment in assignments.values()
                if resource.id in problem.mode_map()[assignment.mode_id][1].resources
            ),
            key=lambda item: (item.start, item.end, _natural_key(item.operation_id)),
        )
        for left, right in zip(sequence, sequence[1:]):
            if left.end > right.start:
                raise ValueError(f"resource overlap on {resource.id}: {left.operation_id}, {right.operation_id}")
            machine_predecessor[right.operation_id] = left.operation_id
        for position, assignment in enumerate(sequence):
            machine_position[assignment.operation_id] = position
        resource_sequences[resource.id] = [item.operation_id for item in sequence]

    gc_source: list[int] = []
    gc_target: list[int] = []
    gc_type: list[int] = []
    gc_ids: list[str] = []
    gc_relation: list[str] = []

    for operation in operations:
        for predecessor_id in sorted(operation.predecessors, key=_natural_key):
            gc_source.append(op_index[predecessor_id])
            gc_target.append(op_index[operation.id])
            gc_type.append(GC_EDGE_TYPES.index("precedence"))
            gc_relation.append("precedence")
            gc_ids.append(f"gc:precedence:{predecessor_id}->{operation.id}")
        predecessor_id = machine_predecessor.get(operation.id)
        if predecessor_id is not None:
            resource_id = problem.mode_map()[assignments[operation.id].mode_id][1].resources[0]
            gc_source.append(op_index[predecessor_id])
            gc_target.append(op_index[operation.id])
            gc_type.append(GC_EDGE_TYPES.index("resource_sequence"))
            gc_relation.append("resource_sequence")
            gc_ids.append(f"gc:resource_sequence:{resource_id}:{predecessor_id}->{operation.id}")

    topological = _topological_order(len(operations), gc_source, gc_target)
    incoming: list[list[int]] = [[] for _ in operations]
    outgoing: list[list[int]] = [[] for _ in operations]
    for edge_index, (source, target) in enumerate(zip(gc_source, gc_target, strict=True)):
        incoming[target].append(edge_index)
        outgoing[source].append(edge_index)

    starts = np.asarray([assignments[item.id].start for item in operations], dtype=np.int64)
    ends = np.asarray([assignments[item.id].end for item in operations], dtype=np.int64)
    durations = ends - starts
    makespan = int(schedule.makespan)
    constraint_est = np.zeros(len(operations), dtype=np.int64)
    release_binding = np.zeros(len(operations), dtype=np.bool_)
    binding_edge = np.zeros(len(gc_source), dtype=np.bool_)
    for target in topological:
        operation = operations[target]
        release = max(operation.release, jobs[job_index[operation.job_id]].release)
        enabling = [release] + [int(ends[gc_source[edge]]) for edge in incoming[target]]
        estimate = max(enabling)
        constraint_est[target] = estimate
        if starts[target] < estimate:
            raise ValueError(f"{operation.id} starts before known enabling time")
        release_binding[target] = release == estimate
        for edge in incoming[target]:
            binding_edge[edge] = ends[gc_source[edge]] == estimate
    decision_delay = starts - constraint_est

    # Longest realised tail: known non-binding edges carry no hidden wait;
    # binding edges carry the target operation's explicit decision delay.
    tail = durations.astype(np.int64).copy()
    for source in reversed(topological):
        candidates = []
        for edge in outgoing[source]:
            target = gc_target[edge]
            candidates.append(int((decision_delay[target] if binding_edge[edge] else 0) + tail[target]))
        if candidates:
            tail[source] = durations[source] + max(candidates)
    cp_slack = makespan - (starts + tail)
    if np.any(cp_slack < 0):
        raise ValueError("negative CP slack indicates an inconsistent execution graph")
    strict_critical = cp_slack == 0
    if near_critical_threshold is None:
        near_threshold = max(1.0, float(makespan) * near_critical_fraction)
        near_threshold_source = "fixed_fraction_of_makespan"
    else:
        if near_critical_threshold < 0:
            raise ValueError("near_critical_threshold must be non-negative")
        near_threshold = float(near_critical_threshold)
        near_threshold_source = "explicit"
    near_critical = (cp_slack > 0) & (cp_slack <= near_threshold)
    criticality = np.clip(1.0 - cp_slack / max(float(makespan), 1.0), 0.0, 1.0)

    gc_continuous = np.zeros((len(gc_source), len(GC_EDGE_CONTINUOUS_FIELDS)), dtype=np.float32)
    gc_critical = np.zeros(len(gc_source), dtype=np.bool_)
    gc_near = np.zeros(len(gc_source), dtype=np.bool_)
    for edge, (source, target) in enumerate(zip(gc_source, gc_target, strict=True)):
        gap = int(starts[target] - ends[source])
        gc_continuous[edge] = (gap, int(durations[source]), gap, int(starts[target] - starts[source]))
        gc_critical[edge] = bool(binding_edge[edge] and strict_critical[source] and strict_critical[target])
        gc_near[edge] = bool(cp_slack[source] <= near_threshold and cp_slack[target] <= near_threshold)

    # G_F is a sparse operation-resource bipartite relation. Multiple modes for
    # the same operation/resource are rejected because the v1 edge would be
    # ambiguous rather than silently choosing one.
    gf_records: list[tuple[int, int, int, bool, str]] = []
    for operation in operations:
        by_resource: dict[str, list[Any]] = defaultdict(list)
        for mode in operation.modes:
            if len(mode.resources) != 1:
                raise ValueError(f"SG-SCT v1 FJSP adapter requires unary-resource modes: {mode.id}")
            by_resource[mode.resources[0]].append(mode)
        for resource_id in sorted(by_resource, key=_natural_key):
            choices = by_resource[resource_id]
            if len(choices) != 1:
                raise ValueError(f"ambiguous duplicate modes for {operation.id} on {resource_id}")
            mode = choices[0]
            gf_records.append((
                operation_offset + op_index[operation.id],
                resource_offset + resource_index[resource_id],
                mode.duration,
                mode.id == assignments[operation.id].mode_id,
                f"gf:eligible_resource:{operation.id}->{resource_id}",
            ))

    blocks, appearance_metadata = _appearance_payload(appearance_payload)
    appearance_labels = np.zeros((len(operations), len(APPEARANCE_RULES)), dtype=np.bool_)
    evidence_tier = np.zeros_like(appearance_labels, dtype=np.int8)
    appearance_member = np.zeros(len(operations), dtype=np.bool_)
    block_rows: list[int] = []
    block_columns: list[int] = []
    block_times = np.zeros((len(blocks), 2), dtype=np.float32)
    block_rules = np.zeros((len(blocks), len(APPEARANCE_RULES)), dtype=np.bool_)
    block_keep = np.zeros(len(blocks), dtype=np.bool_)
    block_h_score = np.zeros(len(blocks), dtype=np.float32)
    # v2 §4: each appearance block gets a primary-rule type id + a fixed-dim
    # continuous feature vector for the AppearanceTypeEmbedding / q_appearance.
    block_appearance_type = np.full(len(blocks), -1, dtype=np.int64)
    block_appearance_features = np.zeros(
        (len(blocks), len(APPEARANCE_FEATURE_SPECS)), dtype=np.float32
    )
    tier_map = {"rejected": 0, "insufficient_evidence": 1, "unknown": 1, "not_applicable": 1, "verified": 3}
    for block_index, block in enumerate(blocks):
        block_times[block_index] = block["interval"]
        block_keep[block_index] = block["keep"] == "keep"
        block_h_score[block_index] = block["h_score"]
        block_appearance_type[block_index] = _appearance_type_index(block)
        block_appearance_features[block_index] = _appearance_feature_vector(
            block, float(makespan), len(operations)
        )
        tier = tier_map.get(block["verification_status"], 1)
        for rule in block["rules"]:
            if rule not in APPEARANCE_RULES:
                raise ValueError(f"unsupported appearance rule in runtime input: {rule}")
            block_rules[block_index, APPEARANCE_RULES.index(rule)] = True
        for operation_id in block["operations"]:
            if operation_id not in op_index:
                raise ValueError(f"appearance block references unknown operation: {operation_id}")
            operation_position = op_index[operation_id]
            block_rows.append(block_index)
            block_columns.append(operation_position)
            appearance_member[operation_position] = True
            for rule in block["rules"]:
                rule_index = APPEARANCE_RULES.index(rule)
                appearance_labels[operation_position, rule_index] = True
                evidence_tier[operation_position, rule_index] = max(
                    evidence_tier[operation_position, rule_index], tier
                )

    # ------------------------------------------------------------------
    # Sparse typed causal/shared graph (replaces the dense L4 cross-block
    # explosion with 4 principled edge families):
    #   * machine-hub  op <-> resource bipartite (role 2): resources are relay
    #     nodes, so same-machine operations connect through the machine hub.
    #   * rule-typed   op -> op per appearance rule A1..A10 (role 2): each rule
    #     gets its own edge embedding ("每种规则不同表征").
    #   * block        op -> op intra-block clique (role 5): members of a
    #     symptom block share a special edge.
    #   * swap         op -> op cross-block (role 5): only cross-machine +
    #     time-nearby + swap-feasible pairs (one op can move to the other's
    #     machine) — the "换机可行" edge that feeds M3 machine-swap operators.
    # G_C precedence + resource_sequence are kept (user: retain original info).
    # ------------------------------------------------------------------
    kept_indices = [b for b in range(len(blocks)) if bool(block_keep[b])]
    block_count = len(kept_indices)
    kept_position = {orig: pos for pos, orig in enumerate(kept_indices)}
    block_member_sets = [
        {op_index[item] for item in blocks[orig]["operations"] if item in op_index}
        for orig in kept_indices
    ]
    order_by_block = [
        sorted(members, key=lambda item: (starts[item], ends[item]))
        for members in block_member_sets
    ]
    l4_time_window = float(max(1.0, makespan) * 0.1)
    max_swap_neighbors = 8

    # Scheduled machine + alternative machines per operation (swap feasibility).
    mode_map = problem.mode_map()
    op_sched_res: dict[int, int] = {}
    op_alt_res: dict[int, set[int]] = {}
    for operation in operations:
        local = op_index[operation.id]
        mode = mode_map[assignments[operation.id].mode_id][1]
        op_sched_res[local] = resource_index[mode.resources[0]]
        op_alt_res[local] = {
            resource_index[r] for m in operation.modes for r in m.resources
        }

    # Where each op sits in its scheduled machine's realised processing order,
    # normalised to [0,1]. Stamped onto the machine-hub edges so the indirect
    # op -> machine -> op path carries explicit order (opA before opC on the
    # same machine is directly readable, not left to node-time inference).
    machine_seq_pos: dict[int, float] = {}
    for resource_id, ordered_ops in resource_sequences.items():
        count = len(ordered_ops)
        for pos, op_id in enumerate(ordered_ops):
            local = op_index.get(op_id)
            if local is not None:
                machine_seq_pos[local] = pos / max(1, count - 1)

    mh_source: list[int] = []
    mh_target: list[int] = []
    mh_ids: list[str] = []
    mh_pos: list[float] = []
    rule_source: list[int] = []
    rule_target: list[int] = []
    rule_type: list[int] = []
    rule_ids: list[str] = []
    l4_source: list[int] = []
    l4_target: list[int] = []
    l4_type: list[int] = []
    l4_ids: list[str] = []
    l4_strength: list[float] = []

    # Machine hub: op <-> resource bipartite (bidirectional, role 2).
    for local, res_idx in op_sched_res.items():
        pos = machine_seq_pos.get(local, 0.0)
        mh_source.append(operation_offset + local)
        mh_target.append(resource_offset + res_idx)
        mh_ids.append(f"gc:machine:{operations[local].id}->{resources[res_idx].id}")
        mh_pos.append(pos)
        mh_source.append(resource_offset + res_idx)
        mh_target.append(operation_offset + local)
        mh_ids.append(f"gc:machine:{resources[res_idx].id}->{operations[local].id}")
        mh_pos.append(pos)

    # Rule-typed op-op edges + block-internal special edges per kept block.
    for b in range(block_count):
        members = order_by_block[b]
        block_id = blocks[kept_indices[b]]["block_id"]
        for rule in np.flatnonzero(block_rules[kept_indices[b]]).tolist():
            rule_name = APPEARANCE_RULES[rule]
            for left in members:
                for right in members:
                    if left == right:
                        continue
                    rule_source.append(operation_offset + left)
                    rule_target.append(operation_offset + right)
                    rule_type.append(rule)
                    rule_ids.append(
                        f"gc:rule:{rule_name}:{operations[left].id}<->{operations[right].id}"
                    )
        for left in members:
            for right in members:
                if left == right:
                    continue
                l4_source.append(operation_offset + left)
                l4_target.append(operation_offset + right)
                l4_type.append(L4_EDGE_TYPES.index("block"))
                l4_strength.append(1.0)
                l4_ids.append(
                    f"l4:block:{block_id}:{operations[left].id}<->{operations[right].id}"
                )

    # Swap-feasible cross-block edges: cross-machine + time-nearby + swappable.
    for b1 in range(block_count):
        for b2 in range(b1 + 1, block_count):
            shared = np.flatnonzero(block_rules[kept_indices[b1]] & block_rules[kept_indices[b2]])
            if shared.size == 0:
                continue
            strength = float(shared.size) / len(APPEARANCE_RULES)
            for left in order_by_block[b1]:
                neighbors = 0
                for right in order_by_block[b2]:
                    if op_sched_res[left] == op_sched_res[right]:
                        continue  # same machine: machine hub already covers it
                    if not (
                        ends[left] <= starts[right] + l4_time_window
                        or ends[right] <= starts[left] + l4_time_window
                    ):
                        continue  # not time-nearby
                    if not (
                        op_sched_res[right] in op_alt_res[left]
                        or op_sched_res[left] in op_alt_res[right]
                    ):
                        continue  # not swap-feasible
                    l4_source.append(operation_offset + left)
                    l4_target.append(operation_offset + right)
                    l4_type.append(L4_EDGE_TYPES.index("swap"))
                    l4_strength.append(strength)
                    l4_ids.append(
                        f"l4:swap:{blocks[kept_indices[b1]]['block_id']}<->{blocks[kept_indices[b2]]['block_id']}:{operations[left].id}->{operations[right].id}"
                    )
                    neighbors += 1
                    if neighbors >= max_swap_neighbors:
                        break
    l4_edge_index = np.asarray((l4_source, l4_target), dtype=np.int64)
    # Keep a 2-D [E4,1] shape even when E4 == 0 (empty list collapses to 1-D).
    l4_edge_continuous = np.asarray(
        [[item] for item in l4_strength], dtype=np.float32
    ).reshape(-1, 1)

    # Full causal adjacency over the node space (operations + resources) for
    # ancestor tracing and neighborhood BFS.  Resources are relay hops; roots
    # stay operation-only via candidate_mask.
    causal_adj: list[list[int]] = [[] for _ in range(node_count)]
    causal_incoming: list[list[int]] = [[] for _ in range(node_count)]

    def _add_undirected(s: int, t: int) -> None:
        causal_adj[s].append(t)
        causal_adj[t].append(s)

    for s, t in zip(gc_source, gc_target):
        _add_undirected(operation_offset + s, operation_offset + t)
        causal_incoming[operation_offset + t].append(operation_offset + s)
    for s, t in zip(mh_source, mh_target):
        _add_undirected(s, t)
        causal_incoming[t].append(s)
    for s, t in zip(rule_source, rule_target):
        _add_undirected(s, t)
        causal_incoming[t].append(s)
    for s, t in zip(l4_source, l4_target):
        _add_undirected(s, t)
        causal_incoming[t].append(s)

    # Strict ancestors walk the full causal graph (machine hubs / rule / swap
    # included) but only operations can be roots.
    strict_ancestor = np.zeros(len(operations), dtype=np.bool_)
    queue = deque(np.flatnonzero(appearance_member).tolist())
    visited = set(queue)
    while queue:
        target = queue.popleft()
        for source in causal_incoming[target]:
            if source not in visited:
                visited.add(source)
                queue.append(source)
            if source < len(operations) and not appearance_member[source]:
                strict_ancestor[source] = True
    candidate_mask = strict_ancestor | appearance_member
    appearance_targets = np.flatnonzero(appearance_member)
    time_mask = np.zeros(len(operations), dtype=np.bool_)
    for source in range(len(operations)):
        time_mask[source] = any(
            source == target or ends[source] <= starts[target]
            for target in appearance_targets
        )
    legal_mask = candidate_mask & time_mask

    # Per-symptom-block structures: node membership, block count, and a
    # neighborhood mask bounding each block's root candidates to a local
    # region (graph BFS over the full causal graph up to radius R,
    # intersected with the candidate/time masks and a temporal window).
    kept_block_rows: list[int] = []
    kept_block_columns: list[int] = []
    for row, col in zip(block_rows, block_columns):
        if row in kept_position:
            kept_block_rows.append(kept_position[row])
            kept_block_columns.append(col)
    appearance_block_node_index = np.asarray(
        (kept_block_rows, [operation_offset + item for item in kept_block_columns]), dtype=np.int64
    )
    node_symptom_block = np.full(len(operations), -1, dtype=np.int64)
    for row, col in zip(kept_block_rows, kept_block_columns):
        if node_symptom_block[col] == -1:
            node_symptom_block[col] = row
    neighborhood_radius = 2
    block_neighborhood_mask = np.zeros((block_count, len(operations)), dtype=np.bool_)
    for b, members in enumerate(block_member_sets):
        frontier = set(members)
        reachable = set(members)
        for _ in range(neighborhood_radius):
            next_frontier: set[int] = set()
            for node in frontier:
                for neighbor in causal_adj[node]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        lo, hi = blocks[kept_indices[b]]["interval"]
        window = l4_time_window
        for node in reachable:
            if node >= len(operations):
                continue  # resources are relay hops, not roots
            if not (candidate_mask[node] and time_mask[node]):
                continue
            if not (float(starts[node]) <= hi + window and float(ends[node]) >= lo - window):
                continue
            block_neighborhood_mask[b, node] = True

    node_type = np.empty(node_count, dtype=np.int16)
    node_type[operation_offset:job_offset] = NODE_TYPES.index("operation")
    node_type[job_offset:resource_offset] = NODE_TYPES.index("job")
    node_type[resource_offset:mode_offset] = NODE_TYPES.index("resource")
    node_type[mode_offset:] = NODE_TYPES.index("mode")
    node_categorical = np.full((node_count, len(NODE_CATEGORICAL_FIELDS)), -1, dtype=np.int64)
    node_continuous = np.zeros((node_count, len(NODE_CONTINUOUS_FIELDS)), dtype=np.float32)
    node_continuous_observed = np.zeros_like(node_continuous, dtype=np.bool_)
    node_time = np.zeros((node_count, len(NODE_TIME_FIELDS)), dtype=np.float32)
    node_time_observed = np.zeros_like(node_time, dtype=np.bool_)
    node_labels = np.zeros((node_count, len(NODE_LABEL_FIELDS)), dtype=np.float32)

    selected_resource_for_operation: dict[str, str] = {}
    resource_busy = defaultdict(int)
    for operation in operations:
        assignment = assignments[operation.id]
        selected_mode = problem.mode_map()[assignment.mode_id][1]
        if len(selected_mode.resources) != 1:
            raise ValueError(f"selected mode is not unary-resource: {selected_mode.id}")
        selected_resource_for_operation[operation.id] = selected_mode.resources[0]
        resource_busy[selected_mode.resources[0]] += assignment.end - assignment.start

    for index, operation in enumerate(operations):
        assignment = assignments[operation.id]
        job = jobs[job_index[operation.job_id]]
        node_categorical[index] = (
            job_index[operation.job_id], operation.index, -1, -1, -1,
            resource_index[selected_resource_for_operation[operation.id]],
        )
        continuous_values = (job.weight, 0, 0, 0, 0, 0, constraint_est[index], decision_delay[index], criticality[index])
        continuous_mask = (1, 0, 0, 0, 0, 0, 1, 1, 1)
        node_continuous[index] = continuous_values
        node_continuous_observed[index] = continuous_mask
        due = operation.due if operation.due is not None else job.due
        node_time[index] = (
            assignment.start, assignment.end, assignment.end - assignment.start,
            operation.release, job.release, due or 0, constraint_est[index],
            decision_delay[index], cp_slack[index],
        )
        node_time_observed[index] = (1, 1, 1, 1, 1, due is not None, 1, 1, 1)
        node_labels[index, :3] = (strict_critical[index], near_critical[index], appearance_member[index])
        node_labels[index, 3:] = appearance_labels[index]

    for local, job in enumerate(jobs):
        index = job_offset + local
        node_categorical[index, 0] = local
        node_continuous[index, 0] = job.weight
        node_continuous_observed[index, 0] = True
        node_time[index, 4] = job.release
        node_time_observed[index, 4] = True
        if job.due is not None:
            node_time[index, 5] = job.due
            node_time_observed[index, 5] = True
    for local, resource in enumerate(resources):
        index = resource_offset + local
        node_categorical[index, 2] = local
        node_continuous[index, 1] = resource.capacity
        node_continuous[index, 4] = resource_busy[resource.id]
        node_continuous[index, 5] = resource_busy[resource.id] / max(makespan * resource.capacity, 1)
        node_continuous_observed[index, [1, 4, 5]] = True
    mode_position_within_operation: dict[str, int] = {}
    for operation in operations:
        for local, mode in enumerate(sorted(operation.modes, key=lambda item: _natural_key(item.id))):
            mode_position_within_operation[mode.id] = local
    for local, (operation, mode) in enumerate(modes):
        index = mode_offset + local
        node_categorical[index] = (
            job_index[operation.job_id], operation.index, resource_index[mode.resources[0]],
            mode_position_within_operation[mode.id], mode.id in selected_modes,
            resource_index[mode.resources[0]],
        )
        node_continuous[index, 2:4] = (mode.cost, mode.energy)
        node_continuous_observed[index, 2:4] = True
        node_time[index, 2] = mode.duration
        node_time_observed[index, 2] = True

    node_continuous_norm, node_cont_center, node_cont_scale, node_cont_methods = _robust_normalize(node_continuous, node_continuous_observed)
    node_time_norm, node_time_center, node_time_scale, node_time_methods = _robust_normalize(node_time, node_time_observed)

    token_order = sorted(range(len(operations)), key=lambda index: (starts[index], ends[index], _natural_key(operations[index].id)))
    token_to_operation = np.asarray(token_order, dtype=np.int64)
    operation_to_token = np.empty(len(operations), dtype=np.int64)
    operation_to_token[token_to_operation] = np.arange(len(operations), dtype=np.int64)
    token_continuous = np.zeros((len(operations), len(TOKEN_CONTINUOUS_FIELDS)), dtype=np.float32)
    token_time = np.zeros((len(operations), len(TOKEN_TIME_FIELDS)), dtype=np.float32)
    token_time_observed = np.ones_like(token_time, dtype=np.bool_)
    token_categorical = np.zeros((len(operations), 4), dtype=np.int64)
    for token, operation_position in enumerate(token_order):
        operation = operations[operation_position]
        assignment = assignments[operation.id]
        job = jobs[job_index[operation.job_id]]
        due = operation.due if operation.due is not None else job.due
        token_continuous[token] = (
            len({mode.resources[0] for mode in operation.modes}), machine_position[operation.id],
            operation.index, constraint_est[operation_position], decision_delay[operation_position],
            cp_slack[operation_position], criticality[operation_position],
        )
        token_time[token] = (
            assignment.start, assignment.end, assignment.end - assignment.start,
            operation.release, job.release, due or 0,
        )
        token_time_observed[token, 5] = due is not None
        token_categorical[token] = (
            job_index[operation.job_id], operation.index,
            resource_index[selected_resource_for_operation[operation.id]],
            mode_index[assignment.mode_id],
        )
    token_cont_observed = np.ones_like(token_continuous, dtype=np.bool_)
    token_cont_norm, token_cont_center, token_cont_scale, token_cont_methods = _robust_normalize(token_continuous, token_cont_observed)
    token_time_norm, token_time_center, token_time_scale, token_time_methods = _robust_normalize(token_time, token_time_observed)

    arrays: dict[str, np.ndarray] = {
        "gs_node_type": node_type,
        "gs_node_categorical": node_categorical,
        "gs_node_continuous_raw": node_continuous,
        "gs_node_continuous_normalized": node_continuous_norm,
        "gs_node_continuous_observed_mask": node_continuous_observed,
        "gs_node_time_raw": node_time,
        "gs_node_time_normalized": node_time_norm,
        "gs_node_time_observed_mask": node_time_observed,
        "gs_node_labels": node_labels,
        "gs_edge_index": np.asarray((gs_source, gs_target), dtype=np.int64),
        "gs_edge_type": np.asarray(gs_type, dtype=np.int16),
        "gc_edge_index": np.asarray((gc_source, gc_target), dtype=np.int64),
        "gc_edge_type": np.asarray(gc_type, dtype=np.int16),
        "gc_edge_continuous": gc_continuous,
        "gc_edge_binding_mask": binding_edge,
        "gc_edge_critical_mask": gc_critical,
        "gc_edge_near_critical_mask": gc_near,
        "gf_edge_index": np.asarray(([item[0] for item in gf_records], [item[1] for item in gf_records]), dtype=np.int64),
        "gf_edge_type": np.zeros(len(gf_records), dtype=np.int16),
        "gf_edge_continuous": np.asarray([[item[2]] for item in gf_records], dtype=np.float32),
        "gf_edge_selected_mask": np.asarray([item[3] for item in gf_records], dtype=np.bool_),
        "operation_constraint_est": constraint_est,
        "operation_solver_decision_delay": decision_delay,
        "operation_release_binding_mask": release_binding,
        "operation_cp_slack": cp_slack,
        "operation_critical_mask": strict_critical,
        "operation_near_critical_mask": near_critical,
        "operation_criticality": criticality.astype(np.float32),
        "operation_appearance_labels": appearance_labels,
        "operation_appearance_evidence_tier": evidence_tier,
        "operation_appearance_mask": appearance_member,
        "operation_ancestor_mask": strict_ancestor,
        "operation_candidate_mask": candidate_mask,
        "operation_time_mask": time_mask,
        "operation_legal_mask": legal_mask,
        "appearance_block_membership_index": np.asarray((kept_block_rows, kept_block_columns), dtype=np.int64),
        "appearance_block_node_index": appearance_block_node_index,
        "appearance_block_neighborhood_mask": block_neighborhood_mask,
        "appearance_block_time": block_times,
        "appearance_block_rule_labels": block_rules,
        "appearance_block_keep_mask": block_keep,
        "appearance_block_h_score": block_h_score,
        "l4_edge_index": l4_edge_index,
        "l4_edge_type": np.asarray(l4_type, dtype=np.int16),
        "l4_edge_continuous": l4_edge_continuous,
        "machine_hub_edge_index": np.asarray((mh_source, mh_target), dtype=np.int64),
        "rule_edge_index": np.asarray((rule_source, rule_target), dtype=np.int64),
        "rule_edge_type": np.asarray(rule_type, dtype=np.int16),
        "gantt_token_to_operation_index": token_to_operation,
        "operation_to_gantt_token_index": operation_to_token,
        "gantt_token_categorical": token_categorical,
        "gantt_token_continuous_raw": token_continuous,
        "gantt_token_continuous_normalized": token_cont_norm,
        "gantt_token_continuous_observed_mask": token_cont_observed,
        "gantt_token_time_raw": token_time,
        "gantt_token_time_normalized": token_time_norm,
        "gantt_token_time_observed_mask": token_time_observed,
        "gantt_token_labels": node_labels[token_to_operation],
        "history_operation_continuous": np.zeros((HISTORY_LIMIT, len(operations), len(TOKEN_CONTINUOUS_FIELDS)), dtype=np.float32),
        "history_operation_observed_mask": np.zeros((HISTORY_LIMIT, len(operations), len(TOKEN_CONTINUOUS_FIELDS)), dtype=np.bool_),
        "history_valid_mask": np.zeros(HISTORY_LIMIT, dtype=np.bool_),
        "history_action_operator_id": np.full(HISTORY_LIMIT, -1, dtype=np.int64),
        "history_reward_metrics": np.zeros((HISTORY_LIMIT, 4), dtype=np.float32),
        "history_reward_observed_mask": np.zeros((HISTORY_LIMIT, 4), dtype=np.bool_),
        "normalization_node_continuous_center": node_cont_center,
        "normalization_node_continuous_scale": node_cont_scale,
        "normalization_node_time_center": node_time_center,
        "normalization_node_time_scale": node_time_scale,
        "normalization_token_continuous_center": token_cont_center,
        "normalization_token_continuous_scale": token_cont_scale,
        "normalization_token_time_center": token_time_center,
        "normalization_token_time_scale": token_time_scale,
    }

    # Direct SGSCTBatch adapter tensors.  The detailed frozen tensors above
    # remain authoritative; these are a deterministic projection into the
    # trainable model's flat sparse-batch interface.
    model_node_numeric = np.concatenate(
        (
            node_continuous_norm,
            node_time_norm,
            node_continuous_observed.astype(np.float32),
            node_time_observed.astype(np.float32),
            (node_categorical[:, 4:5] == 1).astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    model_edge_index = np.concatenate(
        (
            arrays["gs_edge_index"],
            arrays["gf_edge_index"],
            np.asarray((gc_source, gc_target), dtype=np.int64),
            np.asarray((mh_source, mh_target), dtype=np.int64),
            np.asarray((rule_source, rule_target), dtype=np.int64),
            l4_edge_index,
        ),
        axis=1,
    )
    model_edge_type = np.concatenate(
        (
            np.asarray(gs_type, dtype=np.int16),
            np.full(len(gf_records), len(GS_EDGE_TYPES), dtype=np.int16),
            np.asarray(gc_type, dtype=np.int16) + len(GS_EDGE_TYPES) + len(GF_EDGE_TYPES),
            np.full(len(mh_source), MACHINE_HUB_TYPE_INDEX, dtype=np.int16),
            np.asarray(rule_type, dtype=np.int16) + RULE_TYPE_BASE,
            np.asarray(l4_type, dtype=np.int16) + L4_TYPE_BASE,
        )
    )
    model_edge_role = np.concatenate(
        (
            np.zeros(len(gs_ids), dtype=np.int8),
            np.ones(len(gf_records), dtype=np.int8),
            np.full(len(gc_ids) + len(mh_source) + len(rule_source), 2, dtype=np.int8),
            np.full(len(l4_source), CROSS_SYMPTOM_ROLE, dtype=np.int8),
        )
    )
    model_edge_features = np.zeros((model_edge_index.shape[1], 4), dtype=np.float32)
    gf_begin = len(gs_ids)
    gc_begin = gf_begin + len(gf_records)
    mh_begin = gc_begin + len(gc_ids)
    rule_begin = mh_begin + len(mh_source)
    l4_begin = rule_begin + len(rule_source)
    model_edge_features[gf_begin:gc_begin, 0] = arrays["gf_edge_continuous"][:, 0]
    model_edge_features[gc_begin:mh_begin] = gc_continuous
    model_edge_features[mh_begin:rule_begin, 0] = np.asarray(mh_pos, dtype=np.float32)
    model_edge_features[l4_begin:, 0] = l4_edge_continuous[:, 0]
    model_symptom_mask = np.zeros(node_count, dtype=np.bool_)
    model_candidate_mask = np.zeros(node_count, dtype=np.bool_)
    model_symptom_mask[: len(operations)] = appearance_member
    model_candidate_mask[: len(operations)] = legal_mask
    model_gantt_numeric = np.concatenate(
        (
            token_cont_norm,
            token_time_norm,
            token_time_observed.astype(np.float32),
            appearance_labels[token_to_operation].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    mechanism_values = np.zeros((node_count, 3), dtype=np.float32)
    mechanism_mask = np.zeros((node_count, 3), dtype=np.bool_)
    mechanism_values[: len(operations), 0] = np.log1p(starts)
    mechanism_values[: len(operations), 1] = np.log1p(durations)
    mechanism_values[: len(operations), 2] = np.log1p(decision_delay)
    mechanism_mask[: len(operations)] = True
    arrays.update(
        {
            "model_node_numeric": model_node_numeric,
            "model_node_graph": np.zeros(node_count, dtype=np.int64),
            "model_edge_index": model_edge_index,
            "model_edge_type": model_edge_type,
            "model_edge_role": model_edge_role,
            "model_edge_features": model_edge_features,
            "model_symptom_node_mask": model_symptom_mask,
            "model_candidate_node_mask": model_candidate_mask,
            "model_symptom_block_count": np.asarray([block_count], dtype=np.int64),
            "model_appearance_type": np.asarray(
                [block_appearance_type[k] for k in kept_indices], dtype=np.int64
            ),
            # A valid no-query state has no kept blocks.  Preserve the schema
            # dimension instead of calling np.stack([]); no dummy Appearance
            # or dummy feature row is permitted.
            "model_appearance_features": (
                np.stack([block_appearance_features[k] for k in kept_indices]).astype(np.float32)
                if kept_indices
                else np.zeros((0, block_appearance_features.shape[1]), dtype=np.float32)
            ),
            "model_symptom_block_node_index": appearance_block_node_index,
            "model_node_symptom_block": np.concatenate(
                (node_symptom_block, np.full(node_count - len(operations), -1, dtype=np.int64))
            ),
            "model_symptom_block_neighborhood_mask": np.concatenate(
                (block_neighborhood_mask, np.zeros((block_count, node_count - len(operations)), dtype=np.bool_)),
                axis=1,
            ),
            "model_l4_edge_index": l4_edge_index,
            "model_gantt_numeric": model_gantt_numeric,
            "model_gantt_graph": np.zeros(len(operations), dtype=np.int64),
            "model_gantt_operation_node": token_to_operation.copy(),
            "model_gantt_machine_index": token_categorical[:, 2].copy(),
            "model_gantt_job_index": token_categorical[:, 0].copy(),
            "model_gantt_time_index": token_time[:, 0].astype(np.int64),
            "model_trajectory_numeric": np.zeros((0, 4), dtype=np.float32),
            "model_trajectory_graph": np.zeros(0, dtype=np.int64),
            "model_trajectory_step": np.zeros(0, dtype=np.int64),
            "model_mechanism_values": mechanism_values,
            "model_mechanism_value_mask": mechanism_mask,
        }
    )

    missing_fields: list[str] = []
    if split_id is None:
        missing_fields.append("split_id")
    if not any(operation.due is not None for operation in operations) and not any(job.due is not None for job in jobs):
        missing_fields.append("due_dates")
    if not any(resource.calendar for resource in resources):
        missing_fields.append("resource_calendars")
    if not any(mode.setup_family is not None for _, mode in modes):
        missing_fields.extend(("setup_families", "sequence_dependent_setup_times"))
    if not problem.events:
        missing_fields.extend(("dynamic_events", "baseline_schedule"))
    missing_fields.extend((
        "root_cause_block_labels",
        "root_causal_path_labels",
        "counterfactual_intervention_labels",
        "next_state_transition_targets",
        "solver_metric_targets",
        "mechanism_student_t_targets",
    ))

    manifest: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "split_id": split_id,
        "problem_id": problem.id,
        "problem_family": str(problem.kind),
        "makespan": makespan,
        "counts": {
            "operations": len(operations), "jobs": len(jobs), "resources": len(resources),
            "modes": len(modes), "gs_nodes": node_count, "gs_edges": len(gs_ids),
            "gc_edges": len(gc_ids), "gc_binding_edges": int(binding_edge.sum()),
            "gf_edges": len(gf_records), "appearance_blocks": len(blocks),
            "appearance_operations": int(appearance_member.sum()),
            "appearance_block_members": len(kept_block_rows),
            "machine_hub_edges": len(mh_source),
            "rule_edges": len(rule_source),
            "l4_edges": len(l4_source),
            "strict_critical_operations": int(strict_critical.sum()),
            "strict_critical_edges": int(gc_critical.sum()),
            "near_critical_operations": int(near_critical.sum()),
            "positive_solver_decision_delays": int((decision_delay > 0).sum()),
        },
        "id_spaces": {
            "node_ids": node_ids,
            "operation_ids": [item.id for item in operations],
            "job_ids": [item.id for item in jobs],
            "resource_ids": [item.id for item in resources],
            "mode_ids": [item[1].id for item in modes],
            "gs_edge_ids": gs_ids,
            "gc_edge_ids": gc_ids,
            "gf_edge_ids": [item[4] for item in gf_records],
            "machine_hub_edge_ids": mh_ids,
            "rule_edge_ids": rule_ids,
            "l4_edge_ids": l4_ids,
            "gantt_token_ids": [f"gantt:{operations[index].id}" for index in token_order],
            "appearance_block_ids": [blocks[k]["block_id"] for k in kept_indices],
        },
        "vocabularies": {
            "node_types": NODE_TYPES,
            "gs_edge_types": GS_EDGE_TYPES,
            "gc_edge_types": GC_EDGE_TYPES,
            "gf_edge_types": GF_EDGE_TYPES,
            "l4_edge_types": L4_EDGE_TYPES,
            "model_edge_types": MODEL_EDGE_TYPES,
            "model_edge_roles": {
                "0": "context", "1": "feasibility", "2": "causal_hard",
                "3": "causal_soft", "4": "causal_forbidden", "5": "causal_cross_symptom",
            },
            "appearance_rules": APPEARANCE_RULES,
            "appearance_feature_fields": APPEARANCE_FEATURE_SPECS,
            "appearance_evidence_tier": {"0": "rejected_or_absent", "1": "unknown_or_insufficient", "3": "program_verified"},
            "node_continuous_fields": NODE_CONTINUOUS_FIELDS,
            "node_time_fields": NODE_TIME_FIELDS,
            "node_categorical_fields": NODE_CATEGORICAL_FIELDS,
            "node_label_fields": NODE_LABEL_FIELDS,
            "token_continuous_fields": TOKEN_CONTINUOUS_FIELDS,
            "token_time_fields": TOKEN_TIME_FIELDS,
            "token_categorical_fields": ("job_index", "operation_index", "selected_resource_index", "selected_mode_index"),
            "gc_edge_continuous_fields": GC_EDGE_CONTINUOUS_FIELDS,
            "gf_edge_continuous_fields": GF_EDGE_CONTINUOUS_FIELDS,
            "history_reward_fields": ("delta_makespan", "delta_badness", "feasible", "edit_cost"),
            "model_mechanism_fields": ("log1p_start", "log1p_duration", "log1p_solver_decision_delay"),
        },
        "normalization": {
            "method": "column_median_iqr_then_scaled_mad_then_unit",
            "missing_policy": "zero_storage_plus_explicit_observed_mask_no_semantic_imputation",
            "node_continuous_methods": node_cont_methods,
            "node_time_methods": node_time_methods,
            "token_continuous_methods": token_cont_methods,
            "token_time_methods": token_time_methods,
        },
        "critical_subgraph": {
            "definition": "realised binding DAG with explicit per-target solver decision delay",
            "strict": "operation_cp_slack == 0",
            "near": "0 < operation_cp_slack <= configured threshold",
            "near_critical_threshold": near_threshold,
            "threshold_source": near_threshold_source,
            "near_critical_fraction": near_critical_fraction,
        },
        "mask_semantics": {
            "operation_ancestor_mask": "strict G_C ancestors of any appearance operation; appearance members excluded",
            "operation_candidate_mask": "ancestor union appearance",
            "operation_time_mask": "ends no later than at least one appearance target starts, or is that target",
            "operation_legal_mask": "candidate intersect time; diagnostic-candidate legality, not operator/action legality",
            "appearance_block_neighborhood_mask": "per-block root neighbourhood: BFS over L1/L2/L4 up to radius 2, intersected with candidate/time masks and a temporal window around the block interval",
            "history_valid_mask": "first-round input; all eight slots explicitly invalid",
        },
        "compactness": {
            "pairwise_resource_competition_materialized": False,
            "replacement": "sparse G_F operation-resource eligibility plus G_C adjacent realised resource sequence",
        },
        "appearance": appearance_metadata,
        "resource_sequences": resource_sequences,
        "missing_fields": sorted(set(missing_fields)),
        "boundaries": {
            "oracle_or_llm_called": False,
            "root_labels_inferred": False,
            "training_targets_inferred": False,
            "unknown_constraints_imputed": False,
            "safe_tensor_format": "NPZ numeric/bool arrays only; manifest JSON; no pickle/object arrays",
        },
    }
    manifest["arrays"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }
    validate_sg_sct_data_v1(SGSCTDataV1(arrays=arrays, manifest=manifest))
    return SGSCTDataV1(arrays=arrays, manifest=manifest)


def validate_sg_sct_data_v1(bundle: SGSCTDataV1) -> None:
    """Fail closed on unsafe dtypes, inconsistent IDs and broken alignments."""

    arrays, manifest = bundle.arrays, bundle.manifest
    for name, value in arrays.items():
        if value.dtype.kind in {"O", "U", "S", "V"}:
            raise ValueError(f"unsafe/non-numeric NPZ dtype for {name}: {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite tensor values in {name}")
    operation_count = manifest["counts"]["operations"]
    if arrays["gantt_token_to_operation_index"].shape != (operation_count,):
        raise ValueError("Gantt token alignment has wrong shape")
    inverse = arrays["operation_to_gantt_token_index"]
    forward = arrays["gantt_token_to_operation_index"]
    if not np.array_equal(forward[inverse], np.arange(operation_count)):
        raise ValueError("Gantt token/operation mapping is not bijective")
    if arrays["history_valid_mask"].shape != (HISTORY_LIMIT,):
        raise ValueError("history window must have exactly eight slots")
    if arrays["history_valid_mask"].any():
        raise ValueError("v1 real-case compiler expects an explicitly empty first-round history")
    block_count = int(arrays["model_symptom_block_count"][0])
    if block_count and "model_symptom_block_neighborhood_mask" in arrays:
        if arrays["model_symptom_block_neighborhood_mask"].shape[0] != block_count:
            raise ValueError("symptom block neighbourhood mask block count mismatch")
        if arrays["model_symptom_block_neighborhood_mask"].shape[1] != arrays["gs_node_type"].shape[0]:
            raise ValueError("symptom block neighbourhood mask node count mismatch")
        if arrays["model_node_symptom_block"].shape != (arrays["gs_node_type"].shape[0],):
            raise ValueError("model_node_symptom_block must have length node_count")
    appearance_feature_fields = (
        bundle.manifest.get("vocabularies", {}).get("appearance_feature_fields", ())
    )
    if block_count and appearance_feature_fields:
        # v2 appearance query is only required for v2 bundles (manifest declares
        # the feature fields); pre-v2 frozen bundles run the legacy block query.
        if "model_appearance_type" not in arrays:
            raise ValueError("v2 appearance query requires model_appearance_type")
        if "model_appearance_features" not in arrays:
            raise ValueError("v2 appearance query requires model_appearance_features")
        if arrays["model_appearance_type"].shape != (block_count,):
            raise ValueError("model_appearance_type must have shape (block_count,)")
        if arrays["model_appearance_features"].shape[0] != block_count:
            raise ValueError("model_appearance_features must have block_count rows")
        if arrays["model_appearance_features"].shape[1] != len(APPEARANCE_FEATURE_SPECS):
            raise ValueError("model_appearance_features width must match appearance_feature_fields")


def write_sg_sct_data_v1(
    bundle: SGSCTDataV1, *, npz_path: Path, manifest_path: Path
) -> dict[str, str]:
    """Atomically write numeric NPZ and JSON manifest with cross-file hashes."""

    validate_sg_sct_data_v1(bundle)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = npz_path.with_name(f".{npz_path.name}.tmp.npz")
    np.savez_compressed(npz_tmp, **bundle.arrays)
    npz_tmp.replace(npz_path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(bundle.arrays):
            raise ValueError("NPZ key mismatch after write")
    manifest = dict(bundle.manifest)
    manifest["artifacts"] = {
        "npz_file": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "manifest_file": manifest_path.name,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_tmp.write_bytes(encoded)
    manifest_tmp.replace(manifest_path)
    return {"npz_sha256": _sha256(npz_path), "manifest_sha256": _sha256(manifest_path)}


def to_sg_sct_batch_v1(bundle: SGSCTDataV1, *, device: str | None = None) -> Any:
    """Project a compiled case into :class:`sg_sct_model_v1.SGSCTBatch`.

    PyTorch is imported lazily so deterministic dataset export does not require
    loading the model runtime. The returned batch is validated immediately.
    """

    import torch

    from .sg_sct_model_v1 import SGSCTBatch

    validate_sg_sct_data_v1(bundle)
    arrays = bundle.arrays

    def tensor(name: str, *, dtype: Any | None = None) -> Any:
        value = torch.from_numpy(arrays[name])
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value.to(device=device) if device is not None else value

    batch = SGSCTBatch(
        node_numeric=tensor("model_node_numeric", dtype=torch.float32),
        node_type=tensor("gs_node_type", dtype=torch.long),
        node_graph=tensor("model_node_graph", dtype=torch.long),
        edge_index=tensor("model_edge_index", dtype=torch.long),
        edge_type=tensor("model_edge_type", dtype=torch.long),
        edge_role=tensor("model_edge_role", dtype=torch.long),
        edge_features=tensor("model_edge_features", dtype=torch.float32),
        symptom_node_mask=tensor("model_symptom_node_mask", dtype=torch.bool),
        candidate_node_mask=tensor("model_candidate_node_mask", dtype=torch.bool),
        gantt_numeric=tensor("model_gantt_numeric", dtype=torch.float32),
        gantt_graph=tensor("model_gantt_graph", dtype=torch.long),
        gantt_operation_node=tensor("model_gantt_operation_node", dtype=torch.long),
        gantt_machine_index=tensor("model_gantt_machine_index", dtype=torch.long),
        gantt_job_index=tensor("model_gantt_job_index", dtype=torch.long),
        gantt_time_index=tensor("model_gantt_time_index", dtype=torch.long),
        trajectory_numeric=tensor("model_trajectory_numeric", dtype=torch.float32),
        trajectory_graph=tensor("model_trajectory_graph", dtype=torch.long),
        trajectory_step=tensor("model_trajectory_step", dtype=torch.long),
        mechanism_values=tensor("model_mechanism_values", dtype=torch.float32),
        mechanism_value_mask=tensor("model_mechanism_value_mask", dtype=torch.bool),
        symptom_block_count=int(arrays["model_symptom_block_count"][0])
        if "model_symptom_block_count" in arrays
        else 0,
        symptom_block_node_index=tensor("model_symptom_block_node_index", dtype=torch.long)
        if "model_symptom_block_node_index" in arrays
        else None,
        node_symptom_block=tensor("model_node_symptom_block", dtype=torch.long)
        if "model_node_symptom_block" in arrays
        else None,
        symptom_block_neighborhood_mask=tensor("model_symptom_block_neighborhood_mask", dtype=torch.bool)
        if "model_symptom_block_neighborhood_mask" in arrays
        else None,
        appearance_type=tensor("model_appearance_type", dtype=torch.long)
        if "model_appearance_type" in arrays
        else None,
        appearance_features=tensor("model_appearance_features", dtype=torch.float32)
        if "model_appearance_features" in arrays
        else None,
    )
    batch.validate()
    return batch


def compile_fjsp_drl_case_v1(
    *,
    instance_path: Path,
    schedule_path: Path,
    appearance_path: Path,
    case_id: str,
    split_id: str | None = None,
    near_critical_threshold: float | None = None,
    near_critical_fraction: float = 0.05,
) -> SGSCTDataV1:
    """Real FJSP-DRL 20x10 adapter entry without LLM/Oracle execution."""

    from .ir_adapters.fjsp_drl import load_fjsp_problem, schedule_from_fjsp_drl_payload

    problem = load_fjsp_problem(instance_path, problem_id=case_id)
    schedule_payload = _json_object(schedule_path)
    schedule = schedule_from_fjsp_drl_payload(problem, schedule_payload)
    appearance = _json_object(appearance_path)
    bundle = compile_sg_sct_input_v1(
        problem,
        schedule,
        appearance,
        case_id=case_id,
        split_id=split_id,
        near_critical_threshold=near_critical_threshold,
        near_critical_fraction=near_critical_fraction,
    )
    manifest = dict(bundle.manifest)
    manifest["source_inputs"] = {
        "instance": {"path": str(instance_path), "sha256": _sha256(instance_path)},
        "schedule": {"path": str(schedule_path), "sha256": _sha256(schedule_path)},
        "appearance": {"path": str(appearance_path), "sha256": _sha256(appearance_path)},
    }
    return SGSCTDataV1(arrays=bundle.arrays, manifest=manifest)
