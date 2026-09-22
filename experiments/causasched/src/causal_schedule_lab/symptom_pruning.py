"""Deterministic Gantt appearance discovery (A1/A2/A3/A4/A6) + relative-specialness pruning.

This module deliberately separates two questions (per ~/Downloads/新表象.md):

* A-rules locate **observable** poor arrangements in an incumbent schedule and
  score each block with an ``AppearanceScore``.  They do not judge cause or
  predict counterfactual rescheduling.
* The pruning layer ranks blocks by a soft ``Priority``:
  ``AppearanceScore · [λ + (1-λ)·PrototypeMatch]`` where ``PrototypeMatch``
  measures how close a block's *relative specialness* (scarcity / coverage /
  load / flexibility / downstream slack) is to its appearance type's "extreme
  prototype".  Retention is Top-k by Priority, not a hard gate.

Critical-path / causal-root attribution remains downstream.  A4's TailImpact
(downstream slack) only judges **current** makespan-propagation relevance; it
is not a causal identification and does not replace black-box perturbation.
``identified`` is never claimed here.

Rules that require setup semantics or a pre-event reference schedule report
``not_applicable`` through the frozen snapshot.

Phase 2.6 (改A6.md): A6 (slow processing mode) is downgraded from a standalone
primary appearance to auxiliary evidence attached to existing blocks (``a6_aux``),
gated by a dual threshold (relative ``Z_rel`` + absolute ``Z_abs``).  Standalone
A6 emission is off by default (``emit_standalone_A6=False``); it is reached only
through an explicit legacy gate or a strict extreme gate.
"""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import math
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .core_validation import validate_schedule
from .ir import Problem, Schedule


VerificationStatus = Literal[
    "verified", "not_applicable", "insufficient_evidence"
]

RULE_CATALOG_VERSION = "2026.08.06-v3"
RULE_SET_VERSION = "appearance-pruning-v3"
DETECTOR_VERSION = "symptom-pruning-detector-v3"
DEFAULT_CALIBRATION_VERSION = "uncalibrated-defaults-v1"

# Core appearance rules in this version.  A5/A7/A8/A9/A10 are intentionally not
# primary appearances (see 新表象.md §2); they remain in the catalog for audit
# but the detector reports them ``not_applicable``.  The canonical taxonomy
# lives in ``appearance_taxonomy`` -- this module is the *detector*, it consumes
# the single source of truth rather than redeclaring it.
from .appearance_taxonomy import (
    ACTIVE_APPEARANCE_IDS as _PRIMARY_RULES,
    DEPRECATED_APPEARANCE_IDS as _DEPRECATED_RULES,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class SymptomThresholds(FrozenModel):
    """Calibratable defaults; they are not claimed as universal constants."""

    # --- A1: machine adjacent gap ---
    a1_gap_ratio: float = Field(default=1.0, ge=0.0)
    # A1 makespan-relevance admission gate (T1-B5-A1-MAKESPAN-RELEVANT-GAP):
    # a gap is admitted as an active A1 block only when the operation AFTER the
    # gap (the "right" op, whose window the gap is part of) has deterministic
    # MakespanRelevance M_j >= this threshold -- i.e. a perturbation at the
    # downstream op can actually propagate to C_max.  Default = tau_m (0.3),
    # the canonical A4 relevance gate (reused -- a gap whose downstream op
    # cannot reach C_max within the absorbable window is not an active A1).
    # Set to 0.0 to restore the OLD pure gap-size admission.  Reuses the
    # canonical _makespan_relevance / tau_m; A1 anchored severity stays the
    # machine gap quantity (A1_value = gap/typical), relevance is admission-only.
    a1_makespan_relevance_min: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- A2/A3: flexible load imbalance ---
    # Source machine load percentile rank above which a window is "high load".
    a2_high_load_percentile: float = Field(default=0.75, ge=0.0, le=1.0)
    # Minimum flexible-load rate Z_flex for a window to qualify.
    a2_min_flex_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # --- A4: job wait with makespan propagation relevance ---
    a4_wait_ratio: float = Field(default=1.0, ge=0.0)
    # MakespanRelevance gate: M_j < tau_m => discard (perturbation cannot reach C_max).
    tau_m: float = Field(default=0.3, ge=0.0, le=1.0)
    # A4Score gate: A4Score(e) < tau_a4 => discard (weak combined signal).
    tau_a4: float = Field(default=0.05, ge=0.0, le=1.0)
    # A4 does not emit one block per job unconditionally: only the K jobs whose
    # core point has the highest A4Score (= W_e * M_j, the wait both large AND
    # truly propagating to C_max) are kept.  The rest get no A4 block, so the
    # M2 operator only reverses from the makespan-critical subset.
    a4_top_k: int = Field(default=10, ge=1)

    # --- A6: slow processing mode (Phase 2.6: auxiliary-only by default) ---
    # A6 is downgraded from a standalone primary appearance to auxiliary
    # evidence (改A6.md §1-§5).  Default: no standalone A6 block; the mode
    # disadvantage is attached to existing blocks as an ``a6_aux`` feature.
    emit_standalone_A6: bool = Field(default=False)
    allow_extreme_A6_standalone: bool = Field(default=False)
    a6_aux_enabled: bool = Field(default=True)
    # Dual threshold: relative disadvantage Z_rel and absolute gap Z_abs.
    tau_a6_rel: float = Field(default=0.40, ge=0.0, le=1.0)
    tau_a6_abs: float = Field(default=0.50, ge=0.0)
    # Extreme standalone gate (only reached when allow_extreme_A6_standalone).
    extreme_a6_rel: float = Field(default=0.60, ge=0.0, le=1.0)
    extreme_a6_abs: float = Field(default=1.0, ge=0.0)
    # Legacy gate (kept for backward compatibility when emit_standalone_A6=True).
    a6_slow_mode_ratio: float = Field(default=0.25, ge=0.0)
    a6_slow_mode_abs: int = Field(default=2, ge=0)

    # --- block merge ---
    merge_jaccard: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- soft pruning: Top-k retention ---
    top_k: int = Field(default=20, ge=1)
    # Priority = AppearanceScore * [lambda + (1-lambda)*PrototypeMatch].
    # Smaller lambda => prototype match dominates; larger => severity dominates.
    lambda_a1: float = Field(default=0.3, ge=0.0, le=1.0)
    lambda_a23: float = Field(default=0.3, ge=0.0, le=1.0)
    lambda_a4: float = Field(default=0.3, ge=0.0, le=1.0)
    lambda_a6: float = Field(default=0.3, ge=0.0, le=1.0)
    # PrototypeMatch weights per dimension (uniform by default).
    proto_weights_a1: tuple[float, ...] = (1.0, 1.0, 1.0)
    proto_weights_a23: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    proto_weights_a4: tuple[float, ...] = (1.0, 1.0, 1.0, 0.0)
    proto_weights_a6: tuple[float, ...] = (1.0, 1.0)


class RuleAvailability(FrozenModel):
    rule_id: str
    status: VerificationStatus
    reason: str


class A2FlexiblePeer(FrozenModel):
    """One mutable peer machine an A2 overloaded source could hand flexible
    operations to (新表象.md §2.2), captured deterministically at detection time.

    Only populated by the A2 rule on its emitted blocks (other rules leave
    ``related`` empty).  This carries *where* the overloaded source's flexible
    load could go, so downstream root-cause / operator selection no longer has to
    re-derive it (it was previously computed and discarded).
    """

    peer_machine: str
    # pool peers that can take, ranked by z_under_k; the operations on the source
    # machine that are eligible to move to this peer (mode-sharing with the peer).
    flexible_operations: tuple[str, ...] = ()
    # 1 - percentile-rank(utilization of peer) within the pool: how idle this peer is.
    z_under_k: float = 0.0
    # Changably flexible busy on this peer's eligible ops / total segment busy.
    z_share: float = 0.0
    duration_ratio: float = 0.0


class AppearanceBlock(FrozenModel):
    block_id: str
    appearance_rules: tuple[str, ...]
    operations: tuple[str, ...]
    jobs: tuple[str, ...]
    machines: tuple[str, ...]
    time_interval: tuple[int, int]
    appearance_values: dict[str, float]
    evidence_ids: tuple[str, ...]
    verification_status: VerificationStatus = "verified"
    # Phase 2.6: A6 mode-duration disadvantage attached as auxiliary evidence
    # (not a standalone primary appearance).  None when no operation in the block
    # shows a mode disadvantage (改A6.md §2/§7).
    a6_aux: dict[str, Any] | None = None
    # A2-only: deterministic roster of *where* the overloaded source's flexible
    # operations could be handed down (mutable low-load peers + the ops that can
    # move).  Default empty so A1/A3/A4/A6 blocks and all frozen serializations
    # are unchanged.  This is a fit-inspection hint, not an identified root cause.
    related: tuple["A2FlexiblePeer", ...] = ()


class RelativeSpecialness(FrozenModel):
    """Relative-specialness features for the prototype-match pruning layer.

    Only the dimensions relevant to a block's appearance type are populated;
    the rest stay at 0.0 and are excluded from the weighted prototype match
    (the denominator drops dimensions with weight 0).  All values in [0, 1].
    """

    # A1 / general.
    scarcity: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    environment_specificity: float = Field(default=0.0, ge=0.0, le=1.0)
    # A2/A3.
    load_rank: float = Field(default=0.0, ge=0.0, le=1.0)
    flex_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    share_score: float = Field(default=0.0, ge=0.0, le=1.0)
    underuse: float = Field(default=0.0, ge=0.0, le=1.0)
    # A4.
    multi_job: float = Field(default=0.0, ge=0.0, le=1.0)
    # A6.
    time_disadvantage: float = Field(default=0.0, ge=0.0, le=1.0)
    flex_op: float = Field(default=0.0, ge=0.0, le=1.0)


class PrunedAppearanceBlock(FrozenModel):
    block: AppearanceBlock
    specialness: RelativeSpecialness = RelativeSpecialness()
    prototype_match: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    h_score: float = 0.0  # alias for priority (downstream sort key)
    priority_tuple: tuple[float, ...] = ()
    keep_or_prune: Literal["keep", "prune"] = "prune"


class SymptomPruningSnapshot(FrozenModel):
    schema_version: str = "appearance-pruning-3.0"
    problem_id: str
    schedule_makespan: int
    thresholds: SymptomThresholds
    blocks: tuple[PrunedAppearanceBlock, ...]
    rule_availability: tuple[RuleAvailability, ...]
    reference_schedule_supplied: bool
    schedule_feasible: bool = True
    validation_errors: tuple[dict[str, Any], ...] = ()
    validation_warnings: tuple[dict[str, Any], ...] = ()
    critical_path_used_for_pruning: bool = False
    rule_catalog_version: str = RULE_CATALOG_VERSION
    rule_set_version: str = RULE_SET_VERSION
    active_rule_versions: dict[str, int] = Field(
        default_factory=lambda: {
            **{rule: 3 for rule in _PRIMARY_RULES},
            **{rule: 0 for rule in _DEPRECATED_RULES},
        }
    )
    detector_version: str = DETECTOR_VERSION
    calibration_version: str = DEFAULT_CALIBRATION_VERSION
    threshold_hash: str = ""
    calibration_hash: str = ""
    program_trace_ids: tuple[str, ...] = ()
    provenance: dict[str, str] = Field(default_factory=dict)

    @property
    def retained(self) -> tuple[PrunedAppearanceBlock, ...]:
        return tuple(item for item in self.blocks if item.keep_or_prune == "keep")


def _canonical_hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snapshot_provenance(
    problem: Problem,
    schedule: Schedule,
    thresholds: SymptomThresholds,
    *,
    calibration_version: str,
    reference_schedule: Schedule | None,
) -> tuple[str, str, dict[str, str]]:
    threshold_hash = _canonical_hash(thresholds)
    calibration_hash = _canonical_hash(
        {
            "calibration_version": calibration_version,
            "threshold_hash": threshold_hash,
        }
    )
    provenance = {
        "problem_hash": _canonical_hash(problem),
        "schedule_hash": _canonical_hash(schedule),
        "threshold_hash": threshold_hash,
        "calibration_hash": calibration_hash,
        "rule_catalog_version": RULE_CATALOG_VERSION,
        "rule_set_version": RULE_SET_VERSION,
        "detector_version": DETECTOR_VERSION,
        "calibration_version": calibration_version,
    }
    if reference_schedule is not None:
        provenance["reference_schedule_hash"] = _canonical_hash(reference_schedule)
    return threshold_hash, calibration_hash, provenance


def _resource_sequences(
    problem: Problem, schedule: Schedule
) -> dict[str, list[tuple[int, int, str]]]:
    mode_map = problem.mode_map()
    result: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for assignment in schedule.assignments:
        selected = mode_map.get(assignment.mode_id)
        if selected is None:
            continue
        for resource_id in selected[1].resources:
            result[resource_id].append(
                (assignment.start, assignment.end, assignment.operation_id)
            )
    for values in result.values():
        values.sort(key=lambda item: (item[0], item[1], item[2]))
    return result


def _assigned_resources(problem: Problem, schedule: Schedule) -> dict[str, tuple[str, ...]]:
    mode_map = problem.mode_map()
    return {
        assignment.operation_id: tuple(mode_map[assignment.mode_id][1].resources)
        for assignment in schedule.assignments
        if assignment.mode_id in mode_map
    }


def _resource_pools(problem: Problem) -> tuple[tuple[str, ...], ...]:
    """Return only defensible comparable pools, never a global-machine fallback.

    Explicit project groups win. Otherwise resources are connected only when
    at least one operation can actually choose between them. This naturally
    yields same-stage pools for HFSP and alternative-machine pools for FJSP,
    while fixed JSP/FSP machines are not falsely compared as substitutes.
    """

    explicit = problem.metadata.get("comparable_resource_groups")
    if explicit:
        known = set(problem.resource_map())
        return tuple(
            tuple(sorted(set(map(str, group)) & known))
            for group in explicit
            if len(set(map(str, group)) & known) >= 2
        )
    adjacency: dict[str, set[str]] = defaultdict(set)
    for operation in problem.operations:
        options = sorted({resource for mode in operation.modes for resource in mode.resources})
        if len(options) < 2:
            continue
        for left in options:
            adjacency[left].update(item for item in options if item != left)
    unseen = set(adjacency)
    groups: list[tuple[str, ...]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for neighbor in sorted(adjacency[current] & unseen):
                unseen.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        if len(component) >= 2:
            groups.append(tuple(sorted(component)))
    return tuple(sorted(groups))


def _availability(problem: Problem, resource_id: str, horizon: int) -> int:
    resource = problem.resource_map()[resource_id]
    if not resource.calendar:
        return max(1, horizon * resource.capacity)
    available = sum(
        max(0, min(right, horizon) - max(left, 0))
        for left, right in resource.calendar
    )
    return max(1, available * resource.capacity)


def _idle_windows(sequence: list[tuple[int, int, str]], horizon: int) -> list[tuple[int, int, str, str]]:
    windows: list[tuple[int, int, str, str]] = []
    for left, right in zip(sequence, sequence[1:]):
        if right[0] > left[1]:
            windows.append((left[1], right[0], left[2], right[2]))
    return windows


def _new_block(
    *,
    rule: str,
    operations: set[str] | tuple[str, ...],
    machines: set[str] | tuple[str, ...],
    interval: tuple[int, int],
    value: float,
    problem: Problem,
    ordinal: int,
    evidence: tuple[str, ...] = (),
    extra_values: dict[str, float] | None = None,
    related: tuple["A2FlexiblePeer", ...] = (),
) -> AppearanceBlock:
    operation_map = problem.operation_map()
    operation_ids = tuple(sorted(set(operations)))
    appearance_values = {rule: float(value)}
    if extra_values:
        appearance_values.update(extra_values)
    return AppearanceBlock(
        block_id=f"{rule}:{ordinal:04d}",
        appearance_rules=(rule,),
        operations=operation_ids,
        jobs=tuple(sorted({operation_map[item].job_id for item in operation_ids})),
        machines=tuple(sorted(set(machines))),
        time_interval=interval,
        appearance_values=appearance_values,
        evidence_ids=tuple(sorted(set(evidence))),
        related=tuple(related),
    )


def _typical_duration_on_machine(
    sequence: list[tuple[int, int, str]],
) -> float:
    durations = [end - start for start, end, _ in sequence]
    return float(median(durations)) if durations else 1.0


def _percentile_rank(value: float, population: list[float]) -> float:
    """Percentile rank in [0, 1]; 0.5 for a singleton population.

    ``PR = below / (n-1)`` so the largest value ranks 1.0 and the smallest 0.0.
    """
    n = len(population)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.5
    below = sum(1 for item in population if item < value)
    return below / (n - 1)


def _makespan_relevance(
    problem: Problem, schedule: Schedule, p_scale: float
) -> dict[str, float]:
    """MakespanRelevance M_j per operation (新表象_1.md §2.3.4).

    Inject a standard time perturbation delta = p_scale at operation o_j and
    propagate it forward over the realized scheduling DAG to a virtual sink at
    C_max: ``d_v = max_{u in Pred(v)} [d_u - g_uv]_+`` where ``g_uv = S_v - C_u``
    is the absorbable gap on each realized constraint edge.  The residual at the
    sink gives ``M_j = d_sink / delta in [0, 1]``.

    This is equivalent to ``M_j = max(0, 1 - D(o_j)/delta)`` where ``D(o_j)`` is
    the shortest absorbable-gap path from o_j to the virtual sink (terminal
    operations connect to the sink with gap ``C_max - C_o``).  Computed in one
    backward shortest-path pass -- O(V + E).  Only the *current* scheduling
    topology is reflected; this is not a causal identification.
    """
    assignment_map = schedule.assignment_map()
    nodes = {op.id for op in problem.operations if op.id in assignment_map}
    if not nodes:
        return {}
    successors: dict[str, set[str]] = {item: set() for item in nodes}
    indegree: dict[str, int] = {item: 0 for item in nodes}
    # Precedence edges (E_J).
    for operation in problem.operations:
        if operation.id not in nodes:
            continue
        for predecessor in operation.predecessors:
            if predecessor in nodes and operation.id not in successors[predecessor]:
                successors[predecessor].add(operation.id)
                indegree[operation.id] += 1
    # Realized machine-sequence edges (E_M).
    for sequence in _resource_sequences(problem, schedule).values():
        for (_, _, left), (_, _, right) in zip(sequence, sequence[1:], strict=False):
            if right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    # Kahn topological order.
    queue = deque(sorted(item for item, degree in indegree.items() if degree == 0))
    topological: list[str] = []
    while queue:
        current = queue.popleft()
        topological.append(current)
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(topological) != len(nodes):
        # Cycle (should not happen for a feasible schedule): no propagation info.
        return {op: 0.0 for op in nodes}
    makespan = schedule.makespan
    completion = {op: assignment_map[op].end for op in nodes}
    # D(v) = shortest absorbable-gap path from v to the virtual sink.
    # Terminal operations (no successor) connect to sink with gap C_max - C_v.
    # Non-terminal: D(v) = min over successors w of (g_{v,w} + D(w)),
    #   g_{v,w} = S_w - C_v >= 0.
    dist: dict[str, float] = {}
    for current in reversed(topological):
        succ = successors[current]
        if not succ:
            # Terminal: direct edge to the virtual sink.
            dist[current] = max(0.0, float(makespan - completion[current]))
        else:
            best = math.inf
            for successor in succ:
                gap = assignment_map[successor].start - completion[current]
                best = min(best, gap + dist.get(successor, math.inf))
            dist[current] = best
    delta = max(p_scale, 1e-6)
    return {
        op: max(0.0, min(1.0, 1.0 - dist[op] / delta))
        for op in nodes
    }


def _flexibility_weight(operation, current_mode) -> float:
    """phi_o: shareable flexible load weight in [0, 1] (新表象.md §2.2.3).

    phi_o = max over k in E_o minus m of min(1, p_om/p_ok).  0 if the operation has no
    alternative machine.  Uses processing-time ratio, not ready-time feasibility.
    """
    alternatives = [mode for mode in operation.modes if mode.id != current_mode.id]
    if not alternatives:
        return 0.0
    p_cur = current_mode.duration
    return max(min(1.0, p_cur / max(mode.duration, 1)) for mode in alternatives)


def _a6_features(
    operation_id: str,
    operation,
    assignments: dict[str, Any],
    mode_map: dict[Any, Any],
    p_scale: float,
) -> dict[str, Any]:
    """A6 auxiliary evidence for one operation (改A6.md §3/§7).

    ``A6Feature(o) = [Z_rel, Z_abs, R_o, Δp]`` with the dual threshold gate
    ``Z_rel>=tau_a6_rel and Z_abs>=tau_a6_abs`` marking ``strong=True``.  This is
    auxiliary explanation only — it never boosts RootScore nor seeds standalone
    A6 in the main pipeline.
    """
    assignment = assignments.get(operation_id)
    if assignment is None:
        return {}
    selected_mode = mode_map[assignment.mode_id][1]
    p_cur = selected_mode.duration or 0.0
    if len(operation.modes) <= 1 or p_cur <= 0:
        return {}
    p_best = min(mode.duration for mode in operation.modes if mode.duration and mode.duration > 0)
    if not p_best:
        return {}
    delta = p_cur - p_best
    if delta <= 0:
        return {}
    z_rel = max(0.0, 1.0 - p_best / p_cur)
    z_abs = delta / (p_scale + 1e-9)
    return {
        "z_time": z_rel,
        "z_rel": round(z_rel, 6),
        "z_abs": round(z_abs, 6),
        "absolute_gap": round(delta, 6),
        "relative_ratio": round(p_cur / p_best, 6),
        "current_mode": assignment.mode_id,
        "best_p": round(p_best, 6),
    }


def _a6_strong(
    aux: dict[str, Any], thresholds: SymptomThresholds, *, extreme: bool = False
) -> bool:
    rel_tau = thresholds.extreme_a6_rel if extreme else thresholds.tau_a6_rel
    abs_tau = thresholds.extreme_a6_abs if extreme else thresholds.tau_a6_abs
    return bool(
        aux.get("z_rel", 0.0) >= rel_tau and aux.get("z_abs", 0.0) >= abs_tau
    )


def _scan_appearances(
    problem: Problem,
    schedule: Schedule,
    thresholds: SymptomThresholds,
    reference_schedule: Schedule | None,
    relevance_cache: dict[str, float],
) -> tuple[list[AppearanceBlock], list[RuleAvailability]]:
    assignments = schedule.assignment_map()
    operations = problem.operation_map()
    mode_map = problem.mode_map()
    resources_for = _assigned_resources(problem, schedule)
    sequences = _resource_sequences(problem, schedule)
    horizon = schedule.makespan
    blocks: list[AppearanceBlock] = []
    availability: list[RuleAvailability] = []
    ordinal: dict[str, int] = defaultdict(int)

    def emit(rule: str, **kwargs: Any) -> None:
        ordinal[rule] += 1
        blocks.append(_new_block(rule=rule, ordinal=ordinal[rule], problem=problem, **kwargs))

    # --- A1: machine adjacent gap (新表象.md §2.1) ---
    # Normalized by the machine's robust typical processing time (median positive duration).
    # T1-B5-A1: admission now also requires the operation AFTER the gap (``right``)
    # to have deterministic MakespanRelevance M_j >= a1_makespan_relevance_min, so
    # a merely-local idle gap that is fully absorbed downstream does not become an
    # active A1 block.  Severity stays the physical gap quantity (ratio); the
    # relevance value is recorded separately (A1_relevance / A1_gap_severity).
    for machine, sequence in sequences.items():
        typical = _typical_duration_on_machine(sequence)
        for start, end, left, right in _idle_windows(sequence, horizon):
            ratio = (end - start) / max(typical, 1.0)
            if ratio < thresholds.a1_gap_ratio:
                continue
            m_right = relevance_cache.get(right, 0.0)
            if m_right < thresholds.a1_makespan_relevance_min:
                continue
            emit(
                "A1", operations={left, right}, machines={machine},
                interval=(start, end), value=ratio,
                evidence=(f"schedule:{left}", f"schedule:{right}"),
                extra_values={
                    "A1_relevance": round(m_right, 6),
                    "A1_gap_severity": round(ratio, 6),
                },
            )
    availability.append(RuleAvailability(rule_id="A1", status="verified", reason="schedule intervals"))

    # --- A2/A3: flexible load distribution imbalance (新表象.md §2.2) ---
    resource_pools = _resource_pools(problem)
    # Per-machine full-horizon utilization for the percentile rank.
    utilization = {
        machine: (sum(end - start for start, end, _ in sequences.get(machine, ()))
                  / max(_availability(problem, machine, horizon), 1))
        for machine in problem.resource_map()
    }
    a23_emitted = False
    for pool in resource_pools:
        if len(pool) < 2:
            continue
        pool_util = [utilization.get(m, 0.0) for m in pool]
        for machine in pool:
            sequence = sequences.get(machine, [])
            if not sequence:
                continue
            z_load = _percentile_rank(utilization.get(machine, 0.0), pool_util)
            if z_load < thresholds.a2_high_load_percentile:
                continue
            # Window = each maximal contiguous processing segment on this machine.
            segments: list[list[tuple[int, int, str]]] = []
            current_seg: list[tuple[int, int, str]] = []
            for item in sequence:
                if current_seg and item[0] > current_seg[-1][1]:
                    segments.append(current_seg)
                    current_seg = []
                current_seg.append(item)
            if current_seg:
                segments.append(current_seg)
            for segment in segments:
                window_start, window_end = segment[0][0], segment[-1][1]
                window_len = max(1, window_end - window_start)
                busy = sum(end - start for start, end, _ in segment)
                rho_m = busy / window_len
                # Flexible load on this window.
                flex_load = 0.0
                for _, _, op_id in segment:
                    op = operations[op_id]
                    cur_mode = mode_map[assignments[op_id].mode_id][1]
                    flex_load += (assignments[op_id].end - assignments[op_id].start) * _flexibility_weight(op, cur_mode)
                z_flex = flex_load / max(busy, 1)
                if z_flex < thresholds.a2_min_flex_rate:
                    continue
                # Underuse of pool peers + share score -> U_m.
                share_terms: list[tuple[float, float]] = []
                for k in pool:
                    if k == machine:
                        continue
                    z_under_k = 1.0 - _percentile_rank(utilization.get(k, 0.0), pool_util)
                    # Share: operations in segment that can run on k.
                    shared_busy = 0.0
                    for _, _, op_id in segment:
                        op = operations[op_id]
                        if any(k in mode.resources for mode in op.modes):
                            cur_mode = mode_map[assignments[op_id].mode_id][1]
                            shared_busy += (assignments[op_id].end - assignments[op_id].start) * min(1.0, cur_mode.duration / max(
                                next((m.duration for m in op.modes if k in m.resources), cur_mode.duration), 1))
                    z_share = shared_busy / max(busy, 1)
                    share_terms.append((k, z_share, z_under_k))
                denom = sum(s for _, s, _ in share_terms)
                u_m = (sum(s * u for _, s, u in share_terms) / denom) if denom > 0 else 0.0
                score = z_load * z_flex * u_m
                if score <= 0:
                    continue
                related = tuple(sorted({k for k, _, u in share_terms if u > 0}))
                # Roster: for each pool peer that can take, list exactly which
                # operations on this overloaded source it could run (mode-shared),
                # plus relative duration penalty so downstream下放 picks削峰 net.
                peer_ops: dict[str, list[str]] = {k: [] for k in related}
                peer_duration_ratio: dict[str, list[float]] = {k: [] for k in related}
                for _, _, op_id in segment:
                    op = operations[op_id]
                    cur_mode = mode_map[assignments[op_id].mode_id][1]
                    cur_dur = float(cur_mode.duration)
                    for mode in op.modes:
                        if not mode.resources:
                            continue
                        for k in related:
                            if k in mode.resources:
                                peer_ops[k].append(op_id)
                                alt_dur = next((m.duration for m in op.modes if m.resources and k in m.resources), None)
                                ratio = min(1.0, cur_dur / max(float(alt_dur), 1)) if alt_dur else 1.0
                                peer_duration_ratio[k].append(float(ratio))
                                break
                related_roster = tuple(
                    A2FlexiblePeer(
                        peer_machine=k,
                        flexible_operations=tuple(sorted(set(peer_ops.get(k, ())))),
                        z_under_k=next((u for m, _, u in share_terms if m == k), 0.0),
                        z_share=next((s for m, s, _ in share_terms if m == k), 0.0),
                        duration_ratio=sum(peer_duration_ratio.get(k, ())) / max(len(peer_duration_ratio.get(k, ())), 1),
                    )
                    for k in related
                    if peer_ops.get(k)  # only peers that actually have flexible ops to take
                )
                emit(
                    "A2", operations={item[2] for item in segment}, machines={machine},
                    interval=(window_start, window_end), value=score,
                    evidence=(f"resource_pool:{','.join(pool)}", f"flex_rate:{z_flex:.3f}"),
                    related=related_roster,
                )
                a23_emitted = True
    pool_status: VerificationStatus = "verified" if resource_pools else "not_applicable"
    pool_reason = "comparable resource pools" if resource_pools else "no explicit or eligibility-derived comparable resource pool"
    availability.extend((
        RuleAvailability(rule_id="A2", status=pool_status, reason=pool_reason),
        RuleAvailability(rule_id="A3", status=pool_status, reason=pool_reason),
    ))

    # --- A4: job wait with makespan propagation relevance (新表象_1.md §2.3) ---
    # raw wait -> WaitSeverity W_e -> MakespanRelevance M_j -> A4Score = W_e * M_j.
    by_job: dict[str, list] = defaultdict(list)
    for operation in problem.operations:
        if operation.id in assignments:
            by_job[operation.job_id].append(operation)
    durations_all = [a.end - a.start for a in schedule.assignments]
    p_scale = float(median(durations_all)) if durations_all else 1.0
    m_relevance = relevance_cache  # dict[op_id -> M_j] from _makespan_relevance
    # a4 edge: (job, pred, succ, wait, W_e, M_j, A4Score)
    a4_edges: list[tuple[str, str, str, int, float, float, float]] = []
    for job_id, job_ops in by_job.items():
        ordered = sorted(job_ops, key=lambda item: item.index)
        for left, right in zip(ordered, ordered[1:]):
            wait = assignments[right.id].start - assignments[left.id].end
            if wait <= 0:
                continue
            # WaitSeverity gate (raw wait must be noticeable vs p_scale).
            if wait / max(p_scale, 1.0) < thresholds.a4_wait_ratio:
                continue
            w_e = wait / (wait + max(p_scale, 1.0))  # in [0,1)
            m_j = m_relevance.get(right.id, 0.0)  # MakespanRelevance(succ)
            if m_j < thresholds.tau_m:
                continue  # perturbation cannot reach C_max -> discard
            a4_score = w_e * m_j
            if a4_score < thresholds.tau_a4:
                continue
            a4_edges.append((job_id, left.id, right.id, wait, w_e, m_j, a4_score))
    # Aggregate effective A4 edges per job.  A4 = the job's inter-operation
    # wait that is NOT buffered away downstream (M_j gate below tau_m already
    # dropped the absorbed ones), i.e. it propagates to C_max.  One block per
    # job anchored at the job's single most critical gap (highest
    # A4Score = W_e * M_j): the {predecessor, successor} op pair whose delay is
    # both large and closest to the makespan.  M2 then reverses from that
    # successor up its machine (L2) and job (L1) predecessors to the true root.
    if a4_edges:
        best_edge: dict[str, tuple] = {}
        for edge in a4_edges:
            job_id = edge[0]
            if job_id not in best_edge or edge[6] > best_edge[job_id][6]:
                best_edge[job_id] = edge
        # Only the jobs whose core point is among the top-K A4Score (severity *
        # makespan-propagation) get an A4 block.  A4 is not "every job gets one":
        # it surfaces just the makespan-critical core points.
        critical = sorted(best_edge.values(), key=lambda e: e[6], reverse=True)[
            : thresholds.a4_top_k
        ]
        for edge in critical:
            left, succ, wait, w_e, m_j, a4_score = edge[1], edge[2], edge[3], edge[4], edge[5], edge[6]
            ops = {left, succ}
            machines = resources_for.get(succ, ())
            emit(
                "A4", operations=ops, machines=machines,
                interval=(assignments[left].end, assignments[succ].start),
                value=float(a4_score),
                evidence=(f"a4_wait:{left}->{succ}:M{m_j:.2f}",),
                extra_values={
                    "A4_mean": float(a4_score),
                    "a4_mj": float(m_j),
                    "a4_wait_n": int(wait),
                },
            )
    availability.append(RuleAvailability(rule_id="A4", status="verified", reason="job wait + makespan propagation relevance"))

    # --- A6: slow processing mode (改A6.md §1-§5, auxiliary-only by default) ---
    # A6 is downgraded from a standalone primary appearance to auxiliary
    # evidence.  Default (emit_standalone_A6=False): no A6 block is emitted; the
    # mode disadvantage is attached to existing blocks as an ``a6_aux`` feature.
    flexible = any(len(operations[op_id].modes) > 1 for op_id in assignments)
    positive_durations = [
        (mode_map[assignments[op_id].mode_id][1].duration or 0.0)
        for op_id in assignments
        if mode_map.get(assignments[op_id].mode_id) is not None
    ]
    pos_durations = [d for d in positive_durations if d > 0]
    p_scale_a6 = float(median(pos_durations)) if pos_durations else 1.0
    a6_aux_map: dict[str, dict[str, Any]] = {}
    for operation_id in assignments:
        operation = operations[operation_id]
        aux = _a6_features(operation_id, operation, assignments, mode_map, p_scale_a6)
        if aux:
            a6_aux_map[operation_id] = aux

    if thresholds.emit_standalone_A6:
        # Legacy gate: reproduce the old single-threshold standalone emission.
        for operation_id, assignment in assignments.items():
            operation = operations[operation_id]
            if len(operation.modes) <= 1:
                continue
            selected_mode = mode_map[assignment.mode_id][1]
            p_cur = selected_mode.duration or 0.0
            p_best = min(mode.duration for mode in operation.modes if mode.duration and mode.duration > 0)
            if not p_best:
                continue
            delta = p_cur - p_best
            ratio = p_cur / max(p_best, 1)
            if delta >= thresholds.a6_slow_mode_abs and ratio >= 1.0 + thresholds.a6_slow_mode_ratio:
                z_time = 1.0 - p_best / max(p_cur, 1)
                emit(
                    "A6", operations={operation_id},
                    machines=set(selected_mode.resources),
                    interval=(assignment.start, assignment.end), value=z_time,
                    evidence=(f"mode:{assignment.mode_id}", f"best:p{p_best}"),
                )
    elif thresholds.allow_extreme_A6_standalone:
        # Strict experimental standalone mode (改A6.md §5): extreme dual gate
        # plus at least one importance support (propagation-relevant operation).
        for operation_id, aux in a6_aux_map.items():
            if not _a6_strong(aux, thresholds, extreme=True):
                continue
            if relevance_cache.get(operation_id, 0.0) < thresholds.tau_m:
                continue
            assignment = assignments[operation_id]
            selected_mode = mode_map[assignment.mode_id][1]
            emit(
                "A6", operations={operation_id},
                machines=set(selected_mode.resources),
                interval=(assignment.start, assignment.end), value=aux["z_rel"],
                evidence=(f"mode:{assignment.mode_id}", "extreme_a6_standalone"),
            )
    availability.append(RuleAvailability(
        rule_id="A6", status="verified" if flexible else "not_applicable",
        reason="eligible processing modes" if flexible else "no alternative machines",
    ))

    # --- Deprecated appearances (A5/A7/A8/A9/A10): not primary in this version ---
    for rule in _DEPRECATED_RULES:
        availability.append(RuleAvailability(
            rule_id=rule, status="not_applicable",
            reason="not a primary appearance in appearance-pruning-v3",
        ))

    # Attach A6 auxiliary evidence to blocks whose operations show a mode
    # disadvantage (改A6.md §2/§7).  Auxiliary feature only — never a standalone
    # primary appearance, never a RootScore input.
    if thresholds.a6_aux_enabled and a6_aux_map:
        rebuilt: list[AppearanceBlock] = []
        for block in blocks:
            attached = {op: a6_aux_map[op] for op in block.operations if op in a6_aux_map}
            if not attached:
                rebuilt.append(block)
                continue
            best = max(attached.values(), key=lambda a: a.get("z_rel", 0.0))
            aux = {
                "enabled": True,
                "z_time": best["z_time"],
                "z_rel": best["z_rel"],
                "z_abs": best["z_abs"],
                "absolute_gap": best["absolute_gap"],
                "relative_ratio": best["relative_ratio"],
                "strong": _a6_strong(best, thresholds),
                "operations": tuple(sorted(attached)),
            }
            rebuilt.append(block.model_copy(update={"a6_aux": aux}))
        blocks = rebuilt
    return blocks, availability


def _merge_blocks(blocks: list[AppearanceBlock], threshold: float, problem: Problem) -> list[AppearanceBlock]:
    pending = list(sorted(blocks, key=lambda item: item.block_id))
    merged: list[AppearanceBlock] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            for index, candidate in enumerate(pending):
                if not (set(current.machines) & set(candidate.machines)):
                    continue
                union = set(current.operations) | set(candidate.operations)
                jaccard = len(set(current.operations) & set(candidate.operations)) / max(len(union), 1)
                if jaccard < threshold:
                    continue
                pending.pop(index)
                current = AppearanceBlock(
                    block_id=current.block_id,
                    appearance_rules=tuple(sorted(set(current.appearance_rules) | set(candidate.appearance_rules))),
                    operations=tuple(sorted(union)),
                    jobs=tuple(sorted(set(current.jobs) | set(candidate.jobs))),
                    machines=tuple(sorted(set(current.machines) | set(candidate.machines))),
                    time_interval=(min(current.time_interval[0], candidate.time_interval[0]), max(current.time_interval[1], candidate.time_interval[1])),
                    appearance_values={**current.appearance_values, **candidate.appearance_values},
                    evidence_ids=tuple(sorted(set(current.evidence_ids) | set(candidate.evidence_ids))),
                    a6_aux=current.a6_aux or candidate.a6_aux,
                )
                changed = True
                break
        merged.append(current)
    return merged


def _scarcity_of(
    problem: Problem, operation_ids, assignment_map
) -> tuple[float, float]:
    """Z_scar (irreplaceability) and total realized duration for an op set.

    Z_scar = sum(l_o / |E_o|) / sum(l_o); |E_o| = distinct candidate resources
    across all modes of operation o.  Returns (z_scar, total_duration); both 0
    when the set has no positive-duration operation.
    """
    operations = problem.operation_map()
    weighted = 0.0
    total = 0.0
    for op_id in operation_ids:
        operation = operations.get(op_id)
        if operation is None or op_id not in assignment_map:
            continue
        duration = assignment_map[op_id].end - assignment_map[op_id].start
        if duration <= 0:
            continue
        candidate_resources = {r for mode in operation.modes for r in mode.resources}
        if not candidate_resources:
            continue
        weighted += duration / len(candidate_resources)
        total += duration
    if total <= 0:
        return 0.0, 0.0
    return weighted / total, total


def _coverage_of(machine_id: str, problem: Problem, schedule: Schedule) -> float:
    """Z_cover: fraction of the project's jobs that pass through `machine_id`.

    "relevant jobs" = all jobs in the instance; a job passes through m if any of
    its operations is assigned to m.  High => m is a common route position.
    """
    assignment_map = schedule.assignment_map()
    operations = problem.operation_map()
    jobs_on_machine: set[str] = set()
    for op_id, resources in _assigned_resources(problem, schedule).items():
        if machine_id in resources and op_id in assignment_map:
            jobs_on_machine.add(operations[op_id].job_id)
    total_jobs = len({op.job_id for op in problem.operations})
    return len(jobs_on_machine) / max(1, total_jobs)


def _block_primary_machine(
    block: AppearanceBlock, problem: Problem, schedule: Schedule
) -> str | None:
    """The block machine carrying the most block operations (tie -> sorted first)."""
    resources_for = _assigned_resources(problem, schedule)
    counts: dict[str, int] = defaultdict(int)
    for op_id in block.operations:
        for machine in resources_for.get(op_id, ()):
            counts[machine] += 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda m: counts[m])


def _pool_utilization(
    problem: Problem, schedule: Schedule, horizon: int
) -> tuple[dict[str, tuple[str, ...]], dict[str, float]]:
    """Comparable resource pools + per-machine full-horizon utilization rate."""
    pools = _resource_pools(problem)
    sequences = _resource_sequences(problem, schedule)
    utilization = {
        machine: (sum(end - start for start, end, _ in sequences.get(machine, ()))
                  / max(_availability(problem, machine, horizon), 1))
        for machine in problem.resource_map()
    }
    return pools, utilization


def _pool_of(
    machine_id: str, pools: tuple[tuple[str, ...], ...]
) -> tuple[str, ...] | None:
    for pool in pools:
        if machine_id in pool:
            return pool
    return None


def _relative_specialness(
    problem: Problem,
    schedule: Schedule,
    block: AppearanceBlock,
    primary_rule: str,
    ctx: dict[str, Any],
    thresholds: SymptomThresholds,
) -> RelativeSpecialness:
    """Compute the relative-specialness feature vector for one block.

    Only the dimensions relevant to `primary_rule` are populated; the rest stay
    at 0.0 and carry zero prototype weight (excluded from the match denominator).
    This is a *relative* measurement against the current instance, not a causal
    identification (``identified=false`` always).
    """
    assignment_map = ctx["assignment_map"]
    sequences = ctx["sequences"]
    utilization = ctx["utilization"]
    pools = ctx["pools"]
    resources_for = ctx["resources_for"]
    fields: dict[str, float] = {}

    if primary_rule == "A1":
        machine = _block_primary_machine(block, problem, schedule)
        if machine is not None:
            z_scar, _ = _scarcity_of(problem, block.operations, assignment_map)
            fields["scarcity"] = z_scar
            fields["coverage"] = _coverage_of(machine, problem, schedule)
        # environment_specificity stays 0 (no explicit project env constraints modeled).

    elif primary_rule in ("A2", "A3"):
        machine = _block_primary_machine(block, problem, schedule)
        if machine is not None:
            pool = _pool_of(machine, pools) or (machine,)
            pool_util = [utilization.get(k, 0.0) for k in pool]
            fields["load_rank"] = _percentile_rank(utilization.get(machine, 0.0), pool_util)
            # Window flexible load rate over the block's realized interval.
            window_start, window_end = block.time_interval
            busy = 0.0
            flex_load = 0.0
            operations = problem.operation_map()
            mode_map = problem.mode_map()
            for start, end, op_id in sequences.get(machine, ()):
                overlap = max(0, min(end, window_end) - max(start, window_start))
                if overlap <= 0:
                    continue
                busy += overlap
                operation = operations[op_id]
                cur_mode = mode_map[assignment_map[op_id].mode_id][1]
                flex_load += overlap * _flexibility_weight(operation, cur_mode)
            fields["flex_rate"] = flex_load / max(busy, 1.0)
            # Pick the best related peer k* by Z_share; report its share + underuse.
            best_share = 0.0
            best_under = 0.0
            for k in pool:
                if k == machine:
                    continue
                shared_busy = 0.0
                for start, end, op_id in sequences.get(machine, ()):
                    overlap = max(0, min(end, window_end) - max(start, window_start))
                    if overlap <= 0:
                        continue
                    operation = operations[op_id]
                    if any(k in mode.resources for mode in operation.modes):
                        cur_mode = mode_map[assignment_map[op_id].mode_id][1]
                        alt = next((m.duration for m in operation.modes if k in m.resources), cur_mode.duration)
                        shared_busy += overlap * min(1.0, cur_mode.duration / max(alt, 1))
                z_share = shared_busy / max(busy, 1.0)
                if z_share > best_share:
                    best_share = z_share
                    best_under = 1.0 - _percentile_rank(utilization.get(k, 0.0), pool_util)
            fields["share_score"] = best_share
            fields["underuse"] = best_under

    elif primary_rule == "A4":
        # z_A4 = (Z_multi-job, Z_succ-scar, Z_succ-cover, Z_env) (新表象_1.md §4.3).
        # m* = primary successor resource; scarcity/coverage measured on ops on m*.
        job_counts = ctx["a4_job_counts"]
        n_jobs = len(block.jobs)
        fields["multi_job"] = _percentile_rank(float(n_jobs), job_counts) if job_counts else 0.0
        machine = _block_primary_machine(block, problem, schedule)
        if machine is not None:
            ops_on_machine = [
                op_id for op_id in block.operations
                if machine in resources_for.get(op_id, ())
            ] or list(block.operations)
            z_scar, _ = _scarcity_of(problem, ops_on_machine, assignment_map)
            fields["scarcity"] = z_scar
            fields["coverage"] = _coverage_of(machine, problem, schedule)
        # environment_specificity stays 0 (no explicit project env constraint modeled).

    elif primary_rule == "A6":
        # Single operation mode劣势.
        op_id = block.operations[0] if block.operations else None
        if op_id and op_id in assignment_map:
            operation = problem.operation_map()[op_id]
            fields["flex_op"] = 0.0 if len(operation.modes) <= 1 else 1.0 - 1.0 / len(operation.modes)
            fields["time_disadvantage"] = float(block.appearance_values.get("A6", 0.0))

    return RelativeSpecialness(**fields)


# Per-rule prototype dimensions (e_d = 1 for all -> match = weighted mean of z_d).
# A1 drops the environment dimension when no project env constraint is modeled.
_PROTO_DIMS: dict[str, tuple[str, ...]] = {
    "A1": ("scarcity", "coverage"),
    "A2": ("load_rank", "flex_rate", "share_score", "underuse"),
    "A3": ("load_rank", "flex_rate", "share_score", "underuse"),
    "A4": ("multi_job", "scarcity", "coverage", "environment_specificity"),
    "A6": ("time_disadvantage", "flex_op"),
}
_PROTO_WEIGHTS: dict[str, str] = {
    "A1": "proto_weights_a1",
    "A2": "proto_weights_a23",
    "A3": "proto_weights_a23",
    "A4": "proto_weights_a4",
    "A6": "proto_weights_a6",
}
_PROTO_LAMBDA: dict[str, str] = {
    "A1": "lambda_a1",
    "A2": "lambda_a23",
    "A3": "lambda_a23",
    "A4": "lambda_a4",
    "A6": "lambda_a6",
}


def _prototype_match(
    primary_rule: str, specialness: RelativeSpecialness, thresholds: SymptomThresholds
) -> float:
    dims = _PROTO_DIMS.get(primary_rule)
    if not dims:
        return 0.0
    weights = getattr(thresholds, _PROTO_WEIGHTS[primary_rule])
    numerator = 0.0
    denominator = 0.0
    for index, field in enumerate(dims):
        weight = weights[index] if index < len(weights) else 0.0
        if weight <= 0:
            continue
        value = max(0.0, min(1.0, getattr(specialness, field)))
        numerator += weight * value
        denominator += weight
    return numerator / denominator if denominator > 0 else 0.0


def _primary_rule(block: AppearanceBlock) -> str:
    """The dominant appearance rule of a (possibly merged) block by raw value."""
    if not block.appearance_values:
        return block.appearance_rules[0] if block.appearance_rules else "A1"
    # Only consider actual rule-name keys (A1, A2, A3, A4, A6, A23 ...), not
    # extra_values such as "A4_mean" which share the dict but are not rules.
    # Rule names never contain "_"; extra measurement keys always do.
    candidates = {k: v for k, v in block.appearance_values.items() if "_" not in k}
    if not candidates:
        return block.appearance_rules[0] if block.appearance_rules else "A1"
    return max(candidates, key=lambda key: candidates[key])


def _appearance_score(
    primary_rule: str, raw: float, populations: dict[str, list[float]]
) -> float:
    """Normalize the raw appearance value into [0, 1].

    A1 (gap ratio) is unbounded -> a midpoint percentile rank across same-type
    blocks ``(below + 0.5) / n`` so the smallest block still carries a positive
    score and a retained block is never zeroed purely by being the least severe
    of its type.  A2/A3, A4 (1-Π(1-A4Score)) and A6 are already bounded in [0, 1].
    """
    if primary_rule == "A1":
        population = populations.get(primary_rule, [raw])
        n = len(population)
        if n == 0:
            return 0.0
        below = sum(1 for item in population if item < raw)
        return (below + 0.5) / n
    return max(0.0, min(1.0, raw))


def diagnose_and_prune(
    problem: Problem,
    schedule: Schedule,
    *,
    thresholds: SymptomThresholds | None = None,
    reference_schedule: Schedule | None = None,
    calibration_version: str = DEFAULT_CALIBRATION_VERSION,
) -> SymptomPruningSnapshot:
    """Run the appearance-pruning-v3 snapshot.

    Pipeline: A-layer discovers appearance blocks (A1/A2/A3/A4/A6) with raw
    AppearanceScore; the pruning layer computes relative specialness + an
    extreme-prototype match per block, combines them into a soft Priority =
    AppearanceScore * [lambda + (1-lambda) * PrototypeMatch], and retains the
    Top-k blocks by Priority.  This is a frozen non-parametric prior
    (``identified=false``): it ranks which appearances are worth investigating,
    it does not identify root causes.
    """

    selected_thresholds = thresholds or SymptomThresholds()
    threshold_hash, calibration_hash, provenance = _snapshot_provenance(
        problem,
        schedule,
        selected_thresholds,
        calibration_version=calibration_version,
        reference_schedule=reference_schedule,
    )
    validation = validate_schedule(problem, schedule)
    if not validation.feasible:
        return SymptomPruningSnapshot(
            problem_id=problem.id,
            schedule_makespan=schedule.makespan,
            thresholds=selected_thresholds,
            blocks=(),
            rule_availability=tuple(
                RuleAvailability(
                    rule_id=f"A{index}",
                    status="insufficient_evidence",
                    reason="invalid schedule; diagnosis stopped at feasibility gate",
                )
                for index in range(1, 11)
            ),
            reference_schedule_supplied=reference_schedule is not None,
            schedule_feasible=False,
            validation_errors=tuple(
                {
                    "code": item.code,
                    "message": item.message,
                    "entities": item.entities,
                }
                for item in validation.errors
            ),
            validation_warnings=tuple(
                {
                    "code": item.code,
                    "message": item.message,
                    "entities": item.entities,
                }
                for item in validation.warnings
            ),
            calibration_version=calibration_version,
            threshold_hash=threshold_hash,
            calibration_hash=calibration_hash,
            program_trace_ids=("core_validation.validate_schedule:v1",),
            provenance=provenance,
        )

    # MakespanRelevance M_j powers A4's propagation filter (新表象_1.md §2.3/§4.3):
    # perturbation injected at successor o_j propagates to the virtual sink C_max.
    durations_all = [a.end - a.start for a in schedule.assignments]
    p_scale = float(median(durations_all)) if durations_all else 1.0
    relevance_cache = _makespan_relevance(problem, schedule, p_scale)
    raw_blocks, availability = _scan_appearances(
        problem, schedule, selected_thresholds, reference_schedule, relevance_cache
    )
    merged = _merge_blocks(raw_blocks, selected_thresholds.merge_jaccard, problem)

    # Per-type raw populations for AppearanceScore normalization, and A4 job
    # counts for the multi-job percentile rank.
    populations: dict[str, list[float]] = defaultdict(list)
    for block in merged:
        primary = _primary_rule(block)
        raw = float(block.appearance_values.get(primary, 0.0))
        populations[primary].append(raw)
    a4_job_counts = [
        float(len(block.jobs))
        for block in merged
        if _primary_rule(block) == "A4"
    ]

    horizon = schedule.makespan
    pools, utilization = _pool_utilization(problem, schedule, horizon)
    ctx = {
        "assignment_map": schedule.assignment_map(),
        "sequences": _resource_sequences(problem, schedule),
        "utilization": utilization,
        "pools": pools,
        "resources_for": _assigned_resources(problem, schedule),
        "a4_job_counts": a4_job_counts,
    }

    # First pass: compute Priority per block (frozen instances built after the
    # Top-k gate so keep_or_prune is set at construction time).
    scored: list[dict[str, Any]] = []
    for block in merged:
        primary = _primary_rule(block)
        raw = float(block.appearance_values.get(primary, 0.0))
        appearance_score = _appearance_score(primary, raw, populations)
        specialness = _relative_specialness(
            problem, schedule, block, primary, ctx, selected_thresholds
        )
        prototype_match = _prototype_match(primary, specialness, selected_thresholds)
        lam = float(getattr(selected_thresholds, _PROTO_LAMBDA[primary]))
        priority = appearance_score * (lam + (1.0 - lam) * prototype_match)
        block_start = float(block.time_interval[0]) if block.time_interval else 0.0
        scored.append({
            "block": block,
            "rule": primary,
            "specialness": specialness,
            "prototype_match": prototype_match,
            "priority": priority,
            "block_start": block_start,
        })

    # Retention by Priority, applied per appearance rule: each type keeps its own
    # top-k, so an abundant rule (e.g. A4's per-job buffer blocks) never crowds
    # the tight root-cause anchors of another rule out of tne table.  Within a
    # rule, ties break earliest-block-first (front-to-back).
    scored.sort(key=lambda item: (-item["priority"], item["block_start"], item["block"].block_id))
    top_k = max(0, selected_thresholds.top_k)
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for item in scored:
        by_rule.setdefault(item["rule"], []).append(item)
    kept_ids: set[str] = set()
    for rule_items in by_rule.values():
        kept_ids.update(item["block"].block_id for item in rule_items[:top_k])

    # Final ordering: kept blocks front-to-back, then pruned by priority.
    kept = sorted(
        (item for item in scored if item["block"].block_id in kept_ids),
        key=lambda item: (item["block_start"], -item["priority"], item["block"].block_id),
    )
    pruned = sorted(
        (item for item in scored if item["block"].block_id not in kept_ids),
        key=lambda item: (-item["priority"], item["block_start"], item["block"].block_id),
    )
    ordered = [
        PrunedAppearanceBlock(
            block=item["block"],
            specialness=item["specialness"],
            prototype_match=item["prototype_match"],
            priority=item["priority"],
            h_score=item["priority"],
            priority_tuple=(item["priority"], -item["block_start"]),
            keep_or_prune="keep",
        )
        for item in kept
    ] + [
        PrunedAppearanceBlock(
            block=item["block"],
            specialness=item["specialness"],
            prototype_match=item["prototype_match"],
            priority=item["priority"],
            h_score=item["priority"],
            priority_tuple=(item["priority"], -item["block_start"]),
            keep_or_prune="prune",
        )
        for item in pruned
    ]

    return SymptomPruningSnapshot(
        problem_id=problem.id,
        schedule_makespan=schedule.makespan,
        thresholds=selected_thresholds,
        blocks=tuple(ordered),
        rule_availability=tuple(availability),
        reference_schedule_supplied=reference_schedule is not None,
        schedule_feasible=True,
        validation_warnings=tuple(
            {
                "code": item.code,
                "message": item.message,
                "entities": item.entities,
            }
            for item in validation.warnings
        ),
        calibration_version=calibration_version,
        threshold_hash=threshold_hash,
        calibration_hash=calibration_hash,
        program_trace_ids=(
            "core_validation.validate_schedule:v1",
            "symptom_pruning.scan_appearances:v3",
            "symptom_pruning.merge_blocks:v1",
            "symptom_pruning.relative_specialness:v3",
            "symptom_pruning.prototype_match:v3",
            "symptom_pruning.topk_rank:v3",
        ),
        provenance=provenance,
    )


__all__ = [
    "AppearanceBlock",
    "PrunedAppearanceBlock",
    "RelativeSpecialness",
    "RuleAvailability",
    "SymptomPruningSnapshot",
    "SymptomThresholds",
    "DEFAULT_CALIBRATION_VERSION",
    "DETECTOR_VERSION",
    "RULE_CATALOG_VERSION",
    "RULE_SET_VERSION",
    "diagnose_and_prune",
]
