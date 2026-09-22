"""Phase 2.13 atom/operator mapping and optimization-utility contracts.

This module deliberately separates three meanings:

* an ACCT ``DecisionAtom`` is a causal intervention candidate;
* an ``OperatorParameterCandidate`` is the executable production action;
* ``DeterministicOperatorExecutor`` measures atom-conditioned repair utility.

No function in this module changes Teacher scores or produces causal labels.
``identified`` therefore remains false throughout the Phase 2.13 artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from ...objective import evaluate_objective
from ...operator_execution_v1 import build_operator_execution_plan
from ...operator_registry_v1 import (
    REGISTRY_VERSION,
    generate_operator_candidates,
    operator_masks_from_candidates,
)
from ...training_v1 import (
    OperatorParameterCandidate,
    OperatorPolicyState,
    candidate_set_hash,
)
from ...validation import schedule_hash
from ..atom_generator import DecisionAtom, atom_to_candidate

SCHEMA_VERSION = "phase2-13-teacher-utility-1.0"
MAPPING_VERSION = "atom-operator-mapping-1.0"
EXECUTION_STATUSES = frozenset(
    {"success", "infeasible", "mapping_incomplete", "executor_error"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _machine_of(problem, schedule, operation_id: str) -> str | None:
    assignment = schedule.assignment_map().get(operation_id)
    if assignment is None:
        return None
    mode = problem.mode_map()[assignment.mode_id][1]
    return str(mode.resources[0]) if mode.resources else None


def _canonical_action(
    atom: DecisionAtom,
    candidate: OperatorParameterCandidate,
    problem,
    schedule,
) -> dict[str, Any]:
    parameters = dict(candidate.parameters)
    action: dict[str, Any] = {
        "atom_type": atom.atom_type,
        "operation_id": atom.operation,
        "operator_id": candidate.operator_id,
        "parameters": parameters,
    }
    if atom.atom_type == "routing":
        action.update(
            {
                "source_machine": _machine_of(problem, schedule, atom.operation),
                "target_mode_id": parameters.get("mode_id"),
                "target_resource_ids": list(parameters.get("target_resource_ids") or ()),
            }
        )
    elif atom.atom_type == "sequencing":
        if candidate.operator_id == "adjacent_resource_swap":
            action["exact_edit"] = {
                "kind": "swap",
                "left": parameters.get("left_operation_id"),
                "right": parameters.get("right_operation_id"),
            }
        elif candidate.operator_id == "resource_sequence_insertion":
            action["exact_edit"] = {
                "kind": "insert",
                "resource_id": parameters.get("resource_id"),
                "operation_id": parameters.get("operation_id"),
                "position": parameters.get("position"),
                "predecessor_id": parameters.get("predecessor_id"),
                "successor_id": parameters.get("successor_id"),
            }
        elif candidate.operator_id == "critical_block_resequence":
            action["exact_edit"] = {
                "kind": "critical_block_resequence",
                "resource_id": parameters.get("resource_id"),
                "operation_ids": list(parameters.get("operation_ids") or ()),
            }
    return action


@dataclass(frozen=True)
class AtomOperatorMapping:
    instance_uid: str
    schedule_id: str
    schedule_fingerprint: str
    appearance_id: str
    legacy_atom_id: str
    canonical_atom_id: str | None
    atom_type: str
    operator_id: str | None
    candidate_id: str | None
    operator_parameters: dict[str, Any]
    source_machine: str | None
    target_resource_ids: tuple[str, ...]
    enforced_orderings: tuple[tuple[str, str], ...]
    forced_mode_ids: dict[str, str]
    released_operation_count: int
    mapping_status: str
    missing_fields: tuple[str, ...]
    notes: tuple[str, ...]
    mapping_version: str = MAPPING_VERSION
    identified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_operator_state(problem, schedule, atom: DecisionAtom, candidates) -> OperatorPolicyState:
    """Build the frozen production-executor state for one reconstructed atom."""

    candidates = tuple(candidates)
    masks = operator_masks_from_candidates(candidates)
    state = schedule_hash(schedule)
    return OperatorPolicyState(
        instance_id=problem.id,
        base_instance_id=str(problem.metadata.get("base_instance_id", problem.id)),
        family=problem.kind,
        environment=problem.environment,
        state_id=state,
        schedule_hash=state,
        graph_hash="phase2-13-mapping",
        surface_block_ids=(),
        root_cause_id=f"phase2-13:{atom.atom_id}",
        root_cause_node_ids=atom.operations,
        causal_path_edge_ids=(),
        semantic_operator_mask=masks["semantic"],
        local_operator_mask=masks["local"],
        cause_operator_mask=masks["cause"],
        final_operator_mask=masks["final"],
        parameter_candidates=candidates,
        registry_version=REGISTRY_VERSION,
        candidate_set_hash=candidate_set_hash(candidates),
        current_objective=evaluate_objective(problem, schedule, baseline=schedule).values,
    )


def map_atom_to_operator(
    *,
    problem,
    schedule,
    atom: DecisionAtom,
    instance_uid: str,
    schedule_id: str,
    schedule_fingerprint: str,
    appearance_id: str,
    maximum_per_operator: int = 8,
    neighborhood_radius: int = 0,
    reassignment_release_radius: int = 2,
) -> tuple[AtomOperatorMapping, OperatorParameterCandidate | None, OperatorPolicyState | None]:
    """Reconstruct and audit the exact production candidate for an old atom.

    The legacy atom id is never parsed as executable truth.  The stored
    ``probe_operator_id`` and canonical parameters reconstructed from the same
    schedule are matched against the finite registry.  Ambiguous or stale
    atoms fail closed.
    """

    candidate = atom_to_candidate(
        atom,
        problem,
        schedule,
        maximum_per_operator=maximum_per_operator,
        neighborhood_radius=neighborhood_radius,
    )
    missing: list[str] = []
    notes: list[str] = []
    if not atom.probe_operator_id:
        missing.append("probe_operator_id")
    if not atom.probe_parameters:
        missing.append("probe_parameters")
    if candidate is None:
        missing.append("registry_candidate")
        mapping = AtomOperatorMapping(
            instance_uid=instance_uid,
            schedule_id=schedule_id,
            schedule_fingerprint=schedule_fingerprint,
            appearance_id=appearance_id,
            legacy_atom_id=atom.atom_id,
            canonical_atom_id=None,
            atom_type=atom.atom_type,
            operator_id=atom.probe_operator_id or None,
            candidate_id=None,
            operator_parameters=dict(atom.probe_parameters),
            source_machine=_machine_of(problem, schedule, atom.operation),
            target_resource_ids=(),
            enforced_orderings=(),
            forced_mode_ids={},
            released_operation_count=0,
            mapping_status="unsupported" if atom.probe_operator_id else "ambiguous",
            missing_fields=tuple(sorted(set(missing))),
            notes=("legacy atom id is not executable truth",),
        )
        return mapping, None, None

    operation_tuple = atom.operation if isinstance(atom.operation, tuple) else (atom.operation,)
    candidates = generate_operator_candidates(
        problem,
        schedule,
        root_operations=operation_tuple,
        maximum_per_operator=maximum_per_operator,
        neighborhood_radius=neighborhood_radius,
        decision_time=0,
        subject_operations_only=True,
    )
    state = build_operator_state(problem, schedule, atom, candidates)
    try:
        plan = build_operator_execution_plan(
            problem,
            schedule,
            state,
            candidate,
            reassignment_release_radius=reassignment_release_radius,
        )
    except (ValueError, NotImplementedError, KeyError) as error:
        notes.append(f"execution_plan_rejected:{type(error).__name__}:{error}")
        mapping = AtomOperatorMapping(
            instance_uid=instance_uid,
            schedule_id=schedule_id,
            schedule_fingerprint=schedule_fingerprint,
            appearance_id=appearance_id,
            legacy_atom_id=atom.atom_id,
            canonical_atom_id=None,
            atom_type=atom.atom_type,
            operator_id=candidate.operator_id,
            candidate_id=candidate.candidate_id,
            operator_parameters=dict(candidate.parameters),
            source_machine=_machine_of(problem, schedule, atom.operation),
            target_resource_ids=tuple(str(x) for x in candidate.parameters.get("target_resource_ids", ())),
            enforced_orderings=(),
            forced_mode_ids={},
            released_operation_count=0,
            mapping_status="unsupported",
            missing_fields=("execution_plan",),
            notes=tuple(notes),
        )
        return mapping, candidate, state

    action = _canonical_action(atom, candidate, problem, schedule)
    canonical_id = _content_id("atom", action)
    source_machine = _machine_of(problem, schedule, atom.operation)
    if atom.atom_type == "routing" and atom.machine == source_machine:
        notes.append("legacy routing @machine denotes source machine, not target machine")
    if atom.atom_type == "sequencing":
        notes.append("canonical ordering derives from operator parameters, not legacy '<' text")
    mapping = AtomOperatorMapping(
        instance_uid=instance_uid,
        schedule_id=schedule_id,
        schedule_fingerprint=schedule_fingerprint,
        appearance_id=appearance_id,
        legacy_atom_id=atom.atom_id,
        canonical_atom_id=canonical_id,
        atom_type=atom.atom_type,
        operator_id=candidate.operator_id,
        candidate_id=candidate.candidate_id,
        operator_parameters=dict(candidate.parameters),
        source_machine=source_machine,
        target_resource_ids=tuple(str(x) for x in candidate.parameters.get("target_resource_ids", ())),
        enforced_orderings=tuple(tuple(x) for x in plan.enforced_orderings),
        forced_mode_ids=dict(plan.forced_mode_ids),
        released_operation_count=len(plan.released_operations),
        mapping_status="complete",
        missing_fields=(),
        notes=tuple(notes),
    )
    return mapping, candidate, state


def normalize_appearance(value: str) -> str:
    return "A23" if value in {"A2", "A3", "A23"} else value


def select_audit_blocks(
    rows: Iterable[Mapping[str, Any]], *, target_blocks: int, seed: int = 213
) -> list[dict[str, Any]]:
    """Freeze a deterministic, family/instance/appearance/role balanced sample."""

    blocks: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["instance_uid"]), str(row["schedule_id"]), str(row["appearance_id"]))
        blocks.setdefault(
            key,
            {
                "instance_uid": key[0],
                "schedule_id": key[1],
                "appearance_id": key[2],
                "schedule_fingerprint": str(row["schedule_fingerprint"]),
                "benchmark_family": str(row["benchmark_family"]),
                "appearance_type": normalize_appearance(str(row["appearance_type"])),
                "schedule_role": str(row["schedule_role"]),
                "num_jobs": int(row["num_jobs"]),
                "num_machines": int(row["num_machines"]),
            },
        )
    if target_blocks <= 0:
        raise ValueError("target_blocks must be positive")
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks.values():
        by_stratum[(block["benchmark_family"], block["appearance_type"], block["schedule_role"])].append(block)
    for values in by_stratum.values():
        values.sort(key=lambda x: (x["instance_uid"], x["schedule_id"], x["appearance_id"]))
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(by_stratum)
    while len(selected) < min(target_blocks, len(blocks)):
        progressed = False
        for key in keys:
            if by_stratum[key] and len(selected) < target_blocks:
                selected.append(by_stratum[key].pop())
                progressed = True
        if not progressed:
            break
    for index, block in enumerate(selected):
        block["audit_index"] = index
    return selected


@dataclass(frozen=True)
class TopKUtility:
    k: int
    block_count: int
    improvement_rate: float
    no_improve_rate: float
    all_worse_rate: float
    mapping_coverage: float
    executable_coverage: float
    mean_best_gain: float
    median_best_gain: float
    mean_regret: float
    near_oracle_1pct: float
    near_oracle_3pct: float
    near_oracle_5pct: float


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def compute_topk_utility(
    records: Iterable[Mapping[str, Any]], *, rank_field: str, ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[int, TopKUtility]:
    """Compute intent-to-audit utility; failed actions never become gain=0 labels."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        status = str(row.get("execution_status"))
        if status not in EXECUTION_STATUSES:
            raise ValueError(f"unknown execution_status: {status!r}")
        groups[(str(row["instance_uid"]), str(row["schedule_id"]), str(row["appearance_id"]))].append(row)
    result: dict[int, TopKUtility] = {}
    for k in ks:
        improved: list[float] = []
        no_improve: list[float] = []
        all_worse: list[float] = []
        map_cov: list[float] = []
        exec_cov: list[float] = []
        gains: list[float] = []
        regrets: list[float] = []
        near = {0.01: [], 0.03: [], 0.05: []}
        for rows in groups.values():
            ordered = sorted(rows, key=lambda x: (int(x.get(rank_field, 10**9)), str(x["atom_id"])))
            top = ordered[:k]
            mapped = [r for r in top if r.get("mapping_status") == "complete"]
            executed = [r for r in top if r.get("execution_status") == "success"]
            successful_pool = [r for r in rows if r.get("execution_status") == "success"]
            top_gains = [float(r["optimization_gain"]) for r in executed]
            pool_gains = [float(r["optimization_gain"]) for r in successful_pool]
            map_cov.append(len(mapped) / len(top) if top else 0.0)
            exec_cov.append(len(executed) / len(top) if top else 0.0)
            best = max(top_gains) if top_gains else -math.inf
            improved.append(float(best > 0.0))
            no_improve.append(float(best <= 0.0))
            all_worse.append(float(bool(top_gains) and len(top_gains) == len(top) and all(g < 0 for g in top_gains)))
            if top_gains:
                gains.append(best)
            if pool_gains and top_gains:
                regret = max(pool_gains) - best
                regrets.append(regret)
                for epsilon in near:
                    near[epsilon].append(float(regret <= epsilon))
        n = len(groups)
        mean = lambda xs: sum(xs) / len(xs) if xs else math.nan
        result[int(k)] = TopKUtility(
            k=int(k), block_count=n,
            improvement_rate=mean(improved), no_improve_rate=mean(no_improve),
            all_worse_rate=mean(all_worse), mapping_coverage=mean(map_cov),
            executable_coverage=mean(exec_cov), mean_best_gain=mean(gains),
            median_best_gain=_median(gains), mean_regret=mean(regrets),
            near_oracle_1pct=mean(near[0.01]), near_oracle_3pct=mean(near[0.03]),
            near_oracle_5pct=mean(near[0.05]),
        )
    return result


__all__ = [
    "AtomOperatorMapping", "EXECUTION_STATUSES", "MAPPING_VERSION", "SCHEMA_VERSION",
    "TopKUtility", "build_operator_state", "compute_topk_utility", "map_atom_to_operator",
    "normalize_appearance", "select_audit_blocks",
]
