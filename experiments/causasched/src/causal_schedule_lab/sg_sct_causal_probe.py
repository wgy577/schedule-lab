"""Solver counterfactual causal-effect (CE_A) probe generator (架构.md v2 §11/§12/§29, Stage C).

For each kept appearance block ``B_A`` and each candidate root operation ``r``
in its reverse-caused neighbourhood, run atomic probes ``do(r, a)`` through the
real CP-SAT executor, recompute the SAME target appearance on the resulting
schedule, and emit

    CE_A(r, a) = (A_old - A_new) / (A_old + eps)

with ``CE_A(r) = max_a CE_A(r, a)`` when several legal probes exist.  Every
individual probe result is kept (never collapsed into only the max).

Root-truth boundaries (§32) are enforced here by construction:

* ``CE_A`` is a *causal-effect* label on the target appearance, not a makespan
  label.  ``ΔC_max`` is recorded alongside but is never used as the M2 label.
* Mechanism deviation and prior root scores are not used to define ``CE_A``;
  the effect is measured purely from the counterfactual appearance change.

This module is the data-side foundation of the v2 upgrade: it produces the
``node_ce_scores`` / ``edge_ce_scores`` supervision that the new CE heads
regress against.  It does not train anything and does not touch the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ir import Problem, Schedule
from .objective import evaluate_objective
from .operator_registry_v1 import generate_operator_candidates
from .symptom_pruning import (
    AppearanceBlock,
    PrunedAppearanceBlock,
    SymptomPruningSnapshot,
    diagnose_and_prune,
)
from .teacher.atomic_counterfactual_executor import AtomicCounterfactualExecutor
from .teacher.atom_generator import DecisionAtom
from .teacher.counterfactual_label import causal_effect, makespan_gain as _normalized_makespan_gain
from .teacher.solver_probe_runner import recompute_appearance_atomic
from .validation import schedule_hash

PROBE_VERSION = "atomic-counterfactual-ce-probe-2.0"

# Logical boundary: this is a causal-effect label, not a makespan label.
CAUSAL_LABEL_KIND = "appearance_causal_effect"

# Phase 2.7 boundary (code-level, not convention): every CE label that feeds M2
# root supervision MUST originate from the AtomicCounterfactualExecutor (fixed-
# decision DAG replay, no solver).  A solver/repair executor releases job tails
# and globally reoptimizes mode+order, so its label is CE_A(r + global reopt),
# not CE_A(r) -- feeding it to M2 pollutes root-causal supervision.  These
# constants are stamped onto every probe dataset/record and ENFORCED at the
# single consumption chokepoint (build_ce_node_targets), so a non-atomic dataset
# cannot silently become M2 supervision.
ATOMIC_INTERVENTION_SEMANTICS = "atomic"
ATOMIC_EXECUTOR_NAME = "AtomicCounterfactualExecutor"
# Phase 2.10C §18: CE magnitude is the de-saturated continuous causal
# magnitude (D/M split).  Old saturated-CE datasets (A1 clamped, A4 1-prod)
# must not feed M2.
MEASUREMENT_SEMANTICS = "continuous_causal_magnitude"
APPEARANCE_MEASUREMENT_VERSION = "phase2_10b_desaturated_v1"


def build_ce_node_targets(
    bundle: Any, probe_dataset: Mapping[str, Any]
) -> dict[str, object]:
    """Map solver CE_A probe labels onto the bundle's node ordering.

    Node indices for operations are their position in the manifest's
    ``operation_ids`` (operations occupy node indices ``0..N_ops-1``).  Each
    ``node_ce`` row is keyed by ``root_operation`` (an operation id); the
    aggregated ``node_ce_score`` becomes that node's CE target.  Un-probed
    nodes are masked out (no fabricated supervision).  Edge CE labels are not
    yet emitted by the probe generator, so ``ce_edge_*`` stay all-masked until
    §29 edge probes land.

    **Phase 2.7 boundary guard**: this is the single chokepoint where probe
    CE labels become M2 supervision tensors.  It REFUSES any dataset whose
    ``intervention_semantics`` is not ``"atomic"`` (missing field = legacy
    solver-generated dataset).  This makes the atomic-executor boundary a
    code-level assertion rather than a convention: a non-atomic (solver/repair)
    CE dataset cannot feed M2 root-causal supervision.
    """
    semantics = probe_dataset.get("intervention_semantics")
    if semantics != ATOMIC_INTERVENTION_SEMANTICS:
        raise ValueError(
            "M2 CE supervision requires atomic-source labels "
            f"(intervention_semantics={ATOMIC_INTERVENTION_SEMANTICS!r}); "
            f"got {semantics!r}. Solver/repair executors yield "
            "CE_A(r + global reopt) and must not feed M2. Regenerate the "
            "probe dataset via generate_causal_probe_dataset with an "
            "AtomicCounterfactualExecutor (Phase 2.7 §13)."
        )
    # Phase 2.10C §18: also refuse legacy saturated-CE datasets.
    measurement = probe_dataset.get("measurement_semantics")
    if measurement != MEASUREMENT_SEMANTICS:
        raise ValueError(
            "M2 CE supervision requires de-saturated continuous causal "
            f"magnitude (measurement_semantics={MEASUREMENT_SEMANTICS!r}); "
            f"got {measurement!r}. Legacy saturated CE (A1 clamped / A4 "
            "1-prod) hides A1/A4 signal (Phase 2.10A) and must not feed M2. "
            "Regenerate with appearance_magnitude / appearance_detection_score "
            "split (Phase 2.10B/2.10C)."
        )
    operation_ids = bundle.manifest["id_spaces"]["operation_ids"]
    op_to_index = {op_id: index for index, op_id in enumerate(operation_ids)}
    node_count = int(bundle.arrays["gs_node_type"].shape[0])
    edge_count = int(bundle.arrays["model_edge_index"].shape[1])
    target = np.zeros(node_count, dtype=np.float32)
    mask = np.zeros(node_count, dtype=np.bool_)
    for row in probe_dataset.get("node_ce", ()):
        op_id = str(row["root_operation"])
        if op_id not in op_to_index:
            continue
        index = op_to_index[op_id]
        target[index] = float(row["node_ce_score"])
        mask[index] = True
    return {
        "ce_node_target": target,
        "ce_node_mask": mask,
        "ce_edge_target": np.zeros(edge_count, dtype=np.float32),
        "ce_edge_mask": np.zeros(edge_count, dtype=np.bool_),
    }


def primary_rule(block: AppearanceBlock) -> str:
    """The block's primary appearance rule (A1/A2/A3/A4/A6)."""
    return block.appearance_rules[0] if block.appearance_rules else "UNKNOWN"


def appearance_magnitude(block: AppearanceBlock) -> float:
    """A scalar appearance strength for the block's primary rule (CE magnitude).

    Phase 2.10B: **de-saturated**.  Previously every rule was clamped to
    ``[0, 1]``; that pinned A1's gap ratio (unbounded, often > 1.0) at the
    ceiling and zeroed its CE -- the Phase 2.10A finding.  Now:

    * A1's gap ratio is returned **unclamped** (only floored at 0).  CE is a
      relative change, so an unbounded magnitude is fine and the clamp was
      pure signal loss.
    * A4 prefers the non-saturating ``A4_mean`` (mean per-edge A4Score) when
      the block stores it.  The ``1 - prod(1 - A4Score)`` block score
      saturates at 1.0 once many edges aggregate and is kept only as the
      pruning / sorting score.
    * A2 / A3 / A6 raw values are already bounded in ``[0, 1]`` by construction.
    """
    rule = primary_rule(block)
    if rule == "A4":
        mean_val = block.appearance_values.get("A4_mean")
        if mean_val is not None:
            return max(0.0, float(mean_val))
    raw = float(block.appearance_values.get(rule, 0.0))
    return max(0.0, raw)


def appearance_detection_score(block: AppearanceBlock) -> float:
    """D_A: the detection / pruning score (Phase 2.10C §2-§6).

    This is the score used to *find and filter* symptoms.  It may clamp,
    saturate, or use noisy-OR / 1-prod -- that is fine for detection.  It
    MUST NOT be used as the CE magnitude (use :func:`appearance_magnitude`).

    * A1: ``min(1, gap/typical)`` -- clamped detection ratio (§4).
    * A4: ``1 - prod(1 - A4Score)`` -- noisy-OR over edges (§5).
    * A2/A3/A6: raw value (already bounded; D == M for now, §6).
    """
    rule = primary_rule(block)
    raw = float(block.appearance_values.get(rule, 0.0))
    if rule == "A1":
        return max(0.0, min(1.0, raw))
    return max(0.0, raw)


def find_matching_block(
    snapshot: SymptomPruningSnapshot,
    primary: str,
    operations: tuple[str, ...],
    *,
    min_jaccard: float = 0.3,
) -> PrunedAppearanceBlock | None:
    """Locate the 'same' appearance block after an intervention.

    Block ids are ordinal (e.g. ``A4:0001``) and shift after re-scheduling, so
    the same block is matched by maximum operation-set Jaccard overlap among
    blocks of the same primary rule.  If no same-rule block has meaningful
    overlap, the appearance is treated as eliminated (``None`` -> A_new = 0).
    """
    ops = set(operations)
    best: PrunedAppearanceBlock | None = None
    best_score = 0.0
    for candidate in snapshot.blocks:
        if primary_rule(candidate.block) != primary:
            continue
        other = set(candidate.block.operations)
        if not ops and not other:
            continue
        denom = len(ops | other)
        if denom == 0:
            continue
        jac = len(ops & other) / denom
        if jac > best_score:
            best_score = jac
            best = candidate
    if best is None or best_score < min_jaccard:
        return None
    return best


@dataclass(frozen=True)
class ProbeResult:
    """One atomic probe ``do(r, a)`` with its counterfactual appearance effect."""

    state_id: str
    appearance_type: str
    symptom_id: str
    root_operation: str
    probe_type: str
    probe_parameters: dict[str, Any]
    appearance_before: float
    appearance_after: float
    causal_effect: float
    label_valid: bool
    makespan_before: int
    makespan_after: int | None
    makespan_gain: float
    feasible: bool
    classification: str
    solver_status: str
    candidate_id: str
    executor_version: str
    probe_version: str = PROBE_VERSION

    def as_record(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "appearance": {
                "type": self.appearance_type,
                "block_id": self.symptom_id,
                "score_before": self.appearance_before,
            },
            "candidate_root": {
                "node_ids": [self.root_operation],
                "edge_ids": [],
            },
            "probe": {
                "type": self.probe_type,
                "operator_id": self.probe_type,
                "parameters": self.probe_parameters,
            },
            "counterfactual": {
                "appearance_score_after": self.appearance_after,
                "causal_effect": self.causal_effect,
                "label_valid": self.label_valid,
                "makespan_before": self.makespan_before,
                "makespan_after": self.makespan_after,
                "makespan_gain": self.makespan_gain,
            },
            "feasible": self.feasible,
            "classification": self.classification,
            "solver_status": self.solver_status,
            "label_kind": CAUSAL_LABEL_KIND,
            "probe_version": self.probe_version,
            "intervention_semantics": ATOMIC_INTERVENTION_SEMANTICS,
            "executor": ATOMIC_EXECUTOR_NAME,
            "measurement_semantics": MEASUREMENT_SEMANTICS,
            "appearance_measurement_version": APPEARANCE_MEASUREMENT_VERSION,
        }


@dataclass(frozen=True)
class NodeCE:
    """Aggregated node causal effect for one root operation on one appearance."""

    state_id: str
    appearance_type: str
    symptom_id: str
    root_operation: str
    node_ce_score: float
    best_probe_type: str
    probe_count: int
    feasible_probe_count: int = 0
    mean_effect: float = 0.0
    positive_probe_ratio: float = 0.0
    details: tuple[ProbeResult, ...] = field(default_factory=tuple)


def _best_effect(probes: list[ProbeResult], appearance_before: float) -> float:
    """CE_A(r) = max over valid atomic probes (§11.1).

    Only probes with a *valid* CE label (fail-closed appearance recompute)
    contribute.  Empirically the max over a large probe set saturates near 1.0
    (a single appearance-eliminating probe dominates), so the dataset also
    exposes the mean and positive-probe ratio.  ``aggregation`` selects which is
    the canonical ``node_ce_score``; every per-probe record is always kept.
    """
    effects = [p.causal_effect for p in probes if p.label_valid]
    if not effects:
        return 0.0
    return max(effects)


def _aggregated_effect(probes: list[ProbeResult], aggregation: str) -> float:
    effects = sorted(p.causal_effect for p in probes if p.label_valid)
    if not effects:
        return 0.0
    if aggregation == "max":
        return effects[-1]
    if aggregation == "mean":
        return sum(effects) / len(effects)
    if aggregation == "median":
        n = len(effects)
        return effects[n // 2] if n % 2 else (effects[n // 2 - 1] + effects[n // 2]) / 2.0
    if aggregation == "positive_ratio":
        return sum(1 for e in effects if e > 1e-6) / len(effects)
    raise ValueError(f"unknown CE aggregation: {aggregation!r}")


_ROUTING_OPS = {"machine_reassignment", "stage_machine_reassignment"}


def _candidate_to_atom(root_operation: str, candidate) -> DecisionAtom:
    """Wrap a finite candidate as a DecisionAtom for the atomic executor."""
    atom_type = "routing" if candidate.operator_id in _ROUTING_OPS else "sequencing"
    params = dict(candidate.parameters)
    partner = (
        params.get("left_operation_id") or params.get("right_operation_id")
        or params.get("predecessor_id") or params.get("successor_id")
    )
    return DecisionAtom(
        atom_id=f"{candidate.candidate_id}",
        atom_type=atom_type,
        operation=root_operation,
        machine=params.get("resource_id") or (params.get("target_resource_ids") or [None])[0],
        partner_operation=str(partner) if partner is not None else None,
        probe_operator_id=candidate.operator_id,
        probe_parameters=params,
    )


def run_root_probes(
    problem: Problem,
    schedule: Schedule,
    root_operation: str,
    symptom_id: str,
    appearance_type: str,
    appearance_before: float,
    target_operations: tuple[str, ...],
    *,
    executor: AtomicCounterfactualExecutor,
    decision_time: int = 0,
    maximum_per_operator: int = 64,
    neighborhood_radius: int = 1,
) -> list[ProbeResult]:
    """Run every legal atomic probe targeting ``root_operation`` and return them.

    Each candidate is compiled to an exact intervention, realised by fixed-decision
    DAG replay (no solver), and the target appearance is recomputed fail-closed.
    A probe is ``feasible`` if the replay produced a valid schedule; a probe is
    ``label_valid`` only if the appearance recompute confirmed the target block
    (otherwise no CE label is produced, §11).
    """
    candidates = generate_operator_candidates(
        problem,
        schedule,
        root_operations=(root_operation,),
        decision_time=decision_time,
        maximum_per_operator=maximum_per_operator,
        neighborhood_radius=neighborhood_radius,
    )
    legal = tuple(c for c in candidates if c.legal and c.operator_id != "stop")
    if not legal:
        return []
    before = evaluate_objective(problem, schedule, baseline=schedule)
    makespan_before = int(before.values[0])
    results: list[ProbeResult] = []
    for candidate in legal:
        atom = _candidate_to_atom(root_operation, candidate)
        executed = executor.execute_atom(problem, schedule, atom)
        feasible = executed.feasible
        makespan_after = int(executed.schedule.makespan) if (feasible and executed.schedule is not None) else None
        appearance_after = 0.0
        causal = 0.0
        label_valid = False
        if feasible and executed.schedule is not None:
            recompute = recompute_appearance_atomic(
                problem, executed.schedule, appearance_type, target_operations
            )
            if recompute.valid and recompute.score is not None:
                appearance_after = recompute.score
                causal = causal_effect(appearance_before, appearance_after)
                label_valid = True
        results.append(
            ProbeResult(
                state_id=schedule_hash(schedule),
                appearance_type=appearance_type,
                symptom_id=symptom_id,
                root_operation=root_operation,
                probe_type=candidate.operator_id,
                probe_parameters=dict(candidate.parameters),
                appearance_before=appearance_before,
                appearance_after=appearance_after,
                causal_effect=causal,
                label_valid=label_valid,
                makespan_before=makespan_before,
                makespan_after=makespan_after,
                makespan_gain=_normalized_makespan_gain(makespan_before, makespan_after),
                feasible=feasible,
                classification=executed.report.classification,
                solver_status="ATOMIC_REPLAY" if feasible else "INFEASIBLE",
                candidate_id=candidate.candidate_id,
                executor_version=executed.executor_version,
            )
        )
    return results


def generate_causal_probe_dataset(
    problem: Problem,
    schedule: Schedule,
    snapshot: SymptomPruningSnapshot,
    *,
    executor: AtomicCounterfactualExecutor,
    roots_per_block: int = 3,
    blocks: tuple[PrunedAppearanceBlock, ...] | None = None,
    maximum_per_operator: int = 64,
    neighborhood_radius: int = 1,
    ce_aggregation: str = "max",
) -> dict[str, Any]:
    """Generate the full causal-probe dataset for the retained appearance blocks.

    Returns a JSON-serializable dict with per-node CE aggregation and the raw
    per-probe records (both are kept, per §11.1).  Only ``label_valid`` probes
    contribute to the aggregated ``node_ce_score``.
    """
    kept_blocks = blocks if blocks is not None else snapshot.retained
    records: list[dict[str, Any]] = []
    node_ce_rows: list[NodeCE] = []
    for block in kept_blocks:
        appearance_type = primary_rule(block.block)
        appearance_before = appearance_magnitude(block.block)
        symptom_id = block.block.block_id
        target_operations = block.block.operations
        # Candidate roots: the block's own operations (front-to-back, earliest
        # block first) plus, in a later stage, the reverse-caused neighbourhood.
        root_ops = tuple(block.block.operations)[:roots_per_block]
        for root_op in root_ops:
            probes = run_root_probes(
                problem,
                schedule,
                root_op,
                symptom_id,
                appearance_type,
                appearance_before,
                target_operations,
                executor=executor,
                maximum_per_operator=maximum_per_operator,
                neighborhood_radius=neighborhood_radius,
            )
            for probe in probes:
                records.append(probe.as_record())
            if probes:
                valid = [p for p in probes if p.label_valid]
                effects = [p.causal_effect for p in valid]
                node_ce_rows.append(
                    NodeCE(
                        state_id=schedule_hash(schedule),
                        appearance_type=appearance_type,
                        symptom_id=symptom_id,
                        root_operation=root_op,
                        node_ce_score=_aggregated_effect(list(probes), ce_aggregation),
                        best_probe_type=(max(valid, key=lambda p: p.causal_effect).probe_type if valid else ""),
                        probe_count=len(probes),
                        feasible_probe_count=len([p for p in probes if p.feasible]),
                        mean_effect=(sum(effects) / len(effects)) if effects else 0.0,
                        positive_probe_ratio=(sum(1 for e in effects if e > 1e-6) / len(effects))
                        if effects else 0.0,
                        details=tuple(probes),
                    )
                )
    return {
        "schema_version": "causal-probe-dataset-1.0",
        "probe_version": PROBE_VERSION,
        "intervention_semantics": ATOMIC_INTERVENTION_SEMANTICS,
        "executor": ATOMIC_EXECUTOR_NAME,
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "appearance_measurement_version": APPEARANCE_MEASUREMENT_VERSION,
        "label_valid_required": True,
        "problem_id": problem.id,
        "state_id": schedule_hash(schedule),
        "makespan": int(schedule.makespan),
        "ce_aggregation": ce_aggregation,
        "node_ce": [
            {
                "state_id": row.state_id,
                "appearance_type": row.appearance_type,
                "symptom_id": row.symptom_id,
                "root_operation": row.root_operation,
                "node_ce_score": row.node_ce_score,
                "best_probe_type": row.best_probe_type,
                "probe_count": row.probe_count,
                "feasible_probe_count": row.feasible_probe_count,
                "mean_effect": row.mean_effect,
                "positive_probe_ratio": row.positive_probe_ratio,
            }
            for row in node_ce_rows
        ],
        "edge_ce": [],
        "probes": records,
    }


def main() -> None:
    """CLI entry: build a CE probe dataset for one FJS file and dump JSON."""
    import argparse
    import os
    from .io import load_schedule

    parser = argparse.ArgumentParser(description="Generate solver CE probe labels")
    parser.add_argument("--fjs", required=True, help="path to .fjs instance")
    parser.add_argument("--schedule-json", help="path to existing schedule JSON (optional)")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--roots-per-block", type=int, default=3)
    parser.add_argument("--max-per-operator", type=int, default=64)
    parser.add_argument("--ce-aggregation", type=str, default="max",
                        choices=["max", "mean", "median", "positive_ratio"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from .ir_adapters.fjsp_drl import load_fjsp_problem
    from .io import load_schedule
    from .solvers.dispatching import solve_dispatching

    problem = load_fjsp_problem(args.fjs)
    if args.schedule_json and os.path.exists(args.schedule_json):
        schedule = load_schedule(args.schedule_json)
        if schedule.problem_id != problem.id:
            raise ValueError("schedule-json problem_id does not match the .fjs instance")
    else:
        schedule = solve_dispatching(problem)
    snapshot = diagnose_and_prune(problem, schedule)
    executor = AtomicCounterfactualExecutor()
    dataset = generate_causal_probe_dataset(
        problem,
        schedule,
        snapshot,
        executor=executor,
        roots_per_block=args.roots_per_block,
        maximum_per_operator=args.max_per_operator,
        ce_aggregation=args.ce_aggregation,
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}: {len(dataset['node_ce'])} node-CE rows, {len(dataset['probes'])} probe records")


if __name__ == "__main__":
    main()