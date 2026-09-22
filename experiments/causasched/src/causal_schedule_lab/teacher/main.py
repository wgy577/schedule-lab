"""Phase-1 data-production main flow (教师模型.md §26, Phase 1 = no LLM).

For each retained appearance block: build ``G_prop``, seed the appearance,
run Max-Plus reverse sensitivity, generate decision atoms, rank by
TeacherScore, run the rule probe (atomic Solver CE) over a teacher +
exploration candidate mix, search the joint root set, and emit one Gold sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ir import Problem, Schedule
from ..objective import evaluate_objective
from ..symptom_pruning import (
    PrunedAppearanceBlock,
    SymptomPruningSnapshot,
    diagnose_and_prune,
)
from ..validation import schedule_hash
from .atom_generator import DecisionAtom, generate_root_atoms
from .atomic_counterfactual_executor import AtomicCounterfactualExecutor
from .dataset_writer import build_gold_sample
from .exploration_sampler import sample_exploration_atoms
from .propagation_graph import build_realized_propagation_graph
from .reverse_sensitivity import appearance_specific_seed, reverse_sensitivity
from .root_set_search import build_root_set
from .solver_probe_runner import (
    AtomicProbeResult,
    run_atomic_probe,
    run_joint_probe,
)
from .teacher_ranker import RankedAtom, rank_atoms

GOLD_SET_VERSION = "acct-gold-v1"


@dataclass(frozen=True)
class GoldSampleMeta:
    instance_id: str
    state_id: str
    block_id: str
    appearance_type: str
    appearance_before: float
    delta: float
    teacher_ranked: tuple[RankedAtom, ...]
    atomic: tuple[AtomicProbeResult, ...]
    root_set: dict[str, Any]


def _appearance_target_id(block: PrunedAppearanceBlock) -> dict[str, Any]:
    return {
        "block_id": block.block.block_id,
        "rules": list(block.block.appearance_rules),
        "operations": list(block.block.operations),
        "time_interval": list(block.block.time_interval),
    }


def generate_gold_set(
    problem: Problem,
    schedule: Schedule,
    *,
    snapshot: SymptomPruningSnapshot | None = None,
    executor: AtomicCounterfactualExecutor | None = None,
    blocks: tuple[PrunedAppearanceBlock, ...] | None = None,
    max_blocks: int = 8,
    top_atoms: int = 12,
    teacher_budget: int = 8,
    exploration_budget: int = 2,
    max_root_size: int = 3,
    beam_width: int = 4,
    tau_root: float = 0.70,
    seed: int = 0,
    deterministic_time: float = 1.0,  # retained for backward-compat; unused (no solver)
    stability_weight: int = 1,        # retained for backward-compat; unused (no solver)
) -> tuple[list[dict[str, Any]], tuple[GoldSampleMeta, ...]]:
    """Produce Phase-1 Gold samples for the retained appearance blocks.

    Returns ``(samples, metas)`` where each ``samples`` entry is a §21 Gold
    dict ready for ``write_gold_samples``.
    """
    if snapshot is None:
        snapshot = diagnose_and_prune(problem, schedule)
    kept = blocks if blocks is not None else snapshot.retained
    if executor is None:
        executor = AtomicCounterfactualExecutor()

    graph = build_realized_propagation_graph(problem, schedule)
    state_id = schedule_hash(schedule)

    samples: list[dict[str, Any]] = []
    metas: list[GoldSampleMeta] = []

    for block in kept[:max_blocks]:
        sample = _one_block(
            problem, schedule, snapshot, block, graph, executor,
            state_id=state_id,
            top_atoms=top_atoms,
            teacher_budget=teacher_budget,
            exploration_budget=exploration_budget,
            max_root_size=max_root_size,
            beam_width=beam_width,
            tau_root=tau_root,
            seed=seed,
            deterministic_time=deterministic_time,
            stability_weight=stability_weight,
        )
        if sample is None:
            continue
        sample_dict, meta = sample
        samples.append(sample_dict)
        metas.append(meta)

    return samples, tuple(metas)


def _one_block(
    problem: Problem,
    schedule: Schedule,
    snapshot: SymptomPruningSnapshot,
    block: PrunedAppearanceBlock,
    graph,
    executor: AtomicCounterfactualExecutor,
    *,
    state_id: str,
    top_atoms: int,
    teacher_budget: int,
    exploration_budget: int,
    max_root_size: int,
    beam_width: int,
    tau_root: float,
    seed: int,
    deterministic_time: float,
    stability_weight: int,
) -> tuple[dict[str, Any], GoldSampleMeta] | None:
    appearance = block.block
    appearance_type = appearance.appearance_rules[0] if appearance.appearance_rules else "UNKNOWN"
    appearance_before = _appearance_magnitude(appearance)
    if appearance_before <= 0:
        return None
    target_operations = tuple(appearance.operations)

    seeds = appearance_specific_seed(target_operations, appearance_type)
    R = reverse_sensitivity(graph, seeds)

    atoms = generate_root_atoms(problem, schedule, graph, R, top_k=top_atoms)
    if not atoms:
        return None
    ranked = rank_atoms(atoms, R, problem, schedule)

    exclude = frozenset(item.atom.atom_id for item in ranked)
    exploration = sample_exploration_atoms(
        atoms, budget=exploration_budget, exclude_atom_ids=exclude
    )
    mix = ranked + tuple(
        RankedAtom(atom=a, relevance=a.relevance, intervention_prior=0.0, teacher_score=0.0)
        for a in exploration
    )

    atomic: list[AtomicProbeResult] = []
    teacher_records: list[dict[str, Any]] = []
    for ranked_atom in mix:
        result = run_atomic_probe(
            problem, schedule, ranked_atom.atom,
            appearance_type=appearance_type,
            appearance_before=appearance_before,
            target_operations=target_operations,
            executor=executor,
        )
        atomic.append(result)
        teacher_records.append({
            "atom_id": ranked_atom.atom.atom_id,
            "propagation_relevance": ranked_atom.relevance,
            "intervention_prior": ranked_atom.intervention_prior,
            "teacher_score": ranked_atom.teacher_score,
            "source": ranked_atom.atom.source,
        })

    # Atomic CE aggregation: keep every probe, aggregate by atom (max).  Only
    # valid labels (fail-closed appearance recompute) contribute a CE.
    ce_by_atom: dict[str, float] = {}
    for result in atomic:
        if result.label_valid:
            ce_by_atom[result.atom_id] = max(ce_by_atom.get(result.atom_id, 0.0), result.ce)
    top_by_ce = tuple(
        atom for atom in ranked
        if atom.atom.atom_id in ce_by_atom
    )
    top_by_ce = tuple(sorted(top_by_ce, key=lambda a: -ce_by_atom[a.atom.atom_id]))[:6]

    def joint_probe(atoms_tuple: tuple[DecisionAtom, ...]):
        result = run_joint_probe(
            problem, schedule, atoms_tuple,
            appearance_type=appearance_type,
            appearance_before=appearance_before,
            target_operations=target_operations,
            executor=executor,
        )
        return result.ce, result.feasible

    root_set_atoms = tuple(a.atom for a in top_by_ce)
    root_set = build_root_set(
        root_set_atoms,
        ce_by_atom,
        joint_probe,
        max_size=max_root_size,
        beam_width=beam_width,
        tau_root=tau_root,
    )

    sample = build_gold_sample(
        instance_id=problem.id,
        state_id=state_id,
        appearance_type=appearance_type,
        appearance_target=_appearance_target_id(block),
        appearance_before=appearance_before,
        delta=graph.delta,
        teacher=teacher_records,
        atomic_probes=[r.as_record() for r in atomic],
        root_set={
            "atoms": list(root_set.atom_ids),
            "ce_joint": root_set.ce,
            "size": root_set.size,
        },
        seed=seed,
    )
    meta = GoldSampleMeta(
        instance_id=problem.id,
        state_id=state_id,
        block_id=block.block.block_id,
        appearance_type=appearance_type,
        appearance_before=appearance_before,
        delta=graph.delta,
        teacher_ranked=ranked,
        atomic=tuple(atomic),
        root_set=root_set.as_record() if hasattr(root_set, "as_record") else {
            "atoms": list(root_set.atom_ids), "ce_joint": root_set.ce, "size": root_set.size,
        },
    )
    return sample, meta


def _appearance_magnitude(appearance) -> float:
    from ..sg_sct_causal_probe import appearance_magnitude

    return appearance_magnitude(appearance)