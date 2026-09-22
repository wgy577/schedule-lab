"""Phase-2 teacher-candidate quality evaluation (教师模型.md §29/§30).

For each retained appearance block, run the shared candidate pipeline and
measure how well each candidate source ranks the *measured* high-CE root atoms:

* High-CE Recall@K:  ``#{high-CE roots in source TopK} / #{all high-CE roots}``
* BestCE@K:          ``max_{r in TopK} CE_A(r)``
* Solver Budget Efficiency: ``#high-CE probes / #all solver probes``
* Root Set Discovery: ``CE_A(B_R)`` per source under the same joint budget.

All sources probe the *same shared atom set* (one Solver budget), so the
efficiency denominator is identical and Recall@K is directly comparable.  The
ground-truth high-CE set comes only from measured Solver CE, never from the
teacher prior (`identified=false`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..ir import Problem, Schedule
from ..symptom_pruning import PrunedAppearanceBlock, SymptomPruningSnapshot
from .atom_generator import DecisionAtom, generate_root_atoms
from .atomic_counterfactual_executor import AtomicCounterfactualExecutor
from .baselines import dense_page_rank_relevance, random_relevance
from .propagation_graph import RealizedPropagationGraph, build_realized_propagation_graph
from .reverse_sensitivity import appearance_specific_seed, reverse_sensitivity
from .root_set_search import build_root_set
from .solver_probe_runner import run_atomic_probe, run_joint_probe
from .teacher_ranker import RankedAtom, rank_atoms

TAU_CE = 0.30
DEFAULT_KS = (1, 3, 5, 10)

# A relevance source: (name, fn(problem, schedule, graph) -> dict[str,float]).
RelevanceFn = Callable[[Problem, Schedule, RealizedPropagationGraph], dict[str, float]]


@dataclass(frozen=True)
class SourceEval:
    source: str
    top_k: dict[int, tuple[str, ...]]  # K -> ranked atom_ids
    recall_at_k: dict[int, float]
    best_ce_at_k: dict[int, float]
    root_set_ce: float
    root_set_size: int


@dataclass
class BlockEval:
    instance_id: str
    state_id: str
    block_id: str
    appearance_type: str
    appearance_before: float
    n_probed: int
    n_high_ce: int
    high_ce_atoms: tuple[str, ...]
    efficiency: float
    sources: list[SourceEval] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "appearance_type": self.appearance_type,
            "appearance_before": self.appearance_before,
            "n_probed": self.n_probed,
            "n_high_ce": self.n_high_ce,
            "high_ce_atoms": list(self.high_ce_atoms),
            "efficiency": self.efficiency,
            "sources": [
                {
                    "source": s.source,
                    "recall_at_k": {str(k): v for k, v in s.recall_at_k.items()},
                    "best_ce_at_k": {str(k): v for k, v in s.best_ce_at_k.items()},
                    "root_set_ce": s.root_set_ce,
                    "root_set_size": s.root_set_size,
                }
                for s in self.sources
            ],
        }


def _rank_atoms_by_relevance(
    atoms: tuple[DecisionAtom, ...],
    R: dict[str, float],
) -> tuple[DecisionAtom, ...]:
    return tuple(
        sorted(atoms, key=lambda a: (-max(R.get(op, 0.0) for op in a.operations), a.atom_id))
    )


def _rank_by_teacher(atoms: tuple[DecisionAtom, ...], R, problem, schedule) -> tuple[DecisionAtom, ...]:
    ranked = rank_atoms(atoms, R, problem, schedule)
    return tuple(item.atom for item in ranked)


def _recall_and_best(
    ranked: tuple[DecisionAtom, ...],
    high_ce: frozenset[str],
    ce_by_atom: dict[str, float],
    ks: tuple[int, ...],
) -> tuple[dict[int, float], dict[int, float]]:
    positions = {atom.atom_id: i for i, atom in enumerate(ranked)}
    recall: dict[int, float] = {}
    best_ce: dict[int, float] = {}
    for k in ks:
        top = ranked[:k]
        ids = {a.atom_id for a in top}
        if high_ce:
            recall[k] = len(ids & high_ce) / len(high_ce)
        else:
            recall[k] = 0.0
        best_ce[k] = max((ce_by_atom.get(a.atom_id, 0.0) for a in top), default=0.0)
    return recall, best_ce


def evaluate_block_recall(
    problem: Problem,
    schedule: Schedule,
    snapshot: SymptomPruningSnapshot,
    block: PrunedAppearanceBlock,
    *,
    executor: AtomicCounterfactualExecutor,
    graph: RealizedPropagationGraph | None = None,
    tau_ce: float = TAU_CE,
    ks: tuple[int, ...] = DEFAULT_KS,
    top_atoms: int = 16,
    max_root_size: int = 3,
    beam_width: int = 3,
    seed: int = 0,
    deterministic_time: float = 1.0,  # retained for backward-compat; unused (no solver)
    stability_weight: int = 1,        # retained for backward-compat; unused (no solver)
) -> BlockEval:
    appearance = block.block
    appearance_type = appearance.appearance_rules[0] if appearance.appearance_rules else "UNKNOWN"
    from ..sg_sct_causal_probe import appearance_magnitude

    appearance_before = appearance_magnitude(appearance)
    if appearance_before <= 0:
        raise ValueError("block has zero appearance magnitude; cannot evaluate")
    if graph is None:
        graph = build_realized_propagation_graph(problem, schedule)
    target_operations = tuple(appearance.operations)

    # Teacher R (Max-Plus) drives atom generation (shared candidate universe).
    seeds = appearance_specific_seed(target_operations, appearance_type)
    R_teacher = reverse_sensitivity(graph, seeds)
    atoms = generate_root_atoms(problem, schedule, graph, R_teacher, top_k=top_atoms)
    if not atoms:
        raise ValueError("no root atoms generated for block")

    # Source relevance orderings.
    sources: list[tuple[str, tuple[DecisionAtom, ...]]] = [
        ("teacher", _rank_by_teacher(atoms, R_teacher, problem, schedule)),
        ("max_plus", _rank_atoms_by_relevance(atoms, R_teacher)),
        ("page_rank", _rank_atoms_by_relevance(atoms, dense_page_rank_relevance(problem, schedule, graph))),
        ("random", _rank_atoms_by_relevance(atoms, random_relevance(graph, seed=seed))),
    ]

    # Shared atomic probe budget: probe every atom across all sources once.
    ce_by_atom: dict[str, float] = {}
    probed = 0
    for atom in atoms:
        result = run_atomic_probe(
            problem, schedule, atom,
            appearance_type=appearance_type,
            appearance_before=appearance_before,
            target_operations=target_operations,
            executor=executor,
        )
        probed += 1
        if result.label_valid:
            ce_by_atom[atom.atom_id] = max(ce_by_atom.get(atom.atom_id, 0.0), result.ce)

    high_ce = frozenset(aid for aid, ce in ce_by_atom.items() if ce >= tau_ce)
    efficiency = len(high_ce) / probed if probed else 0.0

    def joint_probe(atoms_tuple: tuple[DecisionAtom, ...]):
        result = run_joint_probe(
            problem, schedule, atoms_tuple,
            appearance_type=appearance_type,
            appearance_before=appearance_before,
            target_operations=target_operations,
            executor=executor,
        )
        return result.ce, result.feasible

    source_evals: list[SourceEval] = []
    for name, ranked in sources:
        recall, best_ce = _recall_and_best(ranked, high_ce, ce_by_atom, ks)
        root_set = build_root_set(
            ranked[:6], ce_by_atom, joint_probe,
            max_size=max_root_size, beam_width=beam_width, tau_root=0.70,
        )
        source_evals.append(
            SourceEval(
                source=name,
                top_k={k: tuple(a.atom_id for a in ranked[:k]) for k in ks},
                recall_at_k=recall,
                best_ce_at_k=best_ce,
                root_set_ce=root_set.ce,
                root_set_size=root_set.size,
            )
        )

    return BlockEval(
        instance_id=problem.id,
        state_id=snapshot.schedule_hash if hasattr(snapshot, "schedule_hash") else "",
        block_id=block.block.block_id,
        appearance_type=appearance_type,
        appearance_before=appearance_before,
        n_probed=probed,
        n_high_ce=len(high_ce),
        high_ce_atoms=tuple(sorted(high_ce)),
        efficiency=efficiency,
        sources=source_evals,
    )


def evaluate_recall_at_k(
    problem: Problem,
    schedule: Schedule,
    snapshot: SymptomPruningSnapshot,
    *,
    executor: AtomicCounterfactualExecutor,
    blocks: tuple[PrunedAppearanceBlock, ...] | None = None,
    max_blocks: int = 8,
    **kwargs: Any,
) -> tuple[list[BlockEval], dict[str, Any]]:
    """Evaluate Recall@K across retained blocks; returns (per-block, aggregate)."""
    kept = blocks if blocks is not None else snapshot.retained
    graph = build_realized_propagation_graph(problem, schedule)
    block_evals: list[BlockEval] = []
    for block in kept[:max_blocks]:
        try:
            block_evals.append(
                evaluate_block_recall(
                    problem, schedule, snapshot, block,
                    executor=executor, graph=graph, **kwargs,
                )
            )
        except (ValueError, IndexError) as error:
            # A block with zero atoms / zero magnitude is skipped, not fatal.
            continue

    aggregate: dict[str, Any] = {"instance_id": problem.id, "n_blocks": len(block_evals)}
    for source in ("teacher", "max_plus", "page_rank", "random"):
        agg_k = {}
        for k in DEFAULT_KS:
            vals = [b_eval.sources[i].recall_at_k[k]
                    for b_eval in block_evals
                    for i, s in enumerate(b_eval.sources) if s.source == source]
            agg_k[str(k)] = sum(vals) / len(vals) if vals else 0.0
        aggregate[source] = {"mean_recall_at_k": agg_k}
    return block_evals, aggregate