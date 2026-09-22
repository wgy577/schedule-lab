"""Shared-atom-set probe for one appearance block (教师模型.md Phase2加强 §2/§3).

Every source re-ranks the *same* shared atomic set; the Solver CE ground truth
is measured once per atom.  This is the fairness contract: Teacher / MaxPlus /
PageRank / Random only reorder, they never probe different candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...ir import Problem, Schedule
from ...symptom_pruning import PrunedAppearanceBlock, SymptomPruningSnapshot
from ...validation import schedule_hash
from ..atom_generator import DecisionAtom, generate_root_atoms
from ..atomic_counterfactual_executor import AtomicCounterfactualExecutor
from ..baselines import dense_page_rank_relevance, random_relevance
from ..propagation_graph import RealizedPropagationGraph, build_realized_propagation_graph
from ..reverse_sensitivity import appearance_specific_seed, reverse_sensitivity
from ..solver_probe_runner import run_atomic_probe
from ..teacher_ranker import rank_atoms
from .metrics import spearman_score_ce

# Source orderings, each -> (source_name, ranked_atoms, score_by_atom).
SOURCE_NAMES = ("teacher", "max_plus", "page_rank", "random")


@dataclass
class AtomProbeRecord:
    atom: DecisionAtom
    ce: float
    feasible: bool
    label_valid: bool = False
    scores: dict[str, float] = field(default_factory=dict)   # source -> score
    ranks: dict[str, int] = field(default_factory=dict)      # source -> 1-based rank
    # Phase 2.10C §8/§16: de-saturated magnitude + detection score around the
    # atomic intervention.  Populated by probe_block_shared from run_atomic_probe.
    causal_magnitude_before: float = 0.0
    causal_magnitude_after: float = 0.0
    detection_before: float | None = None
    detection_after: float | None = None
    cmax_before: int | None = None
    cmax_after: int | None = None
    # Phase 2.11 §3: which sampling layer produced this atom.
    sampling_layer: str = "relevance_top"


@dataclass
class BlockProbe:
    problem_id: str
    state_id: str
    block_id: str
    appearance_type: str
    appearance_before: float
    atoms: tuple[DecisionAtom, ...]
    records: dict[str, AtomProbeRecord]        # atom_id -> record
    rankings: dict[str, tuple[str, ...]]       # source -> ordered atom_ids
    spearman: dict[str, float | None]          # source -> rho(score, CE)
    n_probed: int

    @property
    def ce_by_atom(self) -> dict[str, float]:
        return {aid: r.ce for aid, r in self.records.items()}

    def positive_set(self, tau_ce: float = 0.30) -> frozenset[str]:
        # A positive root requires a *valid* CE label, not just feasibility.
        return frozenset(
            aid for aid, r in self.records.items()
            if r.label_valid and r.ce >= tau_ce
        )

    def as_atom_record(self) -> list[dict[str, Any]]:
        rows = []
        for aid, r in self.records.items():
            rows.append({
                "appearance_id": self.block_id,
                "appearance_type": self.appearance_type,
                "atom_id": aid,
                "atom_type": r.atom.atom_type,
                "CE": r.ce,
                "feasible": r.feasible,
                "label_valid": r.label_valid,
                # Phase 2.10C §8: de-saturated single-atomic CE record.
                "causal_magnitude_before": r.causal_magnitude_before,
                "causal_magnitude_after": r.causal_magnitude_after,
                "ce_atomic": r.ce,
                "atomic_valid": True,  # AtomicCounterfactualExecutor honors §16 contract by construction
                "cmax_before": r.cmax_before,
                "cmax_after": r.cmax_after,
                # Phase 2.10C §16: D/M consistency audit fields.
                "detection_before": r.detection_before,
                "detection_after": r.detection_after,
                "magnitude_before": r.causal_magnitude_before,
                "magnitude_after": r.causal_magnitude_after,
                # Phase 2.11 §3: sampling layer.
                "sampling_layer": r.sampling_layer,
                **{f"{src}_score": r.scores.get(src, 0.0) for src in SOURCE_NAMES},
                **{f"{src}_rank": r.ranks.get(src, 0) for src in SOURCE_NAMES},
            })
        return rows


def _rank_by_relevance(atoms: tuple[DecisionAtom, ...], R: dict[str, float]) -> tuple[DecisionAtom, ...]:
    return tuple(
        sorted(atoms, key=lambda a: (-max(R.get(op, 0.0) for op in a.operations), a.atom_id))
    )


def _rank_by_teacher(atoms, R, problem: Problem, schedule: Schedule) -> tuple[DecisionAtom, ...]:
    ranked = rank_atoms(atoms, R, problem, schedule)
    return tuple(item.atom for item in ranked)


def probe_block_shared(
    problem: Problem,
    schedule: Schedule,
    block: PrunedAppearanceBlock,
    *,
    executor: AtomicCounterfactualExecutor,
    graph: RealizedPropagationGraph | None = None,
    top_atoms: int = 16,
    seed: int = 0,
    atoms_override: tuple[DecisionAtom, ...] | None = None,
    atom_layers: dict[str, str] | None = None,
) -> BlockProbe:
    """Probe one block's shared atom set once and rank it by every source.

    ``atoms_override`` (Phase 2.11): if provided, probe exactly these atoms
    instead of ``generate_root_atoms(top_k=top_atoms)``.  Used by the dataset
    runner to probe a stratified exploitation+exploration sample.
    ``atom_layers`` maps ``atom_id -> sampling_layer`` and is recorded on each
    atom row (``relevance_top`` / ``stratified_exploration`` /
    ``uniform_exploration``); defaults to ``relevance_top`` for every atom.
    """
    from ...sg_sct_causal_probe import appearance_magnitude, appearance_detection_score

    appearance = block.block
    appearance_type = appearance.appearance_rules[0] if appearance.appearance_rules else "UNKNOWN"
    appearance_before = appearance_magnitude(appearance)
    detection_before = appearance_detection_score(appearance)
    if graph is None:
        graph = build_realized_propagation_graph(problem, schedule)
    target_operations = tuple(appearance.operations)

    seeds = appearance_specific_seed(target_operations, appearance_type)
    R_teacher = reverse_sensitivity(graph, seeds)
    if atoms_override is not None:
        atoms = atoms_override
    else:
        atoms = generate_root_atoms(problem, schedule, graph, R_teacher, top_k=top_atoms)
    layer_of = atom_layers or {}

    # CE ground truth: probe every atom once through the atomic executor.  Only
    # valid labels (fail-closed appearance recompute) contribute a CE.
    ce: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    label_valid: dict[str, bool] = {}
    # Phase 2.10C §8/§16 per-atom measurement fields (from run_atomic_probe).
    mag_after: dict[str, float] = {}
    det_after: dict[str, float | None] = {}
    cmax_before: dict[str, int] = {}
    cmax_after: dict[str, int | None] = {}
    for atom in atoms:
        result = run_atomic_probe(
            problem, schedule, atom,
            appearance_type=appearance_type,
            appearance_before=appearance_before,
            target_operations=target_operations,
            executor=executor,
            detection_before=detection_before,
        )
        feasible[atom.atom_id] = result.feasible
        label_valid[atom.atom_id] = result.label_valid
        mag_after[atom.atom_id] = result.appearance_after
        det_after[atom.atom_id] = result.detection_after
        cmax_before[atom.atom_id] = result.makespan_before
        cmax_after[atom.atom_id] = result.makespan_after
        if result.label_valid:
            # max over actions for the CE_A(r)=max_a definition, BUT default
            # -inf (not 0.0) so a single negative CE -- an intervention that
            # *worsens* the appearance (magnitude_after > magnitude_before) --
            # is preserved as a signed label.  Phase 2.12 negative-CE audit
            # found the old ``0.0`` default silently clamped every negative CE
            # to 0 (16857 records in the 2.11 pilot), manufacturing a false
            # "no negative CE under clean atomic intervention" conclusion.
            ce[atom.atom_id] = max(ce.get(atom.atom_id, float("-inf")), result.ce)
        else:
            ce.setdefault(atom.atom_id, 0.0)

    # Source rankings.
    teacher_ranked = _rank_by_teacher(atoms, R_teacher, problem, schedule)
    R_maxplus = R_teacher
    maxplus_ranked = _rank_by_relevance(atoms, R_maxplus)
    R_page = dense_page_rank_relevance(problem, schedule, graph)
    pagerank_ranked = _rank_by_relevance(atoms, R_page)
    R_rand = random_relevance(graph, seed=seed)
    random_ranked = _rank_by_relevance(atoms, R_rand)

    rankings: dict[str, tuple[str, ...]] = {}
    orderings = {
        "teacher": teacher_ranked,
        "max_plus": maxplus_ranked,
        "page_rank": pagerank_ranked,
        "random": random_ranked,
    }
    for src, ranking in orderings.items():
        rankings[src] = tuple(a.atom_id for a in ranking)

    # Score per source (for Spearman): teacher uses TeacherScore, others use
    # max-relevance over the atom's operations.
    teacher_scores = {item.atom.atom_id: item.teacher_score for item in rank_atoms(atoms, R_teacher, problem, schedule)}
    maxplus_scores = {a.atom_id: max(R_maxplus.get(op, 0.0) for op in a.operations) for a in atoms}
    pagerank_scores = {a.atom_id: max(R_page.get(op, 0.0) for op in a.operations) for a in atoms}
    random_scores = {a.atom_id: max(R_rand.get(op, 0.0) for op in a.operations) for a in atoms}
    score_map = {
        "teacher": teacher_scores,
        "max_plus": maxplus_scores,
        "page_rank": pagerank_scores,
        "random": random_scores,
    }

    records: dict[str, AtomProbeRecord] = {}
    for atom in atoms:
        records[atom.atom_id] = AtomProbeRecord(
            atom=atom,
            ce=ce.get(atom.atom_id, 0.0),
            feasible=feasible.get(atom.atom_id, False),
            label_valid=label_valid.get(atom.atom_id, False),
            scores={src: score_map[src].get(atom.atom_id, 0.0) for src in SOURCE_NAMES},
            ranks={src: (list(rankings[src]).index(atom.atom_id) + 1) for src in SOURCE_NAMES},
            causal_magnitude_before=appearance_before,
            causal_magnitude_after=mag_after.get(atom.atom_id, 0.0),
            detection_before=detection_before,
            detection_after=det_after.get(atom.atom_id),
            cmax_before=cmax_before.get(atom.atom_id),
            cmax_after=cmax_after.get(atom.atom_id),
            sampling_layer=layer_of.get(atom.atom_id, "relevance_top"),
        )

    spearman: dict[str, float | None] = {}
    for src in SOURCE_NAMES:
        ordered = [records[aid] for aid in rankings[src] if aid in records]
        s = [r.scores[src] for r in ordered]
        c = [r.ce for r in ordered]
        spearman[src] = spearman_score_ce(s, c)

    return BlockProbe(
        problem_id=problem.id,
        state_id=schedule_hash(schedule),
        block_id=block.block.block_id,
        appearance_type=appearance_type,
        appearance_before=appearance_before,
        atoms=atoms,
        records=records,
        rankings=rankings,
        spearman=spearman,
        n_probed=len(atoms),
    )