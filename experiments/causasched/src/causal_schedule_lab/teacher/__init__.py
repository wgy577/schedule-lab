"""ACCT (Appearance-Conditioned Causal Teacher) — offline Solver-counterfactual
CE_A label generator that teaches M2 where root causes live.

Phase 1 (this package, no LLM) implements the propagation teacher, the
Max-Plus reverse sensitivity, decision atoms, the rule probe, the atomic +
joint Solver probes, the greedy/beam root-set search, and a small Gold Set
writer.  It is a *data generator*: at inference time neither the teacher nor
the LLM nor per-candidate Solver probes are needed (教师模型.md §27).

Boundaries (教师模型.md §31, respected here by construction):
* ``CE_A`` is a causal-effect label on the *target appearance*, never a makespan
  label.  ``ΔC_max`` is recorded alongside but never used as the M2 root label.
* The teacher prior (G_prop propagation + TeacherScore) orders candidates only;
  it never defines the label.  The label is measured purely from the
  counterfactual appearance change produced by the real CP-SAT solver.
* ``identified=false``: this generator produces supervised *evidence* for the
  root-cause model; it does not claim to have identified the true cause.
"""

from .propagation_graph import (
    RealizedPropagationGraph,
    build_realized_propagation_graph,
    processing_time_scale,
)
from .reverse_sensitivity import (
    appearance_specific_seed,
    reverse_sensitivity,
)
from .atom_generator import (
    DecisionAtom,
    generate_root_atoms,
)
from .teacher_ranker import (
    compute_teacher_score,
    rank_atoms,
)
from .counterfactual_label import (
    causal_effect,
    makespan_gain,
)
from .solver_probe_runner import (
    AtomicProbeResult,
    JointProbeResult,
    run_atomic_probe,
    run_joint_probe,
)
from .root_set_search import (
    RootSet,
    build_root_set,
)
from .dataset_writer import (
    write_gold_samples,
)
from .main import generate_gold_set
from .eval_recall import (
    BlockEval,
    SourceEval,
    evaluate_block_recall,
    evaluate_recall_at_k,
)
from .baselines import (
    dense_page_rank_relevance,
    random_relevance,
)
from .causal_ranker import (
    atom_causal_root_score,
    causal_root_score,
    rank_by_causal_score,
)
from .probe_ranker import (
    order_by_probe_priority,
    probe_priority,
)
from .intervention_utility import (
    routing_probe_utility,
    sequencing_probe_utility,
)
from .phase2_5 import run_phase2_5

PHASE = 2

__all__ = [
    "PHASE",
    # propagation_graph
    "RealizedPropagationGraph",
    "build_realized_propagation_graph",
    "processing_time_scale",
    # reverse_sensitivity
    "appearance_specific_seed",
    "reverse_sensitivity",
    # atom_generator
    "DecisionAtom",
    "generate_root_atoms",
    # teacher_ranker
    "compute_teacher_score",
    "rank_atoms",
    # counterfactual_label
    "causal_effect",
    "makespan_gain",
    # solver_probe_runner
    "AtomicProbeResult",
    "JointProbeResult",
    "run_atomic_probe",
    "run_joint_probe",
    # root_set_search
    "RootSet",
    "build_root_set",
    # dataset_writer
    "write_gold_samples",
    # main
    "generate_gold_set",
    # eval_recall
    "BlockEval",
    "SourceEval",
    "evaluate_block_recall",
    "evaluate_recall_at_k",
    # baselines
    "dense_page_rank_relevance",
    "random_relevance",
    # causal_ranker (Phase 2.5 official Root Ranking = pure Rel_A)
    "causal_root_score",
    "rank_by_causal_score",
    "atom_causal_root_score",
    # probe_ranker (Phase 2.5 probe ordering = Rel + beta*D)
    "probe_priority",
    "order_by_probe_priority",
    # intervention_utility (Phase 2.5 renamed priors, never enter RootScore)
    "routing_probe_utility",
    "sequencing_probe_utility",
    # phase2_5 (offline decoupling re-rank)
    "run_phase2_5",
]