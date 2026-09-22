"""Phase-2 enhanced evaluation subpackage (教师模型.md Phase2加强)."""

from .metrics import (
    best_ce_at_k,
    hit_at_k,
    lift_at_k,
    mean_ce_at_k,
    precision_at_k,
    recall_at_k,
    spearman_score_ce,
    success_rate_at_k,
)
from .random_baseline import RandSummary, evaluate_random_ranking
from .probe import AtomProbeRecord, BlockProbe, probe_block_shared
from .reporting import (
    BlockMetrics,
    aggregate_overall,
    breakdown_by_appearance,
    breakdown_by_atom_type,
    compute_block_metrics,
    missed_high_ce_roots,
    pairwise_comparison,
    teacher_false_top,
)
from .runner import run_phase2

DEFAULT_KS = (1, 3, 5, 10)

__all__ = [
    "DEFAULT_KS",
    # metrics
    "recall_at_k", "precision_at_k", "best_ce_at_k", "mean_ce_at_k",
    "hit_at_k", "success_rate_at_k", "lift_at_k", "spearman_score_ce",
    # random_baseline
    "RandSummary", "evaluate_random_ranking",
    # probe
    "AtomProbeRecord", "BlockProbe", "probe_block_shared",
    # reporting
    "BlockMetrics", "compute_block_metrics", "aggregate_overall",
    "breakdown_by_appearance", "breakdown_by_atom_type",
    "missed_high_ce_roots", "teacher_false_top", "pairwise_comparison",
    # runner
    "run_phase2",
]