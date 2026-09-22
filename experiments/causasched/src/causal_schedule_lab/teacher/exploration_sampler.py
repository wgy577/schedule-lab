"""Exploration candidate sampler (教师模型.md §12).

Teacher + LLM candidates can form a closed confirmatory loop.  A small fraction
of the probe budget (default 15%) is reserved for *structurally legal* but
*low-prior* atoms chosen by diversity random sampling, so the CE labels are not
biased toward the teacher's own prior.  No LLM in Phase 1; the 25% LLM slice is
folded into teacher/exploration until Phase 3.
"""

from __future__ import annotations

import random
from typing import Sequence

from .atom_generator import DecisionAtom

EXPLORATION_FRACTION = 0.15


def sample_exploration_atoms(
    atoms: Sequence[DecisionAtom],
    *,
    budget: int,
    exclude_atom_ids: frozenset[str] = frozenset(),
    rng: random.Random | None = None,
) -> tuple[DecisionAtom, ...]:
    """Diversely sample ``budget`` legal atoms, excluding ``exclude_atom_ids``.

    To spread diversity we walk the atom list in a deterministic rotating order
    (bucket by operator/probe type) rather than taking the top by score.
    """
    if budget <= 0:
        return ()
    rng = rng or random.Random(0)
    available = [a for a in atoms if a.atom_id not in exclude_atom_ids]
    if not available:
        return ()
    # Bucket by probe operator id so different decisions share the budget.
    buckets: dict[str, list[DecisionAtom]] = {}
    for atom in available:
        buckets.setdefault(atom.probe_operator_id or "unknown", []).append(atom)
    for key in buckets:
        rng.shuffle(buckets[key])
    pooled: list[DecisionAtom] = []
    keys = list(buckets)
    rng.shuffle(keys)
    while keys and len(pooled) < budget:
        drained = False
        for key in list(keys):
            if buckets[key]:
                pooled.append(buckets[key].pop(0))
                if len(pooled) >= budget:
                    break
            else:
                keys.remove(key)
                drained = True
        if not drained and len(pooled) < budget:
            break
    return tuple(pooled)