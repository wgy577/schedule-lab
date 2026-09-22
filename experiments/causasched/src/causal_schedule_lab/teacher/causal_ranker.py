"""Causal Root Ranking (教师模型.md Phase2加强_2 §3.1/§15).

Phase 2.5 decouples *cause* from *ease of intervention*.  The official root
ranking is purely the propagation relevance:

    RootScore_A(r) = Rel_A(r)

from Max-Plus Reverse Sensitivity.  Intervention utility / probe priority must
NOT enter the root ranking (`Root Ranking != Probe Ordering`).  This module is
the single authority for the causal root score.
"""

from __future__ import annotations

from .atom_generator import DecisionAtom


def causal_root_score(propagation_relevance: float) -> float:
    """``RootScore_A(r) = Rel_A(r)`` — the official causal root ranking score."""
    return float(propagation_relevance)


def rank_by_causal_score(atoms, relevance: dict[str, float]):
    """Rank atoms by ``RootScore = max Rel`` over their operations (descending)."""
    return tuple(
        sorted(
            atoms,
            key=lambda a: (-max(relevance.get(op, 0.0) for op in a.operations), a.atom_id),
        )
    )


def atom_causal_root_score(atom: DecisionAtom, relevance: dict[str, float]) -> float:
    """:func:`causal_root_score` applied to one atom's max operation relevance."""
    return causal_root_score(max(relevance.get(op, 0.0) for op in atom.operations))