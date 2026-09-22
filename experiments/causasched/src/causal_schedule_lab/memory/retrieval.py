"""Patch §5 -- retrieval of past (S, P, S') transitions for a query state.

Given a current scheduling state, :func:`retrieve_experiences` returns the
stored transitions whose :func:`state_vector` is most similar (cosine distance)
to the query vector.  This is the primitive M3's "memory prior" and the
counterfactual evaluator's "delayed success uses memory prediction" both consume
(Patch §6/§7): the retrieved outcomes are the empirical distribution of
:math:`P(success | S, P)` for nearby states.
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence

from .experience_store import (
    ExperienceStore,
    InterventionExperience,
    experience_final_gain,
    experience_future_success,
)
from .state_encoder import state_vector


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return sqrt(sum(x * x for x in a))


def experience_similarity(
    query: Sequence[float], candidate: Sequence[float]
) -> float:
    """Cosine similarity in [-1, 1]; 0-safe (empty vectors -> 0.0)."""
    qn, cn = _norm(query), _norm(candidate)
    if qn == 0.0 or cn == 0.0:
        return 0.0
    return _dot(query, candidate) / (qn * cn)


def top_k_by_similarity(
    query: Sequence[float],
    candidates: Sequence[tuple[str, Sequence[float]]],
    *,
    k: int,
) -> list[tuple[str, float]]:
    """Return ``(key, similarity)`` for the ``k`` most-similar candidates desc."""
    scored = [(key, experience_similarity(query, vec)) for key, vec in candidates]
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def retrieve_experiences(
    store: ExperienceStore,
    query: Sequence[float],
    *,
    k: int = 5,
) -> list[InterventionExperience]:
    """Top-``k`` stored experiences closest (in state-vector cosine) to query."""
    if k <= 0 or len(store) == 0:
        return []
    query = tuple(float(x) for x in query)
    cands = [(e.key, state_vector(e.state)) for e in store.experiences()]
    picks = top_k_by_similarity(query, cands, k=k)
    return [store.get(key) for key, _sim in picks if store.get(key) is not None]


def memory_success_estimate(
    store: ExperienceStore,
    query: Sequence[float],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Rough empirical :math:`P(success|S,P)` prior + expected gain from memory.

    Returns ``{"estimate": p, "nv": n_trials, "mean_gain": g}`` over the ``k``
    nearest *resolved* experiences.  Empty / unresolved memory yields
    ``estimate = 0.0`` (fail-safe conservative) -- the exploiter decides.

    ``gain`` is the magnitude of delta-Cmax reduction for success outcomes.
    """
    q = tuple(float(x) for x in query)
    cands = [
        (e.key, state_vector(e.state))
        for e in store.experiences()
        if e.outcome is not None and e.outcome.classification != "pending"
    ]
    if not cands:
        return {"estimate": 0.0, "nv": 0, "mean_gain": 0.0, "fiv": 0.0}
    picks = top_k_by_similarity(q, cands, k=max(1, k))
    records = [store.get(key) for key, _ in picks]
    succ = [record for record in records
            if record is not None and experience_future_success(record)]
    nv = len(picks)
    est = len(succ) / nv if nv else 0.0
    gains = [experience_final_gain(record) for record in succ]
    mean_gain = (sum(gains) / len(gains)) if gains else 0.0
    return {"estimate": float(est), "nv": nv, "mean_gain": float(mean_gain),
            "fiv": float(est * mean_gain)}


__all__ = [
    "experience_similarity",
    "top_k_by_similarity",
    "retrieve_experiences",
    "memory_success_estimate",
]
