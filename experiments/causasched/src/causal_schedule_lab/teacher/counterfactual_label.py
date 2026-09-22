"""Counterfactual labels from the Solver probe (教师模型.md §15/§16).

    CE_A(r, a) = (A_old - A_new) / (A_old + eps)
    G_C(r, a)  = (Cmax(S) - Cmax(S')) / Cmax(S)

``CE_A`` is the M2 causal-effect label; ``G_C`` is recorded alongside but is
never used as the M2 root label (教师模型.md §16).  Both derive from the real
Solver counterfactual, never from the teacher prior.
"""

from __future__ import annotations

EPS = 1e-6


def causal_effect(appearance_before: float, appearance_after: float) -> float:
    """``CE_A = (A_old - A_new)/(A_old + eps)`` — relieving when positive."""
    return (appearance_before - appearance_after) / (appearance_before + EPS)


def makespan_gain(makespan_before: float, makespan_after: float | None) -> float:
    """``G_C = (Cmax(S) - Cmax(S'))/Cmax(S)`` — improving when positive."""
    if makespan_after is None:
        return 0.0
    return (makespan_before - makespan_after) / (makespan_before + EPS)