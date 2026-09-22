"""Probe Priority (教师模型.md Phase2加强_2 §3.3/§15).

Only within the already causally-ranked Top-L candidates does intervention
utility re-order *which atom to probe first*:

    ProbePriority_A(r) = Rel_A(r) + beta * D(r)

``D(r)`` is the intervention utility (probe ease / cost), never a root-causality
signal.  ``beta`` is swept; if no positive beta beats ``beta=0`` the prior is
dropped entirely (ProbePriority = Rel_A).
"""

from __future__ import annotations


def probe_priority(rel: float, utility: float, beta: float) -> float:
    """``Rel + beta * D`` — probe ordering within the causal Top-L."""
    return float(rel) + float(beta) * float(utility)


def order_by_probe_priority(
    ranked_atoms,  # atom objects with .atom_id
    rel_by_atom: dict[str, float],
    utility_by_atom: dict[str, float],
    beta: float,
):
    """Order a pre-ranked candidate list by probe priority (descending)."""
    return tuple(
        sorted(
            ranked_atoms,
            key=lambda a: (
                -probe_priority(rel_by_atom.get(a.atom_id, 0.0),
                                utility_by_atom.get(a.atom_id, 0.0), beta),
                a.atom_id,
            ),
        )
    )