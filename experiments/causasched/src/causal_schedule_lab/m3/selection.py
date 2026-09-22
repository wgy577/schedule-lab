r"""Patch §9/§10 -- final proposal selection + Future Improvement Value (FIV).

Stage 3 pick from a set of counterfactually-*evaluated* proposals (each carrying
a :class:`~causal_schedule_lab.counterfactual.CounterfactualResult`) is governed
by a fixed priority (Patch §10), and the vague "new opportunity" notion is
replaced by a concrete value (Patch §9):

* :func:`future_improvement_value` -- trajectory-memory future value.  It is
  used for a Cmax-neutral proposal; direct success is represented by its
  observed :math:`-\Delta C_{max}` term rather than being counted twice.
* :func:`select_final` -- rank-order kept proposals by:

  ``1. direct Cmax reduction (ΔC<0)  →  2. FIV  →  3. lower Risk``

If no proposal is kept, :func:`select_final` returns a rejected decision (a
``NO_TRUE_IMPROVEMENT`` -- nothing is applied).  The selection is pure
(deterministic, torch-free): it makes no learning claim by itself, it only
surfaces the @fixed@ rule the training pipeline can later learn to approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..counterfactual import CounterfactualResult

DIRECT = 0
FIV = 1


@dataclass(frozen=True)
class SelectionDecision:
    """The final pick (or a rejected decision)."""

    chosen_index: int | None  # index into the input results, None => rejected
    reason: str               # DIRECT_SUCCESS | FIV | NO_TRUE_IMPROVEMENT
    fiv: float = 0.0
    risk: float = 0.0
    keep_count: int = 0


def future_improvement_value(result: CounterfactualResult) -> float:
    """Trajectory-memory FIV for a Cmax-neutral delayed-success proposal."""
    if result.classification == "delayed_success":
        return float(result.ev)
    return 0.0


def risk_score(result: CounterfactualResult) -> float:
    """Observed side-effect risk; Gap is not an optimization signal."""
    return float(max(0.0, result.risk))


def select_final(results: Sequence[CounterfactualResult]) -> SelectionDecision:
    """Pick the best kept proposal by the Patch §10 priority rule."""
    kept = [r for r in results if r.keep]
    keep_count = len(kept)
    if not kept:
        return SelectionDecision(chosen_index=None, reason="NO_TRUE_IMPROVEMENT",
                                 keep_count=0)

    def key(r: CounterfactualResult) -> tuple:
        if r.classification == "direct_success":
            tier = DIRECT
            metric = float(max(0.0, -r.delta_cmax))  # bigger improvement first
        else:  # delayed_success (kept, no direct reduction)
            tier = FIV
            metric = future_improvement_value(r)
        return (tier, -metric, risk_score(r))

    best = min(kept, key=key)
    idx = results.index(best)
    reason = "DIRECT_SUCCESS" if best.classification == "direct_success" else "FIV"
    return SelectionDecision(
        chosen_index=idx, reason=reason,
        fiv=future_improvement_value(best),
        risk=risk_score(best),
        keep_count=keep_count,
    )


__all__ = [
    "SelectionDecision",
    "future_improvement_value",
    "select_final",
    "risk_score",
]
