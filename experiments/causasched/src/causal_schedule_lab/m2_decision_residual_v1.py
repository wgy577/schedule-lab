"""V5 M2 -- decision residual (Phase 5, spec §20 & AERCA-adaptation).

A decision site ``d`` (routing ``o->m`` or sequence ``o_i prec_m o_j``) gets a
**decision residual** :math:`R_d = -\\log P_{ref}(actual\\ decision\\ ;\\ C_d)`
measured against a deterministic *reference* decision model that only sees
decision-time-visible context :math:`C_d` (loads, makespan, processing, slack).
It is then calibrated to a Z-score:

.. math:: Z_d = \\frac{R_d - \\mu_R}{\\sigma_R + \\epsilon}

Rootness is **never** this deviation alone -- the full root score combines
:math:`Z_d` with appearance relevance + edit support + critical context
(multiplication happens in Phase 6/7).  This module produces the residual side
and the (deterministic, decision-time) reference models.

Two invariants the tests lock:
* An **illegal** routing decision (machine not in the operation's eligible set)
  yields :math:`R = +\\infty` (zero reference support), so the network is never
  rewarded for an infeasible route.
* Featurisation is decision-time-only (spec §45-11): no "after" estimate leaks
  into the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ir import Problem, Schedule
from .legal_edit_enumerator import LegalEditEnumerator
from .m2_v5_schema_v1 import (
    DecisionSite,
    ROUTING_DECISION,
    SEQUENCE_DECISION,
    route_site_id,
    sequence_site_id,
)


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    import math

    mx = max(logits.values()) if logits else 0.0
    e = {k: math.exp(v - mx) for k, v in logits.items()}
    s = sum(e.values())
    if s <= 0.0:
        return {k: 1.0 / len(e) for k in e}
    return {k: v / s for k, v in e.items()}


@dataclass(frozen=True)
class RoutingDecisionReferenceModel:
    """Load-balancing default scheduler.

    The reference believes a route to machine ``m`` is more likely when that
    machine's current relative load is lower::

        logit_m = -w_load * rel_load_m - w_p * p(o, m) / p_max

    ``rel_load_m = load_m / makespan``; ``p(o, m)`` is that mode's duration.
    Everything is decision-time-visible (spec §45-11).
    """

    w_load: float = 1.0
    w_p: float = 0.0

    def logits(self, oid: str, loads: dict[str, float], makespan: float, eligible) -> dict[str, float]:
        ms = makespan if makespan > 0 else 1.0
        out: dict[str, float] = {}
        for m in eligible.get(oid, {}):
            rel = (loads.get(m, 0.0)) / ms
            modes = eligible.get(oid, {}).get(m, ())
            duration = min((float(mode.duration) for mode in modes), default=0.0)
            p_max = max(
                (float(mode.duration) for group in eligible.get(oid, {}).values() for mode in group),
                default=1.0,
            )
            out[m] = -self.w_load * rel - self.w_p * duration / max(p_max, 1e-9)
        return out

    def probabilities(self, oid: str, loads, makespan, eligible) -> dict[str, float]:
        return _softmax(self.logits(oid, loads, makespan, eligible))

    def residual(self, oid: str, actual_machine: str | None, loads, makespan, eligible) -> float:
        """-log P_ref(actual | o).  ``+inf`` iff the route is illegal (the machine
        is not in ``eligible[oid]``, so the reference gives it zero support)."""
        import math

        probs = self.probabilities(oid, loads, makespan, eligible)
        p = probs.get(actual_machine, 0.0)
        if p <= 0.0:
            return float("inf")
        return -math.log(p)


@dataclass(frozen=True)
class SequenceDecisionReferenceModel:
    """Reference sequencing default.

    For an adjacent pair ``(left, right)`` on a resource, the reference believes
    the schedule's order is typical with probability

        P(left before right) = 1 / (1 + exp(-(slack_left - slack_right)))

    i.e. it slightly prefers running the tighter (less slack) operation earlier.
    Residual = -log P(actual order).  Deterministic and decision-time-visible.
    """

    w: float = 1.0

    def residual(self, slack_left: float, slack_right: float) -> float:
        import math

        # Less slack means tighter and should therefore increase the
        # probability of being sequenced first.
        z = self.w * (slack_right - slack_left)
        # Numerically stable sigmoid: for large |z| the naive 1/(1+exp(-z))
        # overflows math.exp (e.g. DPpaulli10a has |z| ~ 825 > 709.78).  The two
        # branches are the standard mathematically-equivalent stable form.
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            p = ez / (1.0 + ez)
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return -math.log(p)


def calibrate_residuals(residuals) -> tuple[float, float]:
    """Mean/std of the finite residuals (Z = (r - mu) / (sigma + eps))."""
    arr = np.asarray([r for r in residuals if r != float("inf")], dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    mu = float(arr.mean())
    sigma = float(arr.std())
    if sigma < 1e-12 or not np.isfinite(sigma):
        sigma = 1.0
    return mu, sigma


class DecisionResidualComputer:
    """Deterministic residual + calibration for every routing/sequence decision
    site in a schedule (decision-time context only)."""

    def __init__(
        self,
        problem: Problem,
        schedule: Schedule,
        *,
        routing_ref: RoutingDecisionReferenceModel | None = None,
        seq_ref: SequenceDecisionReferenceModel | None = None,
    ):
        self.problem = problem
        self.schedule = schedule
        self.enum = LegalEditEnumerator(problem, schedule)
        self.routing_ref = routing_ref or RoutingDecisionReferenceModel()
        self.seq_ref = seq_ref or SequenceDecisionReferenceModel()
        self.loads = self.enum.loads
        self.makespan = self.enum.makespan
        # per-operation slack proxy: end of its job's whole-chain gap is too
        # heavy; use the op-level "room" = makespan - (end - start) - earliest
        # approximated by how close its own start is to its predecessor.
        self._slack: dict[str, float] = self._compute_slack()

    def _compute_slack(self) -> dict[str, float]:
        asg = self.schedule.assignment_map()
        slack: dict[str, float] = {}
        for oid, a in asg.items():
            room = max(0.0, self.makespan - float(a.end))
            slack[oid] = room
        return slack

    # -- routing sites --------------------------------------------------------

    def _routing_sites(self) -> list[DecisionSite]:
        sites: list[DecisionSite] = []
        for oid in sorted(self.enum.eligible):
            asg = self.schedule.assignment_map().get(oid)
            if asg is None:
                continue
            mode = self.enum.mode_map[asg.mode_id][1]
            actual = _machine_of(mode)
            r = self.routing_ref.residual(oid, actual, self.loads, self.makespan, self.enum.eligible)
            context = (
                ("p_current", float(mode.duration)),
                ("source_load", self.loads.get(actual, 0.0)),
                ("target_load", self.loads.get(actual, 0.0)),
            )
            sites.append(DecisionSite(
                site_id=route_site_id(oid, actual),
                operation_id=oid,
                decision_type=ROUTING_DECISION,
                source_machine=actual,
                target_machine=actual,
                mode_id=asg.mode_id,
                context=context,
                residual=float(r),
            ))
        return sites

    # -- sequence sites -------------------------------------------------------

    def _sequence_sites(self) -> list[DecisionSite]:
        sites: list[DecisionSite] = []
        for resource_id, seq in sorted(self.enum.sequences.items()):
            for left, right in zip(seq, seq[1:]):
                r = self.seq_ref.residual(
                    slack_left=self._slack.get(left, 0.0),
                    slack_right=self._slack.get(right, 0.0) + 1e-6,
                )
                sites.append(DecisionSite(
                    site_id=sequence_site_id(left, right, resource_id),
                    operation_id=left,
                    decision_type=SEQUENCE_DECISION,
                    resource_id=resource_id,
                    predecessor_id=left,
                    successor_id=right,
                    context=(("idle_right", float(self._gap(left, right, resource_id))),),
                    residual=float(r),
                ))
        return sites

    def _gap(self, left: str, right: str, resource_id: str) -> float:
        asg = self.schedule.assignment_map()
        a, b = asg.get(left), asg.get(right)
        if a is None or b is None:
            return 0.0
        return max(0.0, float(b.start) - float(a.end))

    # -- public ---------------------------------------------------------------

    def compute(self) -> tuple[list[DecisionSite], float, float]:
        """Residual + calibration over all routing & sequence sites.

        Returns ``(sites, mu, sigma)``; each site carries ``residual`` and the
        calibrated ``residual_mean``/``residual_std``/``z_deviation``.
        """
        routing = self._routing_sites()
        sequencing = self._sequence_sites()
        sites = routing + sequencing
        route_mu, route_sigma = calibrate_residuals([s.residual for s in routing])
        seq_mu, seq_sigma = calibrate_residuals([s.residual for s in sequencing])
        for s in sites:
            mu, sigma = (
                (route_mu, route_sigma)
                if s.decision_type == ROUTING_DECISION
                else (seq_mu, seq_sigma)
            )
            if s.residual == float("inf"):
                z = 99.0  # +inf residual -> top tail (flagged)
            else:
                z = (s.residual - mu) / (sigma + 1e-8)
            object.__setattr__(s, "residual_mean", mu)
            object.__setattr__(s, "residual_std", sigma)
            object.__setattr__(s, "z_deviation", float(z))
        all_mu, all_sigma = calibrate_residuals([s.residual for s in sites])
        return sites, all_mu, all_sigma


def _machine_of(mode) -> str:
    if len(mode.resources) != 1:
        raise ValueError("V5 decision residual requires unary-resource modes")
    return mode.resources[0]


def build_decision_sites(
    problem: Problem,
    schedule: Schedule,
) -> tuple[list[DecisionSite], float, float]:
    """Convenience wrapper (deterministic)."""
    return DecisionResidualComputer(problem, schedule).compute()


__all__ = [
    "RoutingDecisionReferenceModel",
    "SequenceDecisionReferenceModel",
    "DecisionResidualComputer",
    "build_decision_sites",
    "calibrate_residuals",
]
