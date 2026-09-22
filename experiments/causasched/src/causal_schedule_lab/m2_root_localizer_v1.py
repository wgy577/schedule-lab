"""V5 M2 -- root decision localization (Phase 6, spec §9, §23-24).

Localization is *branch-local*: starting from an appearance block's member
operations, trace **upstream** along the TRUE_LOCAL_G_C reverse edges (job
predecessors + same-machine immediate predecessors).  Only decision sites that
are causal ancestors of the block within a bounded window are considered; every
site gets exactly one trace state (spec §9):

* ``ROOT_DECISION_CANDIDATE`` -- a terminal(-upstream) decision point in the
  traced frontier: the network may still propagate or resolve later, but this
  is where the causal chain bottoms out locally.
* ``EXPLAINED_PROPAGATION`` -- a site on the traced upstream path below a root
  (the pathway that carries the signal, not the root itself).
* ``UNRESOLVED_ROOT_CANDIDATE`` -- the block's trace could not reach a decidable
  root site (empty trace, or every terminal site is infeasible); it is surfaced
  honestly, never silently dropped.

**Rootness is never deviation alone.**  :func:`rootness` combines the calibrated
Z-deviation with appearance relevance + edit support (+ critical-context
weight), so a site the reference finds surprising *and* that is tightly tied to
the appearance and *actionable* (edits exist) ranks higher -- the Phase 7
proposal builder consumes this scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from .ir import Problem, Schedule
from .legal_edit_enumerator import LegalEditEnumerator
from .m2_decision_residual_v1 import DecisionResidualComputer
from .m2_v5_schema_v1 import (
    DecisionSite,
    TRACE_EXPLAINED,
    TRACE_ROOT,
    TRACE_UNRESOLVED,
)


def rootness(
    z_deviation: float,
    appearance_relevance: float,
    edit_support: int,
    *,
    critical_context: float = 0.0,
    context_penalty: float = 0.0,
    w_z: float = 1.0,
    w_rel: float = 1.0,
    w_edit: float = 1.0,
    w_crit: float = 1.0,
    w_context: float = 0.25,
) -> float:
    """Combined root score -- never deviation alone (spec §20).

    A standardised mix: Z-deviation (calibrated residual), appearance relevance
    in [0,1], edit support (how many legal edits open that decision) and
    critical-context (does the op sit on the makespan-bounding chain).
    """
    edit_norm = min(edit_support, 5) / 5.0
    return (
        w_z * z_deviation
        + w_rel * appearance_relevance
        + w_edit * edit_norm
        + w_crit * critical_context
        - w_context * context_penalty
    )


def _reverse_upstream(problem: Problem, schedule: Schedule) -> dict[str, set[str]]:
    """Reverse-TRUE_LOCAL upstream neighbour map.

    ``upstream[u]`` = the operations that causally feed ``u`` along
    precedence (job predecessors) and resource_sequence (prev on machine).
    """
    asg = schedule.assignment_map()
    mode_map = problem.mode_map()
    # machine -> timed order (start, end, op)
    machine_order: dict[str, list] = {}
    up: dict[str, set[str]] = {oid: set() for oid in asg}
    for oid, a in asg.items():
        mode = mode_map[a.mode_id][1]
        machine = mode.resources[0]
        machine_order.setdefault(machine, []).append((a.start, a.end, oid))
    for seq in machine_order.values():
        seq.sort(key=lambda t: (t[0], t[1], t[2]))
        for (s0, e0, o0), (_, _, o1) in zip(seq, seq[1:]):
            up[o1].add(o0)
    # job precedence
    for op in problem.operations:
        for pred in op.predecessors:
            if pred in up:
                up[op.id].add(pred)
    return up


@dataclass
class LocalizationResult:
    """Roots + full trace for one appearance block."""

    block_id: str
    root_sites: list[DecisionSite] = field(default_factory=list)
    path_sites: list[DecisionSite] = field(default_factory=list)
    unresolved: bool = False

    @property
    def sites(self) -> list[DecisionSite]:
        return self.path_sites + self.root_sites


class RootDecisionLocalizer:
    """Deterministic branch-local upstream tracer + trace-state labeller.

    Works on the full schedule; ``localize`` operates per appearance block.
    """

    def __init__(
        self,
        problem: Problem,
        schedule: Schedule,
        *,
        max_hops: int = 8,
        edit_ops: Sequence[str] | None = None,
    ):
        self.problem = problem
        self.schedule = schedule
        self.max_hops = max_hops
        self.upstream = _reverse_upstream(problem, schedule)
        self.enum = LegalEditEnumerator(problem, schedule)
        self.residual = DecisionResidualComputer(problem, schedule)
        # Routing and sequence sites are both first-class root candidates.
        sites, _, _ = self.residual.compute()
        self._sites_by_op: dict[str, list[DecisionSite]] = {}
        for site in sites:
            self._sites_by_op.setdefault(site.operation_id, []).append(site)
            if site.successor_id:
                self._sites_by_op.setdefault(site.successor_id, []).append(site)
        # how many legal edits open each operation's decision
        self._edit_support: dict[str, int] = {}
        subjects = set(edit_ops) if edit_ops is not None else set(self.enum.eligible)
        for oid in subjects:
            if oid in self.enum.eligible:
                self._edit_support[oid] = len(self.enum.enumerate([oid]))

    def _trace_frontier(self, members: Iterable[str]) -> set[str]:
        """Upstream BFS within ``max_hops`` from the block members."""
        frontier: set[str] = set()
        visited: dict[str, int] = {}
        queue: list[tuple[str, int]] = [(m, 0) for m in set(members) if m in self.upstream]
        while queue:
            node, depth = queue.pop(0)
            if depth > self.max_hops:
                continue
            if visited.get(node, 1 << 30) <= depth:
                continue
            visited[node] = depth
            frontier.add(node)
            for p in self.upstream.get(node, ()):
                queue.append((p, depth + 1))
        return frontier

    def localize(self, block_id: str, members: Iterable[str]) -> LocalizationResult:
        frontier = self._trace_frontier(members)
        if not frontier:
            return LocalizationResult(block_id=block_id, unresolved=True)

        # Build a DecisionSite per candidate op that is decidable (has a
        # routing site) with a combined root score.
        cand: dict[str, tuple[DecisionSite, float, float, int, float]] = {}
        members_set = set(members)
        for oid in frontier:
            for site in self._sites_by_op.get(oid, ()):
                rel = 1.0 if oid in members_set else self._appearance_relevance(oid)
                edit_sup = self._edit_support.get(oid, 0)
                context = self._context_penalty(oid)
                score = rootness(
                    site.z_deviation,
                    rel,
                    edit_sup,
                    context_penalty=context,
                )
                cand[site.site_id] = (site, score, rel, edit_sup, context)
        if not cand:
            # reachable ops exist but none is decidable -> unresolved, honestly
            return LocalizationResult(block_id=block_id, unresolved=True)

        # One composite-score maximum per disconnected causal branch.  This
        # prevents the old degeneration to "furthest upstream operation" while
        # preserving multiple independent roots.
        undirected = {oid: set() for oid in frontier}
        for oid in frontier:
            for parent in self.upstream.get(oid, ()) & frontier:
                undirected[oid].add(parent)
                undirected[parent].add(oid)
        components: list[set[str]] = []
        unseen = set(frontier)
        while unseen:
            seed = unseen.pop()
            component = {seed}
            queue = [seed]
            while queue:
                node = queue.pop()
                for other in undirected[node] & unseen:
                    unseen.remove(other)
                    component.add(other)
                    queue.append(other)
            components.append(component)
        roots: set[str] = set()
        for component in components:
            eligible = [key for key, row in cand.items() if row[0].operation_id in component]
            if eligible:
                roots.add(max(eligible, key=lambda key: cand[key][1]))

        root_sites: list[DecisionSite] = []
        path_sites: list[DecisionSite] = []
        for site_id, (site, score, rel, edit_sup, context) in cand.items():
            state = TRACE_ROOT if site_id in roots else TRACE_EXPLAINED
            d = _copy_site_with(site, rel, edit_sup, state)
            if site_id in roots:
                root_sites.append(d)
            else:
                path_sites.append(d)
        root_sites.sort(key=lambda d: d.site_id)
        path_sites.sort(key=lambda d: d.site_id)
        return LocalizationResult(
            block_id=block_id, root_sites=root_sites, path_sites=path_sites
        )

    def _appearance_relevance(self, oid: str) -> float:
        # decision-time proxy: how loaded the op's machine is (normalised)
        load = self.enum.loads.get(self._machine_of(oid), 0.0)
        return min(load / (self.enum.makespan or 1.0), 1.0)

    def _machine_of(self, oid: str) -> str:
        mode = self.enum.mode_map[self.schedule.assignment_map()[oid].mode_id][1]
        return mode.resources[0]

    def _criticality(self, oid: str) -> bool:
        """Cheap makespan-criticality: the op itself runs into the makespan, or
        it is the last scheduled op on the makespan-bounding machine."""
        a = self.schedule.assignment_map().get(oid)
        if a is None:
            return False
        if float(a.end) >= (self.enum.makespan - 1e-9):
            return True
        machine = self._machine_of(oid)
        seq = self.enum.sequences.get(machine, [])
        if seq and seq[-1] == oid:
            last_end = float(self.schedule.assignment_map()[seq[-1]].end)
            if last_end >= (self.enum.makespan - 1e-9):
                return True
        return False

    def _context_penalty(self, oid: str) -> float:
        """Soft context score: a heavily loaded current machine is explanatory
        context, not by itself proof that this operation is the root."""
        machine = self._machine_of(oid)
        return min(
            self.enum.loads.get(machine, 0.0) / max(self.enum.makespan, 1.0),
            1.0,
        )


def _copy_site_with(
    site: DecisionSite, relevance: float, edit_support: int, trace_state: str
) -> DecisionSite:
    return replace(
        site,
        appearance_relevance=relevance,
        edit_support=edit_support,
        trace_state=trace_state,
    )


def localize_block(
    problem: Problem,
    schedule: Schedule,
    block_id: str,
    members: Iterable[str],
    **kw,
) -> LocalizationResult:
    """Convenience wrapper (deterministic)."""
    return RootDecisionLocalizer(problem, schedule, **kw).localize(block_id, members)


__all__ = [
    "RootDecisionLocalizer",
    "LocalizationResult",
    "rootness",
    "localize_block",
]
