"""Bounded rule/graph search from an M2 root decision to a macro action chain.

The Intervention Transition Reasoner (ITR) is deliberately not a root model and
does not call CP-SAT.  It consumes M2 root decisions, already-enumerated legal
edits, and a deterministic schedule graph view.  When a desired routing edit's
target window is occupied, it recursively searches legal relocations for the
blocking operations and emits a closed, ordered intervention chain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from ..ir import Problem, Schedule
from ..m2_v5_schema_v1 import EDIT_ROUTE, LegalEdit


@dataclass(frozen=True)
class InterventionTransitionConfig:
    max_depth: int
    beam_width: int
    max_chain_length: int
    enable_recursive_expansion: bool
    blocker_search_radius: int
    min_action_score: float
    feasibility_weight: float
    actionability_weight: float
    expected_impact_weight: float
    length_weight: float

    def validate(self) -> None:
        if self.max_depth < 1:
            raise ValueError("intervention_transition.max_depth must be >= 1")
        if self.beam_width < 1:
            raise ValueError("intervention_transition.beam_width must be >= 1")
        if self.max_chain_length < 1:
            raise ValueError("intervention_transition.max_chain_length must be >= 1")
        if self.blocker_search_radius < 1:
            raise ValueError("intervention_transition.blocker_search_radius must be >= 1")
        for name, value in (
            ("feasibility", self.feasibility_weight),
            ("actionability", self.actionability_weight),
            ("expected_impact", self.expected_impact_weight),
            ("length", self.length_weight),
        ):
            if value < 0:
                raise ValueError(f"intervention_transition score weight {name} must be >= 0")


def _repository_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "hyperparameters.yaml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


def load_intervention_transition_config(
    path: str | Path | None = None,
) -> InterventionTransitionConfig:
    """Load the centralized JSON-compatible YAML hyperparameter contract."""
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or (_repository_root() / "configs" / "hyperparameters.yaml")
    ).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"central hyperparameter file not found: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError("hyperparameters.yaml must be JSON-compatible YAML") from error
    if payload.get("schema") != "causal_schedule_lab_hyperparameters_v1":
        raise ValueError("unsupported hyperparameter schema")
    section = payload.get("intervention_transition")
    if not isinstance(section, Mapping):
        raise ValueError("missing intervention_transition hyperparameters")
    weights = section.get("score_weights")
    if not isinstance(weights, Mapping):
        raise ValueError("missing intervention_transition.score_weights")
    config = InterventionTransitionConfig(
        max_depth=int(section["max_depth"]),
        beam_width=int(section["beam_width"]),
        max_chain_length=int(section["max_chain_length"]),
        enable_recursive_expansion=bool(section["enable_recursive_expansion"]),
        blocker_search_radius=int(section["blocker_search_radius"]),
        min_action_score=float(section["min_action_score"]),
        feasibility_weight=float(weights["feasibility"]),
        actionability_weight=float(weights["actionability"]),
        expected_impact_weight=float(weights["expected_impact"]),
        length_weight=float(weights["length"]),
    )
    config.validate()
    return config


@dataclass(frozen=True)
class ScheduledInterval:
    operation_id: str
    machine_id: str
    start: float
    end: float


@dataclass(frozen=True)
class ScheduleGraphView:
    """Minimal deterministic graph/time view required by ITR."""

    intervals: tuple[ScheduledInterval, ...]
    target_durations: Mapping[tuple[str, str], float]
    predecessors: Mapping[str, tuple[str, ...]]
    successors: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_problem_schedule(
        cls, problem: Problem, schedule: Schedule
    ) -> "ScheduleGraphView":
        mode_map = problem.mode_map()
        intervals: list[ScheduledInterval] = []
        for assignment in schedule.assignments:
            mode = mode_map[assignment.mode_id][1]
            if len(mode.resources) != 1:
                raise ValueError("ITR v1 requires unary-resource operation modes")
            intervals.append(ScheduledInterval(
                operation_id=assignment.operation_id,
                machine_id=mode.resources[0],
                start=float(assignment.start),
                end=float(assignment.end),
            ))
        target_durations: dict[tuple[str, str], float] = {}
        predecessors: dict[str, tuple[str, ...]] = {}
        successor_lists: dict[str, list[str]] = {op.id: [] for op in problem.operations}
        for operation in problem.operations:
            predecessors[operation.id] = tuple(operation.predecessors)
            for predecessor in operation.predecessors:
                successor_lists.setdefault(predecessor, []).append(operation.id)
            for mode in operation.modes:
                if len(mode.resources) == 1:
                    key = (operation.id, mode.resources[0])
                    previous = target_durations.get(key)
                    duration = float(mode.duration)
                    target_durations[key] = duration if previous is None else min(previous, duration)
        return cls(
            intervals=tuple(sorted(intervals, key=lambda row: (row.machine_id, row.start, row.operation_id))),
            target_durations=target_durations,
            predecessors=predecessors,
            successors={key: tuple(sorted(value)) for key, value in successor_lists.items()},
        )

    def interval_for(self, operation_id: str) -> ScheduledInterval | None:
        return next((row for row in self.intervals if row.operation_id == operation_id), None)

    def duration_for(self, operation_id: str, machine_id: str) -> float | None:
        return self.target_durations.get((operation_id, machine_id))


@dataclass(frozen=True)
class RootDecisionRequest:
    operation_id: str
    desired_edit_ids: tuple[str, ...] = ()
    candidate_machines: tuple[str, ...] = ()
    decision_site: object | None = None

    @classmethod
    def from_site_edit(cls, site: object, edit: LegalEdit) -> "RootDecisionRequest":
        return cls(
            operation_id=edit.operation_id,
            desired_edit_ids=(edit.edit_id,),
            candidate_machines=((edit.target_machine,) if edit.target_machine else ()),
            decision_site=site,
        )


@dataclass(frozen=True)
class FeasibilityCheck:
    eligibility: bool
    temporal_fit: bool
    acyclic: bool
    blockers: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.eligibility and self.temporal_fit and self.acyclic


@dataclass(frozen=True)
class TransitionDependency:
    before: str
    after: str
    reason: str


@dataclass(frozen=True)
class InterventionChain:
    actions: tuple[LegalEdit, ...]
    dependencies: tuple[TransitionDependency, ...]
    explanation: tuple[str, ...]
    score: float
    complete: bool
    depth: int
    stop_reason: str = ""


@dataclass(frozen=True)
class TransitionExpansionResult:
    chains: tuple[InterventionChain, ...]
    incomplete_chains: tuple[InterventionChain, ...]
    explored_states: int
    max_depth: int
    recursive_expansion: bool


@dataclass(frozen=True)
class _SearchState:
    actions: tuple[LegalEdit, ...]
    dependencies: tuple[TransitionDependency, ...]
    explanation: tuple[str, ...]
    pending: tuple[str, ...]
    resolved: frozenset[str]
    score: float = 0.0
    stop_reason: str = ""


class InterventionTransitionReasoner:
    """Deterministic bounded beam search over legal blocker relocations."""

    def __init__(self, config: InterventionTransitionConfig | None = None) -> None:
        self.config = config or load_intervention_transition_config()
        self.config.validate()

    def expand(
        self,
        root_decision,
        legal_edits: Sequence[LegalEdit],
        schedule_graph: ScheduleGraphView,
        depth: int | None = None,
    ) -> TransitionExpansionResult:
        """Expand one root request into complete macro intervention chains."""
        edits = tuple(legal_edits)
        for edit in edits:
            edit.validate()
        request = self._normalize_root(root_decision)
        roots = self._root_actions(request, edits)
        limit = int(depth if depth is not None else self.config.max_depth)
        if limit < 1:
            raise ValueError("ITR depth must be >= 1")
        limit = min(limit, self.config.max_chain_length)
        beam = [
            _SearchState(
                actions=(root,), dependencies=(), explanation=(),
                pending=(root.edit_id,), resolved=frozenset(),
            )
            for root in roots
        ]
        complete: list[InterventionChain] = []
        incomplete: list[InterventionChain] = []
        explored = 0
        max_iterations = self.config.max_chain_length * max(1, limit)
        iteration = 0
        while beam and iteration < max_iterations:
            iteration += 1
            expanded: list[_SearchState] = []
            for state in beam:
                explored += 1
                if not state.pending:
                    complete.append(self._as_chain(state, complete=True))
                    continue
                action = self._action_by_id(state.actions, state.pending[0])
                check = self.check_feasibility(
                    action, edits, schedule_graph, resolved=state.resolved,
                    planned_actions=state.actions,
                )
                if check.feasible:
                    resolved = frozenset((*state.resolved, action.edit_id))
                    next_state = replace(
                        state, pending=state.pending[1:], resolved=resolved,
                    )
                    expanded.append(replace(next_state, score=self._score(next_state, edits)))
                    continue
                if not check.eligibility or not check.acyclic:
                    incomplete.append(self._as_chain(
                        replace(state, stop_reason="ineligible_or_cyclic"), complete=False
                    ))
                    continue
                if not self.config.enable_recursive_expansion:
                    incomplete.append(self._as_chain(
                        replace(state, stop_reason="recursive_expansion_disabled"), complete=False
                    ))
                    continue
                relocations = self._blocker_relocations(
                    check.blockers, action, edits, state, schedule_graph
                )
                if not relocations:
                    incomplete.append(self._as_chain(
                        replace(state, stop_reason="no_legal_blocker_relocation"), complete=False
                    ))
                    continue
                for blocker, relocation in relocations:
                    if len(state.actions) >= limit or len(state.actions) >= self.config.max_chain_length:
                        incomplete.append(self._as_chain(
                            replace(state, stop_reason="max_depth"), complete=False
                        ))
                        continue
                    action_score = self._single_action_score(relocation, edits)
                    if action_score < self.config.min_action_score:
                        continue
                    actions = self._insert_before(state.actions, relocation, action.edit_id)
                    dependency = TransitionDependency(
                        before=relocation.edit_id,
                        after=action.edit_id,
                        reason="release_machine",
                    )
                    target = action.target_machine or action.resource_id or "target resource"
                    explanation = state.explanation + (
                        f"{target} unavailable because occupied by {blocker}",
                        f"{blocker} can relocate via {relocation.edit_id}",
                    )
                    next_state = _SearchState(
                        actions=actions,
                        dependencies=state.dependencies + (dependency,),
                        explanation=explanation,
                        pending=(relocation.edit_id,) + state.pending,
                        resolved=state.resolved,
                    )
                    expanded.append(replace(next_state, score=self._score(next_state, edits)))
            deduped: dict[tuple[str, ...], _SearchState] = {}
            for state in expanded:
                key = tuple(action.edit_id for action in state.actions)
                old = deduped.get(key)
                if old is None or state.score > old.score:
                    deduped[key] = state
            beam = sorted(
                deduped.values(),
                key=lambda state: (-state.score, tuple(a.edit_id for a in state.actions)),
            )[: self.config.beam_width]
        for state in beam:
            if not state.pending:
                complete.append(self._as_chain(state, complete=True))
            else:
                incomplete.append(self._as_chain(
                    replace(state, stop_reason=state.stop_reason or "search_budget"),
                    complete=False,
                ))
        complete = self._unique_chains(complete)
        incomplete = self._unique_chains(incomplete)
        return TransitionExpansionResult(
            chains=tuple(sorted(complete, key=lambda chain: (-chain.score, tuple(a.edit_id for a in chain.actions)))),
            incomplete_chains=tuple(incomplete),
            explored_states=explored,
            max_depth=limit,
            recursive_expansion=self.config.enable_recursive_expansion,
        )


    def check_feasibility(
        self,
        action: LegalEdit,
        legal_edits: Sequence[LegalEdit],
        schedule_graph: ScheduleGraphView,
        *,
        resolved: frozenset[str] = frozenset(),
        planned_actions: Sequence[LegalEdit] = (),
    ) -> FeasibilityCheck:
        legal_ids = {edit.edit_id for edit in legal_edits}
        eligibility = action.edit_id in legal_ids
        if action.edit_type != EDIT_ROUTE:
            return FeasibilityCheck(
                eligibility=eligibility, temporal_fit=True, acyclic=eligibility,
            )
        if action.target_machine is None:
            return FeasibilityCheck(False, False, False)
        interval = schedule_graph.interval_for(action.operation_id)
        duration = schedule_graph.duration_for(action.operation_id, action.target_machine)
        if interval is None or duration is None:
            return FeasibilityCheck(eligibility, False, eligibility)
        # T1-REASONER-FIX-R1: ``temporal_fit`` computed here is a SEARCH signal
        # that triggers blocker-relocation (enabling-chain) expansion -- NOT a
        # proposal-level hard reject.  The window is pinned to the operation's
        # CURRENT start, which is more conservative than the authoritative
        # Frozen-Local executor: CP-SAT re-times the whole local schedule and
        # can fit an edit whose current-window precheck finds occupants.
        # RoutingOperator therefore emits an exact single-action proposal for a
        # legal ROUTE edit whenever this search cannot assemble a complete
        # chain; actual schedulability is decided there / by the executor.
        window_start = interval.start
        window_end = window_start + duration
        blockers = self._discover_blockers(
            action, window_start, window_end, schedule_graph,
            resolved=resolved, planned_actions=planned_actions,
        )
        # T1-JOINT-LEGALITY: ``acyclic`` is now a REAL composite cycle check
        # (job precedence + machine order, Kahn topological sort), not the
        # ``eligibility`` placeholder.  ``planned_actions`` is the full already-
        # planned chain (including the pending action), so a 2-edit enabling
        # chain whose relocation + dependent order forms a directed cycle with
        # job precedence is rejected here instead of reaching the executor.
        acyclic = self._composite_acyclic(planned_actions, schedule_graph)
        return FeasibilityCheck(
            eligibility=eligibility,
            temporal_fit=not blockers,
            acyclic=acyclic,
            blockers=blockers,
        )

    @staticmethod
    def _composite_acyclic(
        planned_actions: Sequence[LegalEdit],
        schedule_graph: ScheduleGraphView,
    ) -> bool:
        """True composite acyclicity of the planned chain (Kahn on the composed
        machine orders + job precedence), not an ``eligibility`` alias."""
        if not planned_actions:
            return True
        if any(edit.edit_type != EDIT_ROUTE for edit in planned_actions):
            # The composite checker is ROUTE-only this round; defer to eligibility
            # for a mixed/non-ROUTE chain (never silently mark it cyclic).
            return True
        # Lazy import to avoid a module-load circular import (composite_legality
        # imports ScheduleGraphView from this module).
        from .composite_legality import check_composite_structural_legality

        result = check_composite_structural_legality(schedule_graph, tuple(planned_actions))
        return result.legal

    def _discover_blockers(
        self,
        action: LegalEdit,
        window_start: float,
        window_end: float,
        graph: ScheduleGraphView,
        *,
        resolved: frozenset[str],
        planned_actions: Sequence[LegalEdit],
    ) -> tuple[str, ...]:
        target = action.target_machine
        if target is None:
            return ()
        resolved_routes = {
            edit.operation_id: edit
            for edit in planned_actions
            if edit.edit_type == EDIT_ROUTE and edit.edit_id in resolved
        }
        active: list[ScheduledInterval] = []
        for row in graph.intervals:
            moved = resolved_routes.get(row.operation_id)
            if moved is not None:
                if moved.target_machine != target:
                    continue
                duration = graph.duration_for(row.operation_id, target)
                if duration is None:
                    continue
                row = ScheduledInterval(row.operation_id, target, row.start, row.start + duration)
            if row.machine_id == target and row.operation_id != action.operation_id:
                active.append(row)
        active.sort(key=lambda row: (row.start, row.end, row.operation_id))
        overlapping = [
            index for index, row in enumerate(active)
            if row.start < window_end and window_start < row.end
        ]
        if not overlapping:
            return ()
        radius = self.config.blocker_search_radius
        selected: set[int] = set()
        for index in overlapping:
            start = max(0, index - (radius - 1))
            end = min(len(active), index + radius)
            selected.update(range(start, end))
        return tuple(active[index].operation_id for index in sorted(selected))

    _MAX_RELOCATION_ALTERNATIVES = 5

    def _blocker_relocations(
        self,
        blockers: Sequence[str],
        dependent: LegalEdit,
        legal_edits: Sequence[LegalEdit],
        state: _SearchState,
        schedule_graph: ScheduleGraphView,
    ) -> list[tuple[str, LegalEdit]]:
        existing = {action.operation_id for action in state.actions}
        candidates: list[tuple[str, LegalEdit]] = []
        for blocker in blockers:
            if blocker in existing:
                continue
            for edit in legal_edits:
                if (
                    edit.edit_type == EDIT_ROUTE
                    and edit.operation_id == blocker
                    and edit.source_machine == dependent.target_machine
                    and edit.target_machine != dependent.target_machine
                ):
                    candidates.append((blocker, edit))
        # T1-JOINT-LEGALITY: alternative search -- keep only relocations whose
        # composition with the already-planned chain is deterministically
        # acyclic, and bound the surviving alternatives to N<=5.
        acyclic_candidates: list[tuple[str, LegalEdit]] = []
        for blocker, edit in candidates:
            if self._composite_acyclic((edit,) + state.actions, schedule_graph):
                acyclic_candidates.append((blocker, edit))
        return sorted(
            acyclic_candidates,
            key=lambda item: (-self._single_action_score(item[1], legal_edits), item[1].edit_id),
        )[: self._MAX_RELOCATION_ALTERNATIVES]

    def _score(self, state: _SearchState, legal_edits: Sequence[LegalEdit]) -> float:
        count = max(1, len(state.actions))
        feasibility = len(state.resolved) / count
        actionability = sum(self._actionability(action, legal_edits) for action in state.actions) / count
        expected_impact = sum(self._expected_impact(action) for action in state.actions) / count
        return (
            self.config.feasibility_weight * feasibility
            + self.config.actionability_weight * actionability
            + self.config.expected_impact_weight * expected_impact
            - self.config.length_weight * len(state.actions)
        )

    def _single_action_score(
        self, action: LegalEdit, legal_edits: Sequence[LegalEdit]
    ) -> float:
        return (
            self.config.actionability_weight * self._actionability(action, legal_edits)
            + self.config.expected_impact_weight * self._expected_impact(action)
            - self.config.length_weight
        )

    @staticmethod
    def _actionability(action: LegalEdit, legal_edits: Sequence[LegalEdit]) -> float:
        alternatives = sum(
            edit.edit_type == EDIT_ROUTE and edit.operation_id == action.operation_id
            for edit in legal_edits
        )
        return 1.0 if alternatives else 0.0

    @staticmethod
    def _expected_impact(action: LegalEdit) -> float:
        features = dict(action.features)
        return max(0.0, -float(features.get("delta_p", 0.0)))

    @staticmethod
    def _normalize_root(root_decision) -> RootDecisionRequest:
        if isinstance(root_decision, RootDecisionRequest):
            return root_decision
        if isinstance(root_decision, LegalEdit):
            return RootDecisionRequest(
                operation_id=root_decision.operation_id,
                desired_edit_ids=(root_decision.edit_id,),
                candidate_machines=((root_decision.target_machine,) if root_decision.target_machine else ()),
            )
        if isinstance(root_decision, Mapping):
            operation_id = root_decision.get("operation") or root_decision.get("operation_id")
            if not operation_id:
                raise ValueError("root decision mapping requires operation")
            candidates = root_decision.get("candidate_machines", ())
            desired = root_decision.get("desired_edit_ids", ())
            return RootDecisionRequest(
                operation_id=str(operation_id),
                desired_edit_ids=tuple(str(value) for value in desired),
                candidate_machines=tuple(str(value) for value in candidates),
                decision_site=root_decision,
            )
        operation_id = getattr(root_decision, "operation_id", None)
        if operation_id is None:
            raise TypeError("unsupported root decision input")
        candidate = getattr(root_decision, "target_machine", None)
        return RootDecisionRequest(
            operation_id=str(operation_id),
            candidate_machines=((str(candidate),) if candidate else ()),
            decision_site=root_decision,
        )

    @staticmethod
    def _root_actions(
        request: RootDecisionRequest, legal_edits: Sequence[LegalEdit]
    ) -> tuple[LegalEdit, ...]:
        desired = set(request.desired_edit_ids)
        machines = set(request.candidate_machines)
        actions = tuple(
            edit for edit in legal_edits
            if edit.operation_id == request.operation_id
            and (not desired or edit.edit_id in desired)
            and (not machines or edit.target_machine in machines)
        )
        if not actions:
            raise ValueError(f"root decision {request.operation_id} has no matching legal edit")
        return tuple(sorted(actions, key=lambda edit: edit.edit_id))

    @staticmethod
    def _action_by_id(actions: Sequence[LegalEdit], edit_id: str) -> LegalEdit:
        return next(action for action in actions if action.edit_id == edit_id)

    @staticmethod
    def _insert_before(
        actions: Sequence[LegalEdit], new_action: LegalEdit, dependent_id: str
    ) -> tuple[LegalEdit, ...]:
        if any(action.edit_id == new_action.edit_id for action in actions):
            return tuple(actions)
        output: list[LegalEdit] = []
        for action in actions:
            if action.edit_id == dependent_id:
                output.append(new_action)
            output.append(action)
        return tuple(output)

    def _as_chain(self, state: _SearchState, *, complete: bool) -> InterventionChain:
        return InterventionChain(
            actions=state.actions,
            dependencies=state.dependencies,
            explanation=state.explanation,
            score=self._score(state, state.actions),
            complete=complete,
            depth=len(state.actions),
            stop_reason=state.stop_reason,
        )

    @staticmethod
    def _unique_chains(chains: Sequence[InterventionChain]) -> list[InterventionChain]:
        unique: dict[tuple[str, ...], InterventionChain] = {}
        for chain in chains:
            key = tuple(action.edit_id for action in chain.actions)
            old = unique.get(key)
            if old is None or chain.score > old.score:
                unique[key] = chain
        return list(unique.values())


# Public role name after the Causal Explorer refactor.  The historical class
# remains import-compatible, but this reasoner now lives inside RoutingOperator.
TransitionReasoner = InterventionTransitionReasoner


__all__ = [
    "FeasibilityCheck",
    "InterventionChain",
    "InterventionTransitionConfig",
    "InterventionTransitionReasoner",
    "TransitionReasoner",
    "RootDecisionRequest",
    "ScheduleGraphView",
    "TransitionDependency",
    "TransitionExpansionResult",
    "load_intervention_transition_config",
]
