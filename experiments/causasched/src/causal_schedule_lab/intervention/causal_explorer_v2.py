"""True bounded causal beam search for SGSCT V5.

The explorer answers *why* an Appearance may have arisen.  It never generates
actions and never calls CP-SAT.  M2 candidate probabilities seed a deterministic
beam which is expanded through explicit :class:`CauseEdge` records.  Actionable
nodes are retained as root candidates, but search continues until the real
depth bound is reached or no state can be expanded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from ..m2_v5_schema_v1 import (
    EDIT_ROUTE,
    EDIT_SEQ_INSERT,
    EDIT_SEQ_SWAP,
    EDIT_TIMING_SHIFT,
    LegalEdit,
)
from .transition_reasoner_v1 import ScheduleGraphView


REL_PREDECESSOR = "predecessor"
REL_RESOURCE_CONFLICT = "resource_conflict"
REL_RESOURCE_BLOCKER = "resource_blocker"
REL_ROUTING_DEPENDENCY = "routing_dependency"
REL_CONSTRAINT_DEPENDENCY = "constraint_dependency"
REL_SEQUENCING_DEPENDENCY = "sequencing_dependency"
CAUSE_RELATIONS = (
    REL_PREDECESSOR,
    REL_RESOURCE_CONFLICT,
    REL_RESOURCE_BLOCKER,
    REL_ROUTING_DEPENDENCY,
    REL_CONSTRAINT_DEPENDENCY,
    REL_SEQUENCING_DEPENDENCY,
)


def _repository_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "hyperparameters.yaml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


def _hyperparameters(path: str | Path | None = None) -> Mapping[str, object]:
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or (_repository_root() / "configs" / "hyperparameters.yaml")
    ).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "causal_schedule_lab_hyperparameters_v1":
        raise ValueError("unsupported hyperparameter schema")
    return payload


@dataclass(frozen=True)
class CausalExplorerConfig:
    search_mode: str
    max_depth: int
    beam_width: int
    min_score: float
    depth_penalty: float
    relevance_weight: float
    causal_strength_weight: float
    actionability_weight: float
    explanation_gain_weight: float
    relation_weights: Mapping[str, float]

    def validate(self) -> None:
        if self.search_mode != "beam":
            raise ValueError("causal_explorer.search_mode must be 'beam'")
        if self.max_depth < 1 or self.beam_width < 1:
            raise ValueError("causal_explorer depth and beam width must be positive")
        if self.depth_penalty < 0:
            raise ValueError("causal_explorer.depth_penalty must be non-negative")
        missing = set(CAUSE_RELATIONS) - set(self.relation_weights)
        if missing:
            raise ValueError(f"missing causal relation weights: {sorted(missing)}")


def load_causal_explorer_config(path: str | Path | None = None) -> CausalExplorerConfig:
    section = _hyperparameters(path).get("causal_explorer")
    if not isinstance(section, Mapping):
        raise ValueError("missing causal_explorer hyperparameters")
    weights = section.get("score_weights")
    relations = section.get("relation_weights")
    if not isinstance(weights, Mapping) or not isinstance(relations, Mapping):
        raise ValueError("causal_explorer score/relation weights are required")
    config = CausalExplorerConfig(
        search_mode=str(section["search_mode"]),
        max_depth=int(section["max_depth"]),
        beam_width=int(section["beam_width"]),
        min_score=float(section["min_score"]),
        depth_penalty=float(section["depth_penalty"]),
        relevance_weight=float(weights["relevance"]),
        causal_strength_weight=float(weights["causal_strength"]),
        actionability_weight=float(weights["actionability"]),
        explanation_gain_weight=float(weights["explanation_gain"]),
        relation_weights={name: float(relations[name]) for name in CAUSE_RELATIONS},
    )
    config.validate()
    return config


@dataclass(frozen=True)
class ActionableRootSelectorConfig:
    top_k: int
    operator_availability_weight: float
    expected_impact_weight: float
    risk_weight: float


def load_actionable_root_selector_config(
    path: str | Path | None = None,
) -> ActionableRootSelectorConfig:
    section = _hyperparameters(path).get("actionable_root_selector")
    if not isinstance(section, Mapping):
        raise ValueError("missing actionable_root_selector hyperparameters")
    return ActionableRootSelectorConfig(
        top_k=int(section["top_k"]),
        operator_availability_weight=float(section["operator_availability_weight"]),
        expected_impact_weight=float(section["expected_impact_weight"]),
        risk_weight=float(section["risk_weight"]),
    )


@dataclass(frozen=True)
class AppearanceContext:
    appearance_id: str
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionCandidate:
    decision_site: object
    relevance: float


@dataclass(frozen=True)
class CauseEdge:
    source: str
    target: str
    relation_type: str
    strength: float
    evidence: str

    def validate(self) -> None:
        if self.relation_type not in CAUSE_RELATIONS:
            raise ValueError(f"unknown CauseEdge relation: {self.relation_type}")
        if not self.source or not self.target or not self.evidence:
            raise ValueError("CauseEdge requires source, target and evidence")


@dataclass(frozen=True)
class CausalSearchTrace:
    appearance_id: str
    visited_nodes: tuple[str, ...]
    discarded_paths: tuple[str, ...]
    selected_root: str
    reason: str

    def as_tokens(self) -> tuple[str, ...]:
        return (
            f"appearance={self.appearance_id}",
            *(f"visited={node}" for node in self.visited_nodes),
            *(f"discarded={path}" for path in self.discarded_paths),
            f"selected_root={self.selected_root}",
            f"reason={self.reason}",
        )


@dataclass(frozen=True)
class CausalExplanationChain:
    appearance_id: str
    decision_site_id: str
    operation_id: str
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    explanations: tuple[str, ...]
    causal_score: float
    m2_root_score: float
    actionable: bool
    operator_types: tuple[str, ...]
    depth: int
    cause_edges: tuple[CauseEdge, ...] = ()
    seed_decision_site_id: str = ""
    root_candidate_id: str = ""
    search_trace: CausalSearchTrace | None = None


@dataclass(frozen=True)
class ActionableRoot:
    appearance_id: str
    decision_site: object
    causal_chain: CausalExplanationChain
    operator_types: tuple[str, ...]
    operator_availability: float
    expected_impact: float
    risk: float
    selector_score: float
    selection_reason: str
    causal_search_trace: CausalSearchTrace


@dataclass(frozen=True)
class _SearchState:
    seed: DecisionCandidate
    current_node: str
    causal_path: tuple[str, ...]
    edges: tuple[CauseEdge, ...]
    explanations: tuple[str, ...]
    depth: int
    cumulative_score: float


def _matching_edits_for_operation(
    operation_id: str, edits: Sequence[LegalEdit]
) -> tuple[LegalEdit, ...]:
    return tuple(edit for edit in edits if edit.operation_id == operation_id)


def _operator_types(edits: Sequence[LegalEdit]) -> tuple[str, ...]:
    mapping = {
        EDIT_ROUTE: "routing",
        EDIT_SEQ_SWAP: "sequencing",
        EDIT_SEQ_INSERT: "insertion",
        EDIT_TIMING_SHIFT: "timing",
    }
    return tuple(dict.fromkeys(mapping[edit.edit_type] for edit in edits if edit.edit_type in mapping))


class CausalExplorerV2:
    """Layer-wise deterministic beam expansion over auditable CauseEdges."""

    def __init__(self, config: CausalExplorerConfig | None = None) -> None:
        self.config = config or load_causal_explorer_config()
        self.config.validate()

    def explore(
        self,
        appearance: AppearanceContext,
        decision_candidates: Sequence[DecisionCandidate],
        schedule_graph: ScheduleGraphView,
        legal_edits: Sequence[LegalEdit],
        config: CausalExplorerConfig | None = None,
    ) -> tuple[CausalExplanationChain, ...]:
        cfg = config or self.config
        cfg.validate()
        edits = tuple(legal_edits)
        for edit in edits:
            edit.validate()
        relevance = self._operation_relevance(decision_candidates)
        initial: list[_SearchState] = []
        for candidate in decision_candidates:
            site = candidate.decision_site
            operation = str(getattr(site, "operation_id"))
            actionable = bool(_operator_types(_matching_edits_for_operation(operation, edits)))
            score = (
                cfg.relevance_weight * float(candidate.relevance)
                + cfg.actionability_weight * float(actionable)
                - cfg.depth_penalty
            )
            if score >= cfg.min_score:
                initial.append(_SearchState(
                    seed=candidate,
                    current_node=operation,
                    causal_path=(
                        f"appearance:{appearance.appearance_id}",
                        f"decision:{getattr(site, 'site_id')}",
                        operation,
                    ),
                    edges=(),
                    explanations=(
                        f"{appearance.appearance_id} links to decision {getattr(site, 'site_id')}",
                    ),
                    depth=1,
                    cumulative_score=float(score),
                ))

        initial.sort(key=self._state_order)
        discarded: list[str] = []
        if len(initial) > cfg.beam_width:
            discarded.extend(self._discard_token(state, "initial_beam_pruned")
                             for state in initial[cfg.beam_width:])
        beam = initial[: cfg.beam_width]
        visited: list[str] = []
        actionable_states: dict[tuple[str, tuple[str, ...]], _SearchState] = {}

        while beam:
            next_states: list[_SearchState] = []
            for state in beam:
                if state.current_node not in visited:
                    visited.append(state.current_node)
                if self._is_actionable(state.current_node, edits):
                    actionable_states[(self._site_id(state.seed), state.causal_path)] = state
                if state.depth >= cfg.max_depth:
                    discarded.append(self._discard_token(state, "max_depth"))
                    continue
                expansions = self._expand_state(
                    state, appearance, schedule_graph, edits, relevance, cfg
                )
                if not expansions:
                    discarded.append(self._discard_token(state, "no_further_cause"))
                    continue
                for child in expansions:
                    if child.current_node in state.causal_path:
                        discarded.append(self._discard_token(child, "cycle"))
                        continue
                    if child.cumulative_score < cfg.min_score:
                        discarded.append(self._discard_token(child, "below_min_score"))
                        continue
                    if child.current_node not in visited:
                        visited.append(child.current_node)
                    if self._is_actionable(child.current_node, edits):
                        actionable_states[(self._site_id(child.seed), child.causal_path)] = child
                    next_states.append(child)

            deduped: dict[tuple[str, tuple[str, ...]], _SearchState] = {}
            for state in next_states:
                key = (self._site_id(state.seed), state.causal_path)
                old = deduped.get(key)
                if old is None or state.cumulative_score > old.cumulative_score:
                    deduped[key] = state
            ranked = sorted(deduped.values(), key=self._state_order)
            if len(ranked) > cfg.beam_width:
                discarded.extend(self._discard_token(state, "beam_pruned")
                                 for state in ranked[cfg.beam_width:])
            beam = ranked[: cfg.beam_width]

        common_visited = tuple(visited)
        common_discarded = tuple(dict.fromkeys(discarded))
        chains = [
            self._as_chain(
                state, appearance.appearance_id, edits,
                CausalSearchTrace(
                    appearance_id=appearance.appearance_id,
                    visited_nodes=common_visited,
                    discarded_paths=common_discarded,
                    selected_root=state.current_node,
                    reason="actionable_candidate_retained_search_continued_to_bound",
                ),
            )
            for state in actionable_states.values()
        ]
        chains.sort(key=lambda chain: (-chain.causal_score, chain.depth,
                                       chain.root_candidate_id, chain.decision_site_id))
        return tuple(chains)

    def _expand_state(
        self,
        state: _SearchState,
        appearance: AppearanceContext,
        graph: ScheduleGraphView,
        edits: Sequence[LegalEdit],
        relevance: Mapping[str, float],
        cfg: CausalExplorerConfig,
    ) -> list[_SearchState]:
        if state.current_node.startswith("machine:") or state.current_node.startswith("constraint:"):
            return []
        edges = self._cause_edges(
            state.current_node, state.seed.decision_site, appearance, graph, edits, cfg
        )
        children: list[_SearchState] = []
        seen_targets: set[tuple[str, str]] = set()
        for edge in edges:
            edge.validate()
            key = (edge.target, edge.relation_type)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            new_depth = state.depth + 1
            actionable = self._is_actionable(edge.target, edits)
            relation_new = edge.relation_type not in {item.relation_type for item in state.edges}
            explanation_gain = (
                float(relation_new)
                + 0.5 * float(edge.target in appearance.members)
                + 0.25 * float(bool(edge.evidence))
            )
            score = (
                state.cumulative_score
                + cfg.relevance_weight * float(relevance.get(edge.target, 0.0))
                + cfg.causal_strength_weight * edge.strength
                + cfg.actionability_weight * float(actionable)
                + cfg.explanation_gain_weight * explanation_gain
                - cfg.depth_penalty
            )
            children.append(_SearchState(
                seed=state.seed,
                current_node=edge.target,
                causal_path=state.causal_path + (edge.target,),
                edges=state.edges + (edge,),
                explanations=state.explanations + (edge.evidence,),
                depth=new_depth,
                cumulative_score=float(score),
            ))
        return children

    def _cause_edges(
        self,
        operation: str,
        seed_site: object,
        appearance: AppearanceContext,
        graph: ScheduleGraphView,
        edits: Sequence[LegalEdit],
        cfg: CausalExplorerConfig,
    ) -> list[CauseEdge]:
        rows: list[CauseEdge] = []

        for predecessor in graph.predecessors.get(operation, ()):
            rows.append(self._edge(
                operation, predecessor, REL_PREDECESSOR,
                f"{predecessor} is a job-precedence cause of {operation}", cfg,
            ))

        interval = graph.interval_for(operation)
        if interval is not None:
            sequence = sorted(
                (row for row in graph.intervals if row.machine_id == interval.machine_id),
                key=lambda row: (row.start, row.end, row.operation_id),
            )
            index = next((i for i, row in enumerate(sequence) if row.operation_id == operation), None)
            if index is not None:
                for neighbor in sequence[max(0, index - 1):index] + sequence[index + 1:index + 2]:
                    rows.append(self._edge(
                        operation, neighbor.operation_id, REL_SEQUENCING_DEPENDENCY,
                        f"{neighbor.operation_id} is adjacent to {operation} on {interval.machine_id}", cfg,
                    ))

        operation_edits = _matching_edits_for_operation(operation, edits)
        for edit in operation_edits:
            if edit.edit_type == EDIT_ROUTE and edit.target_machine:
                current = graph.interval_for(operation)
                duration = graph.duration_for(operation, edit.target_machine)
                if current is None or duration is None:
                    continue
                blockers = self._overlaps(
                    graph, edit.target_machine, current.start, current.start + duration,
                    exclude={operation},
                )
                if blockers:
                    for blocker in blockers:
                        rows.append(self._edge(
                            operation, blocker, REL_RESOURCE_BLOCKER,
                            f"{operation} routing to {edit.target_machine} is blocked by {blocker}"
                            f" during [{current.start:g},{current.start + duration:g}) via {edit.edit_id}", cfg,
                        ))
                else:
                    rows.append(self._edge(
                        operation, f"machine:{edit.target_machine}", REL_ROUTING_DEPENDENCY,
                        f"{edit.edit_id} has a clear target interval on {edit.target_machine}", cfg,
                    ))

            elif edit.edit_type == EDIT_SEQ_SWAP:
                other = edit.right_id if edit.left_id == operation else edit.left_id
                if other:
                    rows.append(self._edge(
                        operation, other, REL_SEQUENCING_DEPENDENCY,
                        f"{edit.edit_id} couples the order of {operation} and {other}", cfg,
                    ))

            elif edit.edit_type == EDIT_SEQ_INSERT and edit.resource_id:
                duration = self._current_duration(operation, graph)
                predecessor = graph.interval_for(edit.predecessor_id) if edit.predecessor_id else None
                successor = graph.interval_for(edit.successor_id) if edit.successor_id else None
                start = predecessor.end if predecessor is not None else 0.0
                end = start + duration
                blockers = self._overlaps(
                    graph, edit.resource_id, start, end,
                    exclude={operation, edit.predecessor_id or ""},
                )
                if successor is not None and successor.start < end and successor.operation_id not in blockers:
                    blockers.append(successor.operation_id)
                for blocker in blockers:
                    rows.append(self._edge(
                        operation, blocker, REL_RESOURCE_CONFLICT,
                        f"{edit.edit_id} cannot fit [{start:g},{end:g}) because {blocker} occupies the insertion window",
                        cfg,
                    ))
                for predecessor_id in graph.predecessors.get(operation, ()):
                    pred = graph.interval_for(predecessor_id)
                    if pred is not None and pred.end > start:
                        rows.append(self._edge(
                            operation, predecessor_id, REL_CONSTRAINT_DEPENDENCY,
                            f"precedence constraint {predecessor_id}->{operation} closes insertion window for {edit.edit_id}",
                            cfg,
                        ))

            elif edit.edit_type == EDIT_TIMING_SHIFT and edit.target_start is not None:
                duration = self._current_duration(operation, graph)
                blockers = self._overlaps(
                    graph, edit.resource_id or "", edit.target_start,
                    edit.target_start + duration, exclude={operation},
                )
                for blocker in blockers:
                    rows.append(self._edge(
                        operation, blocker, REL_CONSTRAINT_DEPENDENCY,
                        f"{edit.edit_id} timing window is constrained by {blocker}", cfg,
                    ))

        # Prefer the strongest evidence when two mechanisms identify the same
        # target.  This keeps the beam deterministic without hiding relation type.
        unique: dict[tuple[str, str], CauseEdge] = {}
        for edge in rows:
            key = (edge.target, edge.relation_type)
            old = unique.get(key)
            if old is None or edge.strength > old.strength:
                unique[key] = edge
        return sorted(unique.values(), key=lambda edge: (-edge.strength, edge.relation_type,
                                                         edge.target, edge.evidence))

    @staticmethod
    def _overlaps(
        graph: ScheduleGraphView,
        machine_id: str,
        start: float,
        end: float,
        *,
        exclude: set[str],
    ) -> list[str]:
        return [
            row.operation_id for row in graph.intervals
            if row.machine_id == machine_id
            and row.operation_id not in exclude
            and row.start < end and start < row.end
        ]

    @staticmethod
    def _current_duration(operation: str, graph: ScheduleGraphView) -> float:
        interval = graph.interval_for(operation)
        return max(0.0, interval.end - interval.start) if interval is not None else 0.0

    @staticmethod
    def _operation_relevance(
        candidates: Sequence[DecisionCandidate],
    ) -> dict[str, float]:
        relevance: dict[str, float] = {}
        for candidate in candidates:
            operation = str(getattr(candidate.decision_site, "operation_id"))
            relevance[operation] = max(relevance.get(operation, 0.0), float(candidate.relevance))
        return relevance

    @staticmethod
    def _is_actionable(node: str, edits: Sequence[LegalEdit]) -> bool:
        return any(edit.operation_id == node for edit in edits)

    @staticmethod
    def _site_id(candidate: DecisionCandidate) -> str:
        return str(getattr(candidate.decision_site, "site_id"))

    @staticmethod
    def _state_order(state: _SearchState) -> tuple[float, int, str, tuple[str, ...]]:
        return (-state.cumulative_score, state.depth,
                str(getattr(state.seed.decision_site, "site_id")), state.causal_path)

    @staticmethod
    def _discard_token(state: _SearchState, reason: str) -> str:
        return f"{reason}:{'->'.join(state.causal_path)}:score={state.cumulative_score:.6f}"

    def _edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        evidence: str,
        config: CausalExplorerConfig,
    ) -> CauseEdge:
        return CauseEdge(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=float(config.relation_weights[relation_type]),
            evidence=evidence,
        )

    @staticmethod
    def _as_chain(
        state: _SearchState,
        appearance_id: str,
        edits: Sequence[LegalEdit],
        trace: CausalSearchTrace,
    ) -> CausalExplanationChain:
        operators = _operator_types(_matching_edits_for_operation(state.current_node, edits))
        if state.edges:
            relation = state.edges[-1].relation_type
            preferred = {
                REL_RESOURCE_BLOCKER: ("routing",),
                REL_ROUTING_DEPENDENCY: ("routing",),
                REL_SEQUENCING_DEPENDENCY: ("sequencing", "insertion"),
                REL_RESOURCE_CONFLICT: ("insertion", "timing", "sequencing"),
                REL_PREDECESSOR: ("timing", "sequencing", "routing"),
                REL_CONSTRAINT_DEPENDENCY: ("timing", "sequencing", "routing"),
            }[relation]
            aligned = tuple(name for name in operators if name in preferred)
            if aligned:
                operators = aligned
        site = state.seed.decision_site
        return CausalExplanationChain(
            appearance_id=appearance_id,
            decision_site_id=str(getattr(site, "site_id")),
            operation_id=state.current_node,
            nodes=state.causal_path,
            relations=("appearance_to_decision", "decision_to_operation")
                      + tuple(edge.relation_type for edge in state.edges),
            explanations=state.explanations,
            causal_score=state.cumulative_score,
            m2_root_score=float(state.seed.relevance),
            actionable=bool(operators),
            operator_types=operators,
            depth=state.depth,
            cause_edges=state.edges,
            seed_decision_site_id=str(getattr(site, "site_id")),
            root_candidate_id=state.current_node,
            search_trace=trace,
        )


class RootEffectPredictor(Protocol):
    """Replaceable pre-operator root-impact interface; not the trained M2.5 API."""

    def predict_root_impact(
        self,
        root_operation_id: str,
        operator_types: Sequence[str],
        chain: CausalExplanationChain,
        edits: Sequence[LegalEdit],
    ) -> float: ...


class HeuristicRootEffectPlaceholder:
    """Current deterministic placeholder until a root-level effect model exists."""

    def predict_root_impact(
        self,
        root_operation_id: str,
        operator_types: Sequence[str],
        chain: CausalExplanationChain,
        edits: Sequence[LegalEdit],
    ) -> float:
        values: list[float] = []
        for edit in _matching_edits_for_operation(root_operation_id, edits):
            features = dict(edit.features)
            if edit.edit_type == EDIT_ROUTE:
                values.append(max(0.0, -float(features.get("delta_p", 0.0))))
            elif edit.edit_type == EDIT_TIMING_SHIFT:
                values.append(max(0.0, -float(features.get("timing_delta", 0.0))))
            else:
                values.append(max(0.0, float(edit.relevance)))
        return max(values, default=0.0)


class ActionableRootSelectorV2:
    """Select the most valuable actionable cause, not the nearest editable node."""

    def __init__(
        self,
        config: ActionableRootSelectorConfig | None = None,
        *,
        effect_predictor: RootEffectPredictor | None = None,
        risk_predictor: Callable[[str, Sequence[LegalEdit], CausalExplanationChain], float] | None = None,
    ) -> None:
        self.config = config or load_actionable_root_selector_config()
        self.effect_predictor = effect_predictor or HeuristicRootEffectPlaceholder()
        self.risk_predictor = risk_predictor or self._heuristic_risk

    def select(
        self,
        chains: Sequence[CausalExplanationChain],
        decision_sites: Sequence[object],
        legal_edits: Sequence[LegalEdit],
    ) -> tuple[ActionableRoot, ...]:
        by_id = {str(getattr(site, "site_id")): site for site in decision_sites}
        roots: list[ActionableRoot] = []
        for chain in chains:
            if not chain.actionable or chain.seed_decision_site_id not in by_id:
                continue
            site = by_id[chain.seed_decision_site_id]
            root_edits = _matching_edits_for_operation(chain.root_candidate_id, legal_edits)
            availability = float(len(chain.operator_types))
            impact = float(self.effect_predictor.predict_root_impact(
                chain.root_candidate_id, chain.operator_types, chain, legal_edits
            ))
            risk = max(0.0, float(self.risk_predictor(
                chain.root_candidate_id, root_edits, chain
            )))
            score = (
                chain.causal_score
                + self.config.operator_availability_weight * availability
                + self.config.expected_impact_weight * impact
                - self.config.risk_weight * risk
            )
            reason = (
                f"causal={chain.causal_score:.6f};operators={availability:.0f};"
                f"expected_impact={impact:.6f};risk={risk:.6f}"
            )
            base_trace = chain.search_trace or CausalSearchTrace(
                chain.appearance_id, (), (), chain.root_candidate_id, ""
            )
            trace = CausalSearchTrace(
                appearance_id=base_trace.appearance_id,
                visited_nodes=base_trace.visited_nodes,
                discarded_paths=base_trace.discarded_paths,
                selected_root=chain.root_candidate_id,
                reason=reason,
            )
            roots.append(ActionableRoot(
                appearance_id=chain.appearance_id,
                decision_site=site,
                causal_chain=chain,
                operator_types=chain.operator_types,
                operator_availability=availability,
                expected_impact=impact,
                risk=risk,
                selector_score=float(score),
                selection_reason=reason,
                causal_search_trace=trace,
            ))
        roots.sort(key=lambda root: (-root.selector_score, root.causal_chain.depth,
                                     root.causal_chain.root_candidate_id,
                                     root.causal_chain.seed_decision_site_id))
        return tuple(roots[: self.config.top_k])

    @staticmethod
    def _heuristic_risk(
        root_operation_id: str,
        edits: Sequence[LegalEdit],
        chain: CausalExplanationChain,
    ) -> float:
        if not edits:
            return 1.0
        risky = 0.0
        for edit in edits:
            features = dict(edit.features)
            if edit.edit_type == EDIT_ROUTE:
                risky = max(risky, max(0.0, float(features.get("delta_p", 0.0))))
            elif edit.edit_type == EDIT_TIMING_SHIFT:
                risky = max(risky, max(0.0, float(features.get("timing_delta", 0.0))))
            elif edit.edit_type in (EDIT_SEQ_SWAP, EDIT_SEQ_INSERT):
                risky = max(risky, 0.25)
        return risky + 0.05 * max(0, chain.depth - 1)


__all__ = [
    "ActionableRoot",
    "ActionableRootSelectorConfig",
    "ActionableRootSelectorV2",
    "AppearanceContext",
    "CAUSE_RELATIONS",
    "CauseEdge",
    "CausalExplanationChain",
    "CausalExplorerConfig",
    "CausalExplorerV2",
    "CausalSearchTrace",
    "DecisionCandidate",
    "HeuristicRootEffectPlaceholder",
    "RootEffectPredictor",
    "load_actionable_root_selector_config",
    "load_causal_explorer_config",
]
