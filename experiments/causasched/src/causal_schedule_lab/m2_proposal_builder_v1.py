"""V5 M2 -- Causal-Intervention Proposal Builder (Phase 7, spec §21-24).

For each appearance block the builder produces a ranked,
diversity-constrained set :math:`R_A=\\{R_1,\\ldots,R_K\\}` with each

.. math:: R_i=(V_i,\\ E_i,\\ D_i,\\ E_i^{edit},\\ E_i^{dep})

* ``V_i`` -- intervention-relevant operation **and** machine nodes (machines
  carry relevance, spec §15/§45-15).
* ``E_i`` -- true structural edges among those nodes (upstream causal path +
  machine assignment / relay).
* ``D_i`` -- the root decision site (a DecisionSite tagged
  ``ROOT_DECISION_CANDIDATE`` by the Phase 6 localizer).
* ``E_i^{edit}`` -- legal editable opportunities for the root operation,
  hard-feasibility enumerated (never hallucinated, spec §17).
* ``E_i^{dep}`` -- structural enabling/dependency between edits (spec §22).

Selection is **Top-K + diversity**: candidate roots are ranked by root score and
a proposal is only kept if its node overlap (Jaccard on ``V_i``) with every
already-selected proposal stays below a threshold (spec §21).

The regressions A-D (spec §39) assert the four appearance archetypes survive:
A1 (compactness), A2-A3 (indirect load relief), A4, and multi-root blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _prop_replace
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

from .ir import Problem, Schedule
from .appearance_taxonomy import is_actionable_appearance, filtered_reason
from .intervention import (
    ActionableRootSelectorV2,
    AppearanceContext,
    CausalExplanationChain,
    CausalExplorerV2,
    DecisionCandidate,
    InterventionOperatorReasoner,
    InterventionTransitionReasoner,
    OperatorStateGraph,
    ScheduleGraphView,
)
from .intervention.effect_predictor_v1 import DeferredProposalEffectAdapter
from .legal_edit_enumerator import LegalEditEnumerator, enumerate_legal_edits
from .m2_root_localizer_v1 import RootDecisionLocalizer, rootness, _reverse_upstream
from .m2_v5_schema_v1 import (
    CausalInterventionProposal,
    DEP_ENABLES,
    EditDependency,
    LegalEdit,
    ROUTING_DECISION,
    SEQUENCE_DECISION,
    TRACE_ROOT,
    proposal_id,
)


@dataclass(frozen=True)
class _Selected:
    nodes: frozenset[str]
    score: float


@dataclass(frozen=True)
class OperatorRuntimeResult:
    """Torch-free audit surface for the Explorer -> Operator runtime path."""

    proposals: tuple[CausalInterventionProposal, ...]
    causal_chains: tuple[CausalExplanationChain, ...]
    actionable_root_ids: tuple[str, ...]
    decision_candidate_scores: tuple[tuple[str, str, float], ...]
    proposal_search_traces: tuple[tuple[str, tuple[str, ...]], ...]
    # T1-REASONER-FIX-R1 provenance: per-proposal root attribution score,
    # parallel to ``proposals`` (index-aligned).  In policy (root_drive) mode
    # this is the exact B5 attribution score that seeded the root; in explorer
    # mode it is the selector's structural score.
    proposal_attribution_scores: tuple[float, ...] = ()
    # Fail-closed appearance gate audit surface (spec: pre-D6 appearance
    # cleanup).  ``filtered_appearance_blocks`` lists ``(block_id, reason)`` for
    # every block dropped before it could become a candidate/root/proposal --
    # deprecated (A5/A7/A8/A9/A10) or unknown ids.  A non-empty list means a
    # stale/out-of-taxonomy artifact was loaded; the runtime stayed actionable
    # only for the active set.
    filtered_appearance_blocks: tuple[tuple[str, str], ...] = ()


class CausalProposalBuilder:
    """Deterministic Top-K + diversity proposal builder per appearance block."""

    def __init__(
        self,
        problem: Problem,
        schedule: Schedule,
        *,
        max_hops: int = 3,
        max_overlap: float = 0.5,
        edit_model=None,
    ) -> None:
        self.problem = problem
        self.schedule = schedule
        self.max_overlap = max_overlap
        self.edit_model = edit_model  # optional V5 model for edit relevance
        self.enum = LegalEditEnumerator(problem, schedule)
        self.localizer = RootDecisionLocalizer(problem, schedule, max_hops=max_hops)
        self._upstream = _reverse_upstream(problem, schedule)

    # -- per-root proposal ----------------------------------------------------

    def _build_proposal(
        self,
        appearance_id: str,
        block_id: str,
        members: Sequence[str],
        root: object,
        rank: int,
        score: float,
    ) -> CausalInterventionProposal:
        root_op = root.operation_id
        actual_machine = self._machine_of(root_op)
        rp = self._root_path(root_op, set(members))

        # V_i = intervention-relevant decision neighborhood: the root op, its
        # machine + alternate (target) machines, and the traced causal-ancestor
        # path ops.  Block-mates that are not causal ancestors of THIS root do
        # not bloat V_i -- that would suppress legitimate diversity (spec §21).
        nodes: list[str] = [root_op, f"machine:{actual_machine}"]
        for m in self.enum.eligible.get(root_op, {}):
            if m != actual_machine:
                nodes.append(f"machine:{m}")
        for p in rp:
            if p != root_op:
                nodes.append(p)
        nodes = list(dict.fromkeys(nodes))

        # E_i: upstream causal edges among included ops + machine relay/assign.
        edges: list[tuple[str, str]] = []
        inc = set(nodes)
        for u in inc:
            for p in self._upstream.get(u, ()):
                if p in inc:
                    edges.append((u, p))  # upstream path
        edges.append((root_op, f"machine:{actual_machine}"))  # assignment
        if root_op in inc:
            for m in self.enum.eligible.get(root_op, {}):
                if m != actual_machine:
                    edges.append((root_op, f"machine:{m}"))  # relay/opportunity

        # D_i; root_decisions
        d_sites = (root,)
        root_decisions = (root,)

        # E_i^edit: legal edits for the root operation (routing), re-scored.
        edits = self._edits_for(root_op)
        edits = self._relevance(edits)

        # E_i^dep: structural ENABLES between edits.
        deps = self._dependencies(edits)

        conf = min(max(0.5 + score / 100.0, 0.05), 0.99)
        return CausalInterventionProposal(
            proposal_id=proposal_id(appearance_id, rank),
            appearance_id=appearance_id,
            nodes=tuple(nodes),
            edges=tuple(edges),
            decision_sites=d_sites,
            root_decisions=root_decisions,
            edits=edits,
            dependencies=deps,
            root_path=rp,
            proposal_score=float(score),
            confidence=float(conf),
            source_block_id=block_id,
        )

    def _edits_for(self, root_op: str) -> tuple[LegalEdit, ...]:
        return enumerate_legal_edits(
            self.problem, self.schedule, [root_op],
            request_seq_insert=False, request_seq_swap=False,
        )

    def _relevance(self, edits: tuple[LegalEdit, ...]) -> tuple[LegalEdit, ...]:
        if not edits:
            return edits
        rels = None
        if self.edit_model is not None:
            rels = self.edit_model.score_edits(list(edits))
        else:
            rels = _heuristic_edit_relevance(edits)
        out = [replace(e, relevance=float(rels[i])) for i, e in enumerate(edits)]
        return tuple(out)

    @staticmethod
    def _dependencies(edits: tuple[LegalEdit, ...]) -> tuple[EditDependency, ...]:
        """Return only executable, intra-proposal vacate-before-receive edges.

        Alternatives for the same operation are mutually exclusive and are
        therefore never dependencies.  ``a`` enables ``b`` only when ``a``
        vacates the machine that ``b`` intends to receive.
        """
        deps: list[EditDependency] = []
        for i, e1 in enumerate(edits):
            if e1.edit_type != "ROUTE" or e1.source_machine is None or e1.target_machine is None:
                continue
            for e2 in edits:
                if e2 == e1 or e2.edit_type != "ROUTE":
                    continue
                if (
                    e1.operation_id != e2.operation_id
                    and e1.source_machine == e2.target_machine
                ):
                    deps.append(EditDependency(
                        dependency_id=f"{e1.edit_id}->{e2.edit_id}",
                        dependency_type=DEP_ENABLES,
                        editor_id=e1.edit_id,
                        dependent_id=e2.edit_id,
                        kind="releases_machine",
                    ))
        # unique by dependency_id, deterministic
        seen = set()
        out = []
        for d in deps:
            if d.dependency_id not in seen:
                seen.add(d.dependency_id)
                out.append(d)
        return tuple(out)

    def _root_path(self, root_op: str, members: set[str]) -> tuple[str, ...]:
        if root_op in members:
            return (root_op,)
        # walk upstream from any member until the root is reached (bounded)
        path: list[str] = []
        frontier: list[str] = list(members)
        seen: set[str] = set()
        while frontier and root_op not in path:
            nxt = frontier.pop(0)
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            if nxt == root_op:
                break
            for p in self._upstream.get(nxt, ()):
                if p not in seen:
                    frontier.insert(0, p)
        if root_op not in path:
            path.append(root_op)
        return tuple(path)

    # -- top-k + diversity ----------------------------------------------------

    def build_block(self, block_id: str, members: Sequence[str]) -> tuple[CausalInterventionProposal, ...]:
        return self.build(
            {block_id: members}, appearance_ids={block_id: block_id},
        )

    def build(
        self,
        blocks: Mapping[str, Sequence[str]],
        *,
        top_k: int = 3,
        appearance_ids: Mapping[str, str] | None = None,
    ) -> tuple[CausalInterventionProposal, ...]:
        """Build proposals for all blocks; each is a Top-K + diversity-ranked set
        flattened against its own block (regression D: multi-root blocks)."""
        proposals: list[CausalInterventionProposal] = []
        for block_id, members in blocks.items():
            appearance_id = (appearance_ids or {}).get(block_id, block_id)
            loc = self.localizer.localize(block_id, members)
            if loc.unresolved or not loc.root_sites:
                continue
            selected: list[_Selected] = []
            # rank candidate roots by root score, then diversity-filter
            ranked = sorted(
                loc.root_sites,
                key=lambda s: -rootness(
                    s.z_deviation, s.appearance_relevance, s.edit_support
                ),
            )
            rank = 1
            block_props: list[CausalInterventionProposal] = []
            for root in ranked:
                nodes = frozenset(self._candidate_nodes(root))
                if any(_jaccard(nodes, sel.nodes) > self.max_overlap for sel in selected):
                    continue
                score = rootness(root.z_deviation, root.appearance_relevance, root.edit_support)
                prop = self._build_proposal(
                    appearance_id, block_id, list(members), root, rank, score
                )
                block_props.append(prop)
                proposals.append(prop)
                selected.append(_Selected(nodes=nodes, score=score))
                rank += 1
                if rank > top_k:
                    break
        return tuple(proposals)

    @staticmethod
    def _seam_dependencies(
        block_proposals: Sequence[CausalInterventionProposal],
    ) -> dict[int, list[EditDependency]]:
        """Deprecated compatibility helper.

        Cross-proposal dependencies are intentionally forbidden: a macro action
        graph must carry both endpoint actions.  The model-integrated builder
        below performs that closure.
        """
        return {}

    def _candidate_nodes(self, root) -> frozenset[str]:
        actual = self._machine_of(root.operation_id)
        nodes = {root.operation_id, f"machine:{actual}"}
        for m in self.enum.eligible.get(root.operation_id, {}):
            nodes.add(f"machine:{m}")
        return frozenset(nodes)

    def _machine_of(self, oid: str) -> str:
        mode = self.enum.mode_map[self.schedule.assignment_map()[oid].mode_id][1]
        return mode.resources[0]


def _policy_driven_root(site: object, score: float) -> SimpleNamespace:
    """Minimal actionable-root shaped for direct policy-Top-K root drive.

    T1-REASONER-FIX-R1: B5 attribution (policy scores) are the contributor
    priority; root selection must equal the B5-selected contributor operation.
    The structural Explorer/Selector re-ranking (depth/actionability bonuses)
    is bypassed, so the driven root's decision_site IS the policy top-K site.
    """
    return SimpleNamespace(
        decision_site=site,
        selector_score=float(score),
        causal_chain=SimpleNamespace(
            nodes=(getattr(site, "operation_id", None),),
            decision_site_id=getattr(site, "site_id", ""),
        ),
        causal_search_trace=SimpleNamespace(
            as_tokens=lambda: (f"policy-root:{getattr(site, 'site_id', '')}",),
        ),
        # ROUTE-first round: the pilot budget is routing-only; sequencing stays
        # deferred (B3 semantic-alignment history) so no operator fall-through.
        operator_types=("routing",),
    )


def build_operator_runtime(
    *,
    block_ids: Sequence[str],
    block_members: Mapping[str, Sequence[str]],
    decision_sites: Sequence[object],
    legal_edits: Sequence[LegalEdit],
    root_scores,
    edit_scores,
    top_k: int = 3,
    transition_reasoner: InterventionTransitionReasoner | None = None,
    schedule_graph: ScheduleGraphView | None = None,
    root_drive: str = "explorer",
    proposal_cap: int | None = None,
) -> OperatorRuntimeResult:
    """Run M2 distribution -> causal expansion -> operators -> proposals.

    The Transition Reasoner is reachable only inside ``RoutingOperator``.
    Operator generation is deterministic graph reasoning and never calls
    CP-SAT.

    ``root_drive`` selects how roots are chosen per block:
      * ``"explorer"`` (default) -- CausalExplorerV2 -> ActionableRootSelectorV2,
        i.e. structural chain scoring (depth/actionability bonuses).
      * ``"policy"`` (T1-REASONER-FIX-R1) -- the policy ``root_scores`` directly
        seed the top-K root operations; the structural re-ranking is bypassed
        so attribution controls root selection (contributor -> root contract).
        In policy mode the top-K root budget is the only per-block throttle
        (the mixed-score proposal cap is disabled) so every legal edit of a
        B5-selected contributor can reach the executor.

    ``proposal_cap`` optionally re-arms a per-block cap on the number of
    emitted proposals (applies to both drives; the explorer path already caps
    at ``top_k`` when this is ``None``).
    """
    sites = tuple(decision_sites)
    edits = tuple(legal_edits)
    edit_index = {edit.edit_id: idx for idx, edit in enumerate(edits)}
    proposals: list[CausalInterventionProposal] = []
    if schedule_graph is None:
        raise ValueError("scored macro proposal construction requires schedule_graph")
    explorer = CausalExplorerV2()
    # Full learned effect requires a concrete macro proposal and runs after
    # Operator Reasoning.  The frozen pre-operator selector therefore receives
    # a neutral adapter instead of the obsolete edit-duration heuristic.
    selector = ActionableRootSelectorV2(effect_predictor=DeferredProposalEffectAdapter())
    operator_reasoner = InterventionOperatorReasoner()
    if transition_reasoner is not None:
        operator_reasoner.operators["routing"].transition_reasoner = transition_reasoner
    state_graph = OperatorStateGraph(schedule_graph=schedule_graph, legal_edits=edits)
    all_chains: list[CausalExplanationChain] = []
    all_roots: list[str] = []
    distribution_rows: list[tuple[str, str, float]] = []
    proposal_search_traces: list[tuple[str, tuple[str, ...]]] = []
    proposal_attribution_scores: list[float] = []
    filtered_blocks: list[tuple[str, str]] = []

    for b, block_id in enumerate(block_ids):
        # --- Fail-closed appearance gate (pre-D6 appearance cleanup) ---------
        # The canonical taxonomy (appearance_taxonomy.ACTIVE_APPEARANCE_IDS) is
        # the single authority for what may become actionable.  A deprecated
        # (A5/A7/A8/A9/A10) or unknown appearance id can only reach this loop by
        # loading a stale pre-v3 artifact; it is recorded and skipped BEFORE any
        # candidate / root / proposal is built.  This is the runtime fix -- it
        # holds regardless of which fixture was loaded.
        if not is_actionable_appearance(block_id):
            filtered_blocks.append(
                (str(block_id), filtered_reason(block_id) or "non_actionable_appearance")
            )
            continue
        raw = [float(root_scores[b, idx]) for idx in range(len(sites))]
        if raw:
            import math
            maximum = max(raw)
            exponents = [math.exp(value - maximum) for value in raw]
            total = sum(exponents)
            probabilities = [value / total for value in exponents]
        else:
            probabilities = []
        distribution_rows.extend(
            (block_id, site.site_id, float(probability))
            for site, probability in zip(sites, probabilities)
        )
        if root_drive == "policy":
            # T1-REASONER-FIX-R1: policy Top-K operations are the exact root
            # seeds.  ``root_scores[b, idx]`` IS the attribution (B5 top-K /
            # c_prior / random / heuristic per pilot policy); the explorer and
            # selector are bypassed so no structural re-ranking can substitute
            # a different root for the B5-selected contributor.
            ranked = sorted(
                range(len(sites)),
                key=lambda idx: float(root_scores[b, idx]),
                reverse=True,
            )
            positive = [idx for idx in ranked if float(root_scores[b, idx]) > 0.0]
            ordered = positive if positive else ranked
            # T1-REASONER-FIX-R1 (dup-site starvation): ``build_decision_sites``
            # returns ~2 decision sites per operation, so a top-K selection over
            # SITES fills the budget with duplicate operations and starves the
            # 3rd B5 contributor.  The B5 contributor budget (b5_topk) is over
            # OPERATIONS (deduped by name); seeding one root per distinct op is
            # what honors "root seed == B5 selected contributor operation".
            seen_ops: set[str] = set()
            top_idx: list[int] = []
            for idx in ordered:
                op = sites[idx].operation_id
                if op in seen_ops:
                    continue
                seen_ops.add(op)
                top_idx.append(idx)
                if len(top_idx) >= top_k:
                    break
            roots = [
                _policy_driven_root(sites[idx], float(root_scores[b, idx]))
                for idx in top_idx
            ]
            chains: list[object] = []
        else:
            chains = explorer.explore(
                AppearanceContext(
                    appearance_id=block_id,
                    members=tuple(block_members.get(block_id, ())),
                ),
                tuple(DecisionCandidate(site, probability)
                      for site, probability in zip(sites, probabilities)),
                schedule_graph,
                edits,
            )
            roots = selector.select(chains, sites, edits)
        all_chains.extend(chains)
        all_roots.extend(root.causal_chain.decision_site_id for root in roots)
        candidates: list[tuple[float, object, object]] = []
        for root in roots:
            for candidate in operator_reasoner.generate(root, state_graph):
                score = root.selector_score + candidate.score + sum(
                    float(edit_scores[b, edit_index[e.edit_id]])
                    for e in candidate.chain.actions
                )
                candidates.append((float(score), root, candidate))

        candidates.sort(key=lambda row: (-row[0], row[2].operator_type))
        selected: list[frozenset[str]] = []
        rank = 1
        for score, root, candidate in candidates:
            site = root.decision_site
            chain = candidate.chain
            actions = tuple(chain.actions)
            dependencies = tuple(EditDependency(
                dependency_id=f"{dep.before}->{dep.after}",
                dependency_type=DEP_ENABLES,
                editor_id=dep.before,
                dependent_id=dep.after,
                kind="releases_machine",
                description=dep.reason,
            ) for dep in chain.dependencies)
            order = tuple(edit.edit_id for edit in actions)
            # Provenance nodes keep the full appearance context (block members +
            # action structure).  The duplicate-detection signature is computed
            # separately below and must be ACTION-STRUCTURE ONLY: block members
            # are a shared constant across every proposal of a block, so putting
            # them in the similarity set inflates Jaccard and drops two
            # genuinely-different actions that share the same block (e.g. Mk11
            # J19.O2->M4 vs J24.O5->M4).  See T1-PROPOSAL-CLEANUP STEP 1.
            nodes: list[str] = list(block_members.get(block_id, ()))
            action_sig: list[str] = []
            for edit in actions:
                nodes.append(edit.operation_id)
                action_sig.append(f"{edit.edit_type}:{edit.operation_id}")
                if edit.source_machine:
                    nodes.append(f"machine:{edit.source_machine}")
                    action_sig.append(f"src:{edit.source_machine}")
                if edit.target_machine:
                    nodes.append(f"machine:{edit.target_machine}")
                    action_sig.append(f"tgt:{edit.target_machine}")
                if edit.target_mode_id:
                    action_sig.append(f"mode:{edit.target_mode_id}")
            for dep in dependencies:
                action_sig.append(f"dep:{dep.editor_id}->{dep.dependent_id}")
            nodes = list(dict.fromkeys(nodes))
            node_set = frozenset(dict.fromkeys(action_sig))
            if any(_jaccard(node_set, old) > 0.85 for old in selected):
                continue
            structural_edges: list[tuple[str, str]] = []
            for edit in actions:
                if edit.source_machine:
                    structural_edges.append((edit.operation_id, f"machine:{edit.source_machine}"))
                if edit.target_machine:
                    structural_edges.append((edit.operation_id, f"machine:{edit.target_machine}"))
            rooted = _prop_replace(site, trace_state=TRACE_ROOT)
            prop = CausalInterventionProposal(
                proposal_id=proposal_id(block_id, rank),
                appearance_id=block_id,
                nodes=tuple(nodes),
                edges=tuple(dict.fromkeys(structural_edges)),
                decision_sites=(rooted,),
                root_decisions=(rooted,),
                edits=actions,
                dependencies=dependencies,
                root_path=(site.operation_id,),
                proposal_score=score,
                confidence=float(1.0 / (1.0 + __import__("math").exp(-max(min(score, 30.0), -30.0)))),
                action_order=order,
                transition_explanation=chain.explanation,
                transition_depth=chain.depth,
                transition_complete=chain.complete,
                causal_chain=root.causal_chain.nodes,
                root_decision_id=site.site_id,
                operator_type=candidate.operator_type,
                source_block_id=block_id,
            )
            prop.validate()
            proposals.append(prop)
            proposal_search_traces.append((
                prop.proposal_id, root.causal_search_trace.as_tokens()
            ))
            proposal_attribution_scores.append(float(root.selector_score))
            selected.append(node_set)
            rank += 1
            # T1-REASONER-FIX-R1: in policy mode the B5 root budget (top-K per
            # block) is the ONLY throttle -- the mixed score (attribution +
            # relevance + edit scores) must not further drop a B5-selected
            # contributor's legal edits, or the attribution can never reach the
            # executor.  ``proposal_cap`` overrides; the explorer path keeps the
            # historical top_k cap so structural candidates stay bounded.
            cap = proposal_cap if proposal_cap is not None else (int(1e9) if root_drive == "policy" else top_k)
            if rank > cap:
                break
    return OperatorRuntimeResult(
        proposals=tuple(proposals),
        causal_chains=tuple(all_chains),
        actionable_root_ids=tuple(all_roots),
        decision_candidate_scores=tuple(distribution_rows),
        proposal_search_traces=tuple(proposal_search_traces),
        proposal_attribution_scores=tuple(proposal_attribution_scores),
        filtered_appearance_blocks=tuple(filtered_blocks),
    )


def build_scored_macro_proposals(**kwargs) -> tuple[CausalInterventionProposal, ...]:
    """Compatibility wrapper retaining the previous public return type."""
    return build_operator_runtime(**kwargs).proposals


def _jaccard(nodes_a: frozenset[str], nodes_b: frozenset[str]) -> float:
    if not nodes_a and not nodes_b:
        return 0.0
    return len(nodes_a & nodes_b) / len(nodes_a | nodes_b)


def _heuristic_edit_relevance(edits: Sequence[LegalEdit]) -> list[float]:
    """Deterministic decision-time relevance ranking when no model is given:
    a faster route onto a relatively-lighter receiver ranks higher."""
    scores: list[float] = []
    for e in edits:
        f = dict(e.features)
        delta_p = f.get("delta_p", 0.0)
        target_rel = f.get("target_relative_load", 0.0)
        score = -delta_p - target_rel
        scores.append(float(-score))  # low delta_p/heavy target => higher score
    # softmax to probabilities
    if not scores:
        return []
    import math

    mx = max(scores)
    ex = [math.exp(s - mx) for s in scores]
    s = sum(ex)
    return [x / s if s > 0 else 1.0 / len(ex) for x in ex]


def replace(d: LegalEdit, **kw):
    from dataclasses import replace as _r

    return _r(d, **kw)


def build_proposals(
    problem: Problem,
    schedule: Schedule,
    blocks: Mapping[str, Sequence[str]],
    *,
    top_k: int = 3,
    appearance_ids: Mapping[str, str] | None = None,
    **init_kw,
) -> tuple[CausalInterventionProposal, ...]:
    """Convenience wrapper (deterministic).  ``top_k``/``appearance_ids`` go to
    ``build``; all other keyword args go to the :class:`CausalProposalBuilder`
    constructor."""
    return CausalProposalBuilder(problem, schedule, **init_kw).build(
        blocks, top_k=top_k, appearance_ids=appearance_ids
    )


__all__ = [
    "CausalProposalBuilder",
    "OperatorRuntimeResult",
    "build_operator_runtime",
    "build_scored_macro_proposals",
    "build_proposals",
]
