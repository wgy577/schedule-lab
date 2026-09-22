"""Patch V2 §6 -- three-component memory similarity :math:`sim=\\alpha sim_G+\\beta sim_A+\\gamma sim_P`.

The patch's retrieval is richer than a bare state-vector cosine: a query
:math:`(S,P)` is matched against stored :math:`(S_i,P_i)` by the sum of three
weighted components:

* :func:`sim_graph` (:math:`sim_G`) -- schedule-state similarity (cosine over the
  deterministic 5-dim :func:`~.state_encoder.state_vector`, mapped to :math:`[0,1]`).
* :func:`sim_appearance` (:math:`sim_A`) -- appearance-block similarity (same
  `appearance_type` matched; an unnamed operand yields a neutral ``0.5``).
* :func:`sim_proposal` (:math:`sim_P`) -- proposal-structure similarity (Jaccard
  of the stored action/root sets).

:func:`composite_similarity` blends them with ``alpha``/``beta``/``gamma`` (the
patch default is balanced: graph and appearance and proposal each earn weight),
and :func:`retrieve_similar` / :func:`memory_prior_composite` do Top-K retrieval
and the empirical success prior over that composite ordering.  ``identified=false``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .experience_store import (
    ExperienceStore,
    InterventionExperience,
    ProposalRecord,
    experience_final_gain,
    experience_future_success,
)
from .retrieval import experience_similarity
from .state_encoder import StateFeatures, state_vector

__all__ = [
    "sim_graph",
    "sim_appearance",
    "sim_proposal",
    "composite_similarity",
    "retrieve_similar",
    "memory_prior_composite",
    "memory_effect_prior",
    "retrieve_similar_scored",
]


def sim_graph(query: StateFeatures, cand: StateFeatures) -> float:
    """Graph/state similarity, preferring learned graph embeddings when set."""
    q = query.graph_embedding or state_vector(query)
    c = cand.graph_embedding or state_vector(cand)
    embedding = 0.5 * (experience_similarity(q, c) + 1.0)
    q_struct = set(query.local_operation_nodes + query.local_machine_nodes + query.local_mode_nodes)
    c_struct = set(cand.local_operation_nodes + cand.local_machine_nodes + cand.local_mode_nodes)
    structural = _jaccard(q_struct, c_struct) if (q_struct or c_struct) else embedding
    return 0.75 * embedding + 0.25 * structural


def _appearance_type(state: StateFeatures | None,
                     proposal: ProposalRecord | None) -> str | None:
    """Appearance type lives on the state block (§5.2 A); fall back to the
    proposal's (a caller that only has P still gets a comparable label)."""
    if state is not None and state.appearance_type:
        return state.appearance_type
    if proposal is not None and proposal.appearance_type:
        return proposal.appearance_type
    return None


def sim_appearance(query: StateFeatures | None, cand: StateFeatures | None,
                   query_p: ProposalRecord | None, cand_p: ProposalRecord | None) -> float:
    """Appearance similarity: 1.0 on same appearance_type, else 0.5 (neutral)."""
    q = _appearance_type(query, query_p)
    c = _appearance_type(cand, cand_p)
    if q is None and c is None:
        return 0.5
    if q is None or c is None:
        return 0.5          # one side untyped -> neutral, not penalised
    type_score = 1.0 if q == c else 0.0
    if type_score == 0.0:
        return 0.0
    if query is None or cand is None:
        return type_score
    scale = max(abs(query.appearance_score), abs(cand.appearance_score), 1.0)
    score_similarity = max(0.0, 1.0 - abs(query.appearance_score - cand.appearance_score) / scale)
    return 0.75 * type_score + 0.25 * score_similarity


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0          # both empty -> identical
    return len(a & b) / len(union)


def sim_proposal(query: ProposalRecord, cand: ProposalRecord) -> float:
    """Proposal action-graph similarity (nodes and dependency edges)."""
    q = (set(query.intervention_actions) | set(query.root_nodes)
         | set(query.affected_region) | set(query.causal_chain))
    c = (set(cand.intervention_actions) | set(cand.root_nodes)
         | set(cand.affected_region) | set(cand.causal_chain))
    if query.root_decision_id:
        q.add(f"root:{query.root_decision_id}")
    if cand.root_decision_id:
        c.add(f"root:{cand.root_decision_id}")
    if query.operator_type:
        q.add(f"operator:{query.operator_type}")
    if cand.operator_type:
        c.add(f"operator:{cand.operator_type}")
    node_sim = _jaccard(q, c)
    query_relations = set(query.causal_relations)
    candidate_relations = set(cand.causal_relations)
    relation_sim = (
        _jaccard(query_relations, candidate_relations)
        if query_relations or candidate_relations else node_sim
    )
    q_edges, c_edges = set(query.dependency_edges), set(cand.dependency_edges)
    edge_sim = _jaccard(q_edges, c_edges) if (q_edges or c_edges) else node_sim
    return 0.60 * node_sim + 0.20 * edge_sim + 0.20 * relation_sim


def composite_similarity(
    query_state: StateFeatures,
    query_proposal: ProposalRecord,
    cand_state: StateFeatures,
    cand_proposal: ProposalRecord,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """The §6 blend :math:`\\alpha sim_G + \\beta sim_A + \\gamma sim_P` (normalised)."""
    total = alpha + beta + gamma
    if total <= 0:
        return 0.0
    return (alpha * sim_graph(query_state, cand_state)
            + beta * sim_appearance(query_state, cand_state, query_proposal, cand_proposal)
            + gamma * sim_proposal(query_proposal, cand_proposal)) / total


def retrieve_similar(
    store: ExperienceStore,
    query_state: StateFeatures,
    query_proposal: ProposalRecord | None = None,
    *,
    k: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    exclude_key: str | None = None,
) -> list[InterventionExperience]:
    """Top-``k`` stored experiences closest by composite ``(S,P)`` similarity."""
    if k <= 0 or len(store) == 0:
        return []
    scored = []
    for e in store.experiences():
        if exclude_key is not None and e.key == exclude_key:
            continue
        sim = composite_similarity(
            query_state, query_proposal or ProposalRecord(),
            e.state, e.proposal, alpha=alpha, beta=beta, gamma=gamma,
        )
        scored.append((sim, e))
    scored.sort(key=lambda t: -t[0])
    return [e for _, e in scored[:k]]


def retrieve_similar_scored(
    store: ExperienceStore,
    query_state: StateFeatures,
    query_proposal: ProposalRecord | None = None,
    *,
    k: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    exclude_key: str | None = None,
) -> list[tuple[InterventionExperience, float]]:
    """Top-K trajectories with their actual composite similarity weights."""
    if store is None or k <= 0 or len(store) == 0:
        return []
    scored: list[tuple[InterventionExperience, float]] = []
    for experience in store.experiences():
        if exclude_key is not None and experience.key == exclude_key:
            continue
        similarity = composite_similarity(
            query_state, query_proposal or ProposalRecord(),
            experience.state, experience.proposal,
            alpha=alpha, beta=beta, gamma=gamma,
        )
        scored.append((experience, float(similarity)))
    scored.sort(key=lambda row: (-row[1], row[0].key))
    return scored[:k]


def memory_effect_prior(
    store: ExperienceStore | None,
    query_state: StateFeatures,
    query_proposal: ProposalRecord | None = None,
    *,
    k: int = 5,
    exclude_key: str | None = None,
) -> dict[str, float]:
    """Similarity-weighted contrastive trajectory prior.

    ``expected_value = sum(sim_i * reward_i) / sum(sim_i)``.  A trajectory
    that eventually lowers Cmax has positive reward equal to its final gain;
    a failed trajectory has negative reward equal to observed worsening, with
    a unit/risk floor so Cmax-neutral failures remain negative evidence.
    """
    picks = [
        (experience, similarity)
        for experience, similarity in retrieve_similar_scored(
            store, query_state, query_proposal, k=k, exclude_key=exclude_key
        )
        if experience.outcome is not None and experience.outcome.classification != "pending"
    ]
    if not picks:
        return {
            "success_probability": 0.0, "estimate": 0.0, "nv": 0,
            "mean_gain": 0.0, "fiv": 0.0, "expected_value": 0.0,
            "mean_risk": 0.0,
        }
    weights = [max(0.0, similarity) for _, similarity in picks]
    denominator = sum(weights)
    if denominator <= 1e-12:
        weights = [1.0] * len(picks)
        denominator = float(len(picks))
    success_weight = 0.0
    success_gain = 0.0
    rewards: list[float] = []
    risks: list[float] = []
    for (experience, _), weight in zip(picks, weights):
        success = experience_future_success(experience)
        gain = experience_final_gain(experience)
        outcome = experience.outcome
        risk = max(0.0, min(1.0, float(outcome.risk)))
        if success:
            success_weight += weight
            success_gain += weight * gain
            reward = gain
        else:
            worsening = max(0.0, float(outcome.delta_cmax))
            reward = -max(worsening, risk, 1.0)
        rewards.append(weight * reward)
        risks.append(weight * risk)
    probability = success_weight / denominator
    mean_gain = success_gain / success_weight if success_weight > 1e-12 else 0.0
    return {
        "success_probability": float(probability),
        "estimate": float(probability),
        "nv": len(picks),
        "mean_gain": float(mean_gain),
        "fiv": float(probability * mean_gain),
        "expected_value": float(sum(rewards) / denominator),
        "mean_risk": float(sum(risks) / denominator),
    }


def memory_prior_composite(
    store: ExperienceStore,
    query_state: StateFeatures,
    query_proposal: ProposalRecord | None = None,
    *,
    k: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    exclude_key: str | None = None,
) -> dict[str, float]:
    """Empirical success prior over the composite-ordered Top-K (§7).

    ``estimate`` = P(future success|S,P), ``nv`` = similar completed trials,
    ``mean_gain`` = mean final Cmax gain among successful trajectories, and
    ``fiv=estimate*mean_gain``.  Empty memory is
    fail-safe ``estimate=0.0`` (the exploiter decides, as in the numeric prior).
    """
    picks = [
        e for e in retrieve_similar(store, query_state, query_proposal,
                                    k=k, alpha=alpha, beta=beta, gamma=gamma,
                                    exclude_key=exclude_key)
        if e.outcome is not None and e.outcome.classification != "pending"
    ]
    if not picks:
        return {"estimate": 0.0, "nv": 0, "mean_gain": 0.0, "fiv": 0.0}
    nv = len(picks)
    succ = [e for e in picks if experience_future_success(e)]
    gains = [experience_final_gain(e) for e in succ]
    estimate = float(len(succ) / nv)
    mean_gain = float(sum(gains) / len(gains)) if gains else 0.0
    return {
        "estimate": estimate,
        "nv": nv,
        "mean_gain": mean_gain,
        "fiv": estimate * mean_gain,
    }
