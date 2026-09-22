"""Patch V2 §9-§11 -- M3 proposal selector :math:`Q(P)=f(S,P,M)`.

M3 consumes the current schedule state ``S``, a candidate intervention proposal
``P``, and the retrieved memory ``M`` and scores how good ``P`` is.  Per the
V2 spec (§10) it is a **three-prong** encoder:

* :class:`GlobalStateEncoder` -- encodes the schedule state (Cmax / critical
  structure / load) from the deterministic ``state_vector``.
* :class:`ProposalEncoder` -- encodes the proposal shape plus M2.5 predicted
  signed delta-Cmax, FIV, success probability and risk.
* :class:`MemoryEncoder` -- encodes the memory prior (similar-case count,
  success rate, expected gain, failure rate).

The three embeddings are concatenated:
:math:`h=Concat(h_s,h_p,h_m)` and fed to the three heads -- ranking, acceptance,
risk (Patch §11).

The feature vectors are built torch-free (:func:`m3_proposal_features`,
:func:`m3_memory_features`, :func:`m3_state_features`, and :func:`m3_input_features`
for the flat composition), so the whole (S,P,M) contract is testable without a
model.  ``M3ProposalSelector`` then learns the encoder + fusion on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..memory import ProposalRecord, StateFeatures, memory_prior_composite, state_vector

STATE_PRUNGS = 5      # state_vector(S)
BASE_PROPOSAL_PRUNGS = 6  # root/action/dependency graph + affected region
EFFECT_PRUNGS = 4         # predicted delta-Cmax / FIV / success / risk
PROPOSAL_PRUNGS = BASE_PROPOSAL_PRUNGS + EFFECT_PRUNGS
MEMORY_PRUNGS = 4     # estimate / nv / mean_gain / failure rate
DEFAULT_FEATURE_DIM = STATE_PRUNGS + PROPOSAL_PRUNGS + MEMORY_PRUNGS


@dataclass
class M3Targets:
    """Supervision for the three M3 heads (Stage-3)."""

    accept: float | tuple[float, ...] = 0.0
    risk: float | tuple[float, ...] = 0.0
    outcome_class: tuple[int, ...] = ()  # 2 direct, 1 delayed, 0 failure
    rank_order: tuple[int, ...] = ()  # candidate indices, best first


# -- deterministic (torch-free) per-prong feature builders --------------------

def m3_state_features(state: StateFeatures) -> tuple[float, ...]:
    """``S`` prong: the deterministic schedule-state vector (§10 Global State)."""
    return tuple(float(x) for x in state_vector(state))


def m3_base_proposal_features(proposal: ProposalRecord) -> tuple[float, ...]:
    """Macro proposal graph features before learned effect predictions.

    Features describe the action graph itself: roots, actions, dependencies,
    graph density, and routing-action share.  No identifier-character hash is
    used as a semantic proxy.
    """
    actions = tuple(proposal.intervention_actions)
    n_actions = len(actions)
    max_edges = max(n_actions * (n_actions - 1), 1)
    route_fraction = (
        sum(action.startswith("ROUTE:") for action in actions) / n_actions
        if n_actions else 0.0
    )
    return (
        float(len(proposal.root_nodes)),
        float(n_actions),
        float(len(proposal.dependency_edges)),
        float(len(proposal.dependency_edges) / max_edges),
        float(route_fraction),
        float(len(proposal.affected_region)),
    )


def m3_proposal_features(
    proposal: ProposalRecord,
    *,
    predicted_delta_cmax: float = 0.0,
    predicted_fiv: float = 0.0,
    predicted_success_probability: float = 0.0,
    predicted_risk: float = 0.0,
    predicted_gain: float = 0.0,  # compatibility-only; not part of V1 effect contract
) -> tuple[float, ...]:
    """``P`` prong = macro graph shape + M2.5 Intervention Effect output."""
    return m3_base_proposal_features(proposal) + (
        float(predicted_delta_cmax),
        float(predicted_fiv),
        float(predicted_success_probability),
        float(predicted_risk),
    )


def m3_memory_features(
    *,
    estimate: float = 0.0,
    nv: int = 0,
    mean_gain: float = 0.0,
    k: int = 5,
) -> tuple[float, ...]:
    """``M`` prong: similar-case count, success rate, expected gain, failure rate."""
    nv_norm = float(min(nv / max(1, k), 1.0))
    return (float(estimate), nv_norm, float(mean_gain), float(1.0 - estimate))


def m3_input_features(
    state: StateFeatures,
    proposal: ProposalRecord,
    *,
    estimate: float = 0.0,
    nv: int = 0,
    mean_gain: float = 0.0,
    predicted_success_probability: float = 0.0,
    predicted_delta_cmax: float = 0.0,
    predicted_gain: float = 0.0,
    predicted_fiv: float = 0.0,
    predicted_risk: float = 0.0,
    k: int = 5,
) -> tuple[float, ...]:
    """Flat ``(S,P,M)`` vector = concat of the three per-prong builders."""
    return (
        m3_state_features(state)
        + m3_proposal_features(
            proposal,
            predicted_delta_cmax=predicted_delta_cmax,
            predicted_success_probability=predicted_success_probability,
            predicted_gain=predicted_gain,
            predicted_fiv=predicted_fiv,
            predicted_risk=predicted_risk,
        )
        + m3_memory_features(estimate=estimate, nv=nv, mean_gain=mean_gain, k=k)
    )


def memory_prior(
    store,
    query_state: StateFeatures,
    *,
    k: int = 5,
    query_proposal: ProposalRecord | None = None,
    exclude_key: str | None = None,
) -> tuple[float, int, float]:
    """The memory prior ``M`` -- estimate / trials / mean-gain.

    Uses the Patch V2 §6 composite ``(S,P)`` similarity (via
    :func:`memory_prior_composite`) so the proposal prong contributes to
    retrieval, not just the state vector cosine.
    """
    prior = memory_prior_composite(
        store, query_state, query_proposal, k=k, exclude_key=exclude_key
    )
    return prior["estimate"], int(prior["nv"]), prior["mean_gain"]


@dataclass
class M3Output:
    """Outputs of one forward pass for one proposal."""

    features: Tensor          # fused hidden embedding
    score: Tensor             # scalar ranking score
    accept_prob: Tensor       # scalar in (0,1)
    risk_prob: Tensor         # scalar in (0,1)


class GlobalStateEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ProposalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MemoryEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class M3ProposalSelector(nn.Module):
    """Three-prong M3 selector (Patch V2 §10): concat three embeddings -> heads.

    ``acceptance_head`` / ``risk_head`` emit logits (sigmoid at output);
    ``ranking_head`` is a raw regression score for @relative@ comparison.
    ``state_f``/``prop_f``/``mem_f`` are the per-prong deterministic feature
    tensors (each shape ``[n, prong_dim]`` or ``[prong_dim]`` for one proposal).
    """

    def __init__(
        self,
        state_dim: int = STATE_PRUNGS,
        proposal_dim: int = PROPOSAL_PRUNGS,
        memory_dim: int = MEMORY_PRUNGS,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.proposal_dim = proposal_dim
        self.memory_dim = memory_dim
        self.state_encoder = GlobalStateEncoder(state_dim, hidden_dim)
        self.proposal_encoder = ProposalEncoder(proposal_dim, hidden_dim)
        self.memory_encoder = MemoryEncoder(memory_dim, hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU())
        self.ranking_head = nn.Linear(hidden_dim, 1)
        self.acceptance_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_f: Tensor, prop_f: Tensor, mem_f: Tensor) -> M3Output:
        hs = self.state_encoder(state_f)
        hp = self.proposal_encoder(prop_f)
        hm = self.memory_encoder(mem_f)
        # concat the three prongs: h = Concat(h_s, h_p, h_m)  (§10)
        hidden = self.fusion(torch.cat([hs, hp, hm], dim=-1))
        return M3Output(
            features=hidden,
            score=self.ranking_head(hidden).squeeze(-1),
            accept_prob=torch.sigmoid(self.acceptance_head(hidden).squeeze(-1)),
            risk_prob=torch.sigmoid(self.risk_head(hidden).squeeze(-1)),
        )


def m3_loss(
    selector: M3ProposalSelector,
    state_f: Tensor,
    prop_f: Tensor,
    mem_f: Tensor,
    targets: M3Targets,
    *,
    accept_weight: float = 1.0,
    risk_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Stage-3 loss :math:`L_{M3}=L_{rank}+\\lambda_1L_{accept}+\\lambda_2L_{risk}`.

    ``targets.rank_order`` (best-first) drives the ranking term; a single
    proposal with no pair order contributes a zero ranking term (no supervision).
    Accept/risk use BCE against the scalar target broadcast to the batch
    (``torch.full_like`` keeps scalar-single and ``[N,...]`` batch consistent).
    """
    out = selector(state_f, prop_f, mem_f)
    losses: dict[str, Tensor] = {}

    ranking = state_f[..., 0].new_zeros(())
    order = targets.rank_order
    if len(order) >= 2:
        pairs = list(zip(order, order[1:]))
        scores = out.score
        margin = torch.stack([scores[hi] - scores[lo] for hi, lo in pairs])
        losses["ranking"] = nn.functional.softplus(-margin).mean()  # L_rank = -log σ(Q+−Q−)
    else:
        losses["ranking"] = ranking

    def target_tensor(value, like: Tensor) -> Tensor:
        if isinstance(value, tuple):
            tensor = like.new_tensor(value)
            if tensor.shape != like.shape:
                raise ValueError("per-proposal M3 target shape mismatch")
            return tensor
        return torch.full_like(like, float(value))

    accept_tgt = target_tensor(targets.accept, out.accept_prob)
    risk_tgt = target_tensor(targets.risk, out.risk_prob)
    losses["acceptance"] = nn.functional.binary_cross_entropy(out.accept_prob, accept_tgt)
    losses["risk"] = nn.functional.binary_cross_entropy(out.risk_prob, risk_tgt)
    losses["total"] = (
        losses["ranking"] + accept_weight * losses["acceptance"] + risk_weight * losses["risk"]
    )
    return losses


__all__ = [
    "DEFAULT_FEATURE_DIM",
    "STATE_PRUNGS",
    "BASE_PROPOSAL_PRUNGS",
    "EFFECT_PRUNGS",
    "PROPOSAL_PRUNGS",
    "MEMORY_PRUNGS",
    "M3Targets",
    "M3Output",
    "GlobalStateEncoder",
    "ProposalEncoder",
    "MemoryEncoder",
    "M3ProposalSelector",
    "m3_state_features",
    "m3_proposal_features",
    "m3_base_proposal_features",
    "m3_memory_features",
    "m3_input_features",
    "memory_prior",
    "m3_loss",
]
