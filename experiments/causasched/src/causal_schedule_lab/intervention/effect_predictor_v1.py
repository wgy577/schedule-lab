"""Trainable intervention-effect prediction between proposals and M3.

The predictor is deliberately solver-free.  CP-SAT may create offline labels
and validate the final shortlist, but is never imported or called here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import log1p
import os
from pathlib import Path
from statistics import pstdev
from typing import Sequence

import torch
from torch import Tensor, nn

from ..memory import ProposalRecord, StateFeatures, memory_effect_prior


CAUSE_RELATIONS = (
    "predecessor",
    "resource_conflict",
    "resource_blocker",
    "routing_dependency",
    "constraint_dependency",
    "sequencing_dependency",
)
OPERATOR_TYPES = ("routing", "sequencing", "insertion", "timing")
APPEARANCE_TYPES = tuple(f"A{i}" for i in range(1, 11))

STATE_FEATURE_DIM = 19
CHAIN_FEATURE_DIM = 12
PROPOSAL_FEATURE_DIM = 9
MEMORY_FEATURE_DIM = 6


@dataclass(frozen=True)
class EffectPrediction:
    delta_cmax_pred: float
    fiv_pred: float
    success_probability: float
    risk: float
    memory_expected_value: float = 0.0
    memory_count: int = 0


@dataclass(frozen=True)
class InterventionEffectOutput:
    features: Tensor
    delta_cmax_pred: Tensor
    success_probability: Tensor
    expected_future_gain: Tensor
    fiv_pred: Tensor
    risk: Tensor

    # Compatibility aliases for the previous M2.5 surface.
    @property
    def success_prob(self) -> Tensor:
        return self.success_probability

    @property
    def expected_gain(self) -> Tensor:
        return self.expected_future_gain

    @property
    def fiv(self) -> Tensor:
        return self.fiv_pred


def state_effect_features(state: StateFeatures) -> tuple[float, ...]:
    """Cmax/gap/utilisation/critical/load/appearance features.

    Gap is an input state descriptor only.  It is absent from every target,
    reward and acceptance formula in this module.
    """
    cmax = max(float(state.cmax), 1.0)
    loads = tuple(float(row[0]) / cmax for row in state.machine_load.values())
    mean_u = sum(loads) / len(loads) if loads else 0.0
    appearance = tuple(1.0 if state.appearance_type == name else 0.0
                       for name in APPEARANCE_TYPES)
    unknown = 1.0 if state.appearance_type and state.appearance_type not in APPEARANCE_TYPES else 0.0
    return (
        log1p(max(float(state.cmax), 0.0)) / 10.0,
        float(state.gap) / max(len(state.machine_load) * cmax, 1.0),
        mean_u,
        max(loads, default=0.0),
        min(loads, default=0.0),
        pstdev(loads) if len(loads) > 1 else 0.0,
        float(state.critical_path_length) / cmax,
        float(state.appearance_score),
        *appearance,
        unknown,
    )


def causal_chain_effect_features(chain: object | None) -> tuple[float, ...]:
    relations = tuple(str(item) for item in getattr(
        chain, "relations", getattr(chain, "causal_relations", ())
    ))
    depth = int(getattr(chain, "depth", getattr(chain, "causal_chain_depth", 0)))
    counts = tuple(float(relations.count(name)) for name in CAUSE_RELATIONS)
    nodes = tuple(getattr(chain, "nodes", getattr(chain, "causal_chain", ())))
    root = str(getattr(chain, "root_candidate_id", ""))
    root_position = (
        nodes.index(root) / max(len(nodes) - 1, 1)
        if root in nodes else float(getattr(chain, "causal_root_position", 0.0))
    )
    explanation_gain = max(
        0.0,
        float(getattr(
            chain, "causal_explanation_gain",
            float(getattr(chain, "causal_score", 0.0))
            - float(getattr(chain, "m2_root_score", 0.0)),
        )),
    )
    return (
        float(depth),
        *counts,
        float(relations.count("resource_blocker")),
        float(relations.count("resource_conflict")),
        float(root_position),
        float(explanation_gain),
        1.0 if bool(getattr(chain, "actionable", False)) else 0.0,
    )


def proposal_effect_features(proposal: ProposalRecord) -> tuple[float, ...]:
    actions = tuple(proposal.intervention_actions)
    dependencies = tuple(proposal.dependency_edges)
    operator = tuple(1.0 if proposal.operator_type == name else 0.0
                     for name in OPERATOR_TYPES)
    maximum_edges = max(len(actions) * (len(actions) - 1), 1)
    complexity = len(actions) + len(dependencies) + 0.5 * len(proposal.affected_region)
    return (
        *operator,
        float(len(actions)),
        float(max(len(proposal.causal_chain), 1)),
        float(len(dependencies)),
        float(len(dependencies) / maximum_edges),
        float(complexity),
    )


def memory_effect_features(prior: dict[str, float], *, k: int = 5) -> tuple[float, ...]:
    return (
        float(prior.get("success_probability", prior.get("estimate", 0.0))),
        float(min(float(prior.get("nv", 0.0)) / max(k, 1), 1.0)),
        float(prior.get("mean_gain", 0.0)),
        float(prior.get("fiv", 0.0)),
        float(prior.get("expected_value", 0.0)),
        float(prior.get("mean_risk", 0.0)),
    )


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class InterventionEffectPredictor(nn.Module):
    """Four-prong learned estimator over ``(State, Chain, Proposal, Memory)``."""

    def __init__(self, *, hidden_dim: int = 32, memory_store=None, memory_k: int = 5) -> None:
        super().__init__()
        self.memory_store = memory_store
        self.memory_k = int(memory_k)
        self.state_encoder = _Encoder(STATE_FEATURE_DIM, hidden_dim)
        self.chain_encoder = _Encoder(CHAIN_FEATURE_DIM, hidden_dim)
        self.proposal_encoder = _Encoder(PROPOSAL_FEATURE_DIM, hidden_dim)
        self.memory_encoder = _Encoder(MEMORY_FEATURE_DIM, hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU())
        self.delta_cmax_head = nn.Linear(hidden_dim, 1)
        self.success_head = nn.Linear(hidden_dim, 1)
        self.future_gain_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_features: Tensor,
        chain_features: Tensor,
        proposal_features: Tensor,
        memory_context: Tensor,
    ) -> InterventionEffectOutput:
        hidden = self.fusion(torch.cat((
            self.state_encoder(state_features),
            self.chain_encoder(chain_features),
            self.proposal_encoder(proposal_features),
            self.memory_encoder(memory_context),
        ), dim=-1))
        learned_success = torch.sigmoid(self.success_head(hidden).squeeze(-1))
        learned_gain = nn.functional.softplus(self.future_gain_head(hidden).squeeze(-1))
        learned_risk = torch.sigmoid(self.risk_head(hidden).squeeze(-1))
        # The retrieval prior is an auditable baseline, while the neural heads
        # remain trainable.  With no similar rows (weight=0) predictions are
        # purely learned; with a full Top-K memory context they reproduce the
        # empirical success/gain/risk prior.  FIV remains exactly p*gain.
        memory_weight = memory_context[..., 1].clamp(0.0, 1.0)
        success = (1.0 - memory_weight) * learned_success + memory_weight * memory_context[..., 0]
        gain = (1.0 - memory_weight) * learned_gain + memory_weight * memory_context[..., 2]
        risk = (1.0 - memory_weight) * learned_risk + memory_weight * memory_context[..., 5]
        return InterventionEffectOutput(
            features=hidden,
            delta_cmax_pred=self.delta_cmax_head(hidden).squeeze(-1),
            success_probability=success,
            expected_future_gain=gain,
            fiv_pred=success * gain,
            risk=risk.clamp(0.0, 1.0),
        )

    def predict(
        self,
        state: StateFeatures,
        causal_chain: object | None,
        proposal: ProposalRecord,
    ) -> EffectPrediction:
        prior = memory_effect_prior(
            self.memory_store, state, proposal, k=self.memory_k
        ) if self.memory_store is not None else {
            "expected_value": 0.0, "nv": 0, "mean_gain": 0.0,
            "fiv": 0.0, "success_probability": 0.0, "mean_risk": 0.0,
        }
        device = next(self.parameters()).device
        with torch.no_grad():
            output = self(
                torch.tensor([state_effect_features(state)], dtype=torch.float32, device=device),
                torch.tensor([causal_chain_effect_features(causal_chain)], dtype=torch.float32, device=device),
                torch.tensor([proposal_effect_features(proposal)], dtype=torch.float32, device=device),
                torch.tensor([memory_effect_features(prior, k=self.memory_k)], dtype=torch.float32, device=device),
            )
        return EffectPrediction(
            delta_cmax_pred=float(output.delta_cmax_pred.item()),
            fiv_pred=float(output.fiv_pred.item()),
            success_probability=float(output.success_probability.item()),
            risk=float(output.risk.item()),
            memory_expected_value=float(prior["expected_value"]),
            memory_count=int(prior["nv"]),
        )


def _mask_stats(mask: Tensor | None, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Return (float mask broadcast to ref, valid count) for masked reduction.

    ``None`` means "all rows valid" (backward-compatible default).  A supplied
    mask is a per-head validity firewall: only ``True`` rows contribute loss.
    """
    if mask is None:
        m = torch.ones_like(ref)
    else:
        m = mask.to(ref.dtype).reshape(ref.shape)
    return m, m.sum()


def _masked_mse(pred: Tensor, true: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    true = true.to(pred.dtype)
    m, n = _mask_stats(mask, pred)
    per = (pred - true) ** 2
    denom = torch.clamp(n, min=1.0)
    return (per * m).sum() / denom, n


def _masked_bce(pred: Tensor, true: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    true = true.to(pred.dtype)
    m, n = _mask_stats(mask, pred)
    per = nn.functional.binary_cross_entropy(pred, true, reduction="none")
    denom = torch.clamp(n, min=1.0)
    return (per * m).sum() / denom, n


def intervention_effect_loss(
    output: InterventionEffectOutput,
    delta_cmax_true: Tensor,
    success_true: Tensor,
    failure_true: Tensor,
    *,
    cmax_weight: float = 1.0,
    success_weight: float = 1.0,
    risk_weight: float = 1.0,
    delta_valid: Tensor | None = None,
    success_valid: Tensor | None = None,
    risk_valid: Tensor | None = None,
    future_gain_true: Tensor | None = None,
    future_gain_valid: Tensor | None = None,
    future_gain_weight: float = 0.0,
) -> dict[str, Tensor]:
    """L_effect = lambda1*MSE(delta Cmax) + lambda2*BCE(success) + lambda3*BCE(failure).

    Per-head validity masks (``*_valid``) are the D3 supervision firewall: a row
    contributes to a head only if its mask entry is True.  ``None`` masks mean
    "all rows valid" (backward compatible; existing callers unchanged).

    ``future_gain`` is INTENTIONALLY excluded from ``total``.  The live label
    ``experience_final_gain`` is a clamp_min(0) terminal trajectory gain vs S0,
    not the continuation-minus-immediate attributable target the contract
    requires, and there is no trajectory-value machinery to ground it.  Its
    head is therefore fail-closed: ``future_gain_weight`` defaults to 0 and the
    term never joins ``total``.  The key is reported for observability only.
    """
    cmax, n_delta = _masked_mse(output.delta_cmax_pred, delta_cmax_true, delta_valid)
    success, n_success = _masked_bce(output.success_probability, success_true, success_valid)
    risk, n_risk = _masked_bce(output.risk, failure_true, risk_valid)

    # future_gain: fail-closed. Compute a diagnostic residual only if a caller
    # explicitly supplies a target AND a nonzero weight AND a mask; otherwise the
    # loss is a hard zero, disconnected from the graph, and NEVER in ``total``.
    if future_gain_true is not None and future_gain_weight != 0.0:
        future_gain, n_future_gain = _masked_mse(
            output.expected_future_gain, future_gain_true, future_gain_valid
        )
    else:
        future_gain = torch.zeros((), dtype=output.delta_cmax_pred.dtype,
                                  device=output.delta_cmax_pred.device)
        n_future_gain = torch.zeros((), dtype=output.delta_cmax_pred.dtype,
                                    device=output.delta_cmax_pred.device)

    total = cmax_weight * cmax + success_weight * success + risk_weight * risk

    return {
        # Legacy keys (unchanged contract for existing callers) ----------------
        "cmax": cmax,
        "success": success,
        "risk": risk,
        "total": total,
        # D3 explicit per-head losses -----------------------------------------
        "delta_loss": cmax,
        "success_loss": success,
        "risk_loss": risk,
        "future_gain_loss": future_gain,  # observability only; NOT in total
        "total_loss": total,
        # Closure: valid-row counts per head ----------------------------------
        "n_delta_valid": n_delta,
        "n_success_valid": n_success,
        "n_risk_valid": n_risk,
        "n_future_gain_valid": n_future_gain,
    }


@dataclass(frozen=True)
class M3EffectWeights:
    delta_cmax: float = 1.0
    fiv: float = 1.0
    success: float = 1.0
    risk: float = 1.0


def load_m3_effect_weights(path: str | Path | None = None) -> M3EffectWeights:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[3]
    source = Path(
        path or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or root / "configs" / "hyperparameters.yaml"
    )
    section = json.loads(source.read_text(encoding="utf-8"))["effect_predictor"]["m3_score_weights"]
    return M3EffectWeights(
        delta_cmax=float(section["delta_cmax"]), fiv=float(section["fiv"]),
        success=float(section["success"]), risk=float(section["risk"]),
    )


def m3_effect_score(
    prediction: EffectPrediction, weights: M3EffectWeights | None = None
) -> float:
    """Frozen pre-validation ranking formula; Gap is intentionally absent."""
    weights = weights or load_m3_effect_weights()
    return (
        weights.delta_cmax * (-prediction.delta_cmax_pred)
        + weights.fiv * prediction.fiv_pred
        + weights.success * prediction.success_probability
        - weights.risk * prediction.risk
    )


class DeferredProposalEffectAdapter:
    """Neutral pre-operator adapter for ActionableRootSelectorV2.

    Full effect prediction requires a concrete macro proposal and therefore
    runs after Operator Reasoning.  Returning zero prevents the obsolete
    heuristic from influencing root selection while preserving the frozen
    Explorer/Selector call contract.
    """

    def predict_root_impact(self, root_operation_id, operator_types, chain, edits) -> float:
        return 0.0


__all__ = [
    "CHAIN_FEATURE_DIM", "MEMORY_FEATURE_DIM", "PROPOSAL_FEATURE_DIM",
    "STATE_FEATURE_DIM", "DeferredProposalEffectAdapter", "EffectPrediction",
    "InterventionEffectOutput", "InterventionEffectPredictor", "M3EffectWeights",
    "causal_chain_effect_features", "intervention_effect_loss",
    "m3_effect_score", "memory_effect_features", "proposal_effect_features",
    "state_effect_features", "load_m3_effect_weights",
]
