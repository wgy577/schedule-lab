"""M3 GRPO layer (Patch V2 §14 + T1-M3 canonical upgrade).

Two generations live here, explicitly differentiated:

LEGACY "GRPO-lite" (Patch V2 §14 wiring, **deprecated for training**):
  - :func:`m3_reward`             lexicographic reward  w1·(−ΔCmax) + w2·FIV − w3·Risk
  - :func:`m3_grpo_loss`          softmax current scores × group-relative advantage
                                  -- NO pi_old ratio, NO clipped surrogate, NO ref KL
  - :func:`m3_grpo_reward_for_group`
  Kept verbatim for API compatibility and for :mod:`tests.test_m3_grpo`; it is
  NEVER the default of the canonical pipeline.  It was wired in the V5 era and
  never trained (identified=false, SAFE_TO_TRAIN NO).

CANONICAL GRPO (T1-M3-CANONICAL-CONSOLIDATION-UTILITY-RERANK-GRPO-R5 Phase 2):
  - reward        : Cmax(S_start) − Cmax(S_terminal)  ONLY (Frozen-Local one-step)
                    → :func:`m3_canonical_reward`
  - group         : actions of one state (Proposals ∪ STOP), G sampled w/o replacement
  - advantage     : group-relative   A_i = (R_i − mean_R) / (std_R + eps)
  - surrogate     : min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)   ε = 0.2
  - anchor        : small β·KL to the frozen SFT reference policy
  - residual      : score_GRPO = score_SFT + α·tanh(Δ)   (bounded)
  Implementation lives in :mod:`causal_schedule_lab.m3.policy`
  (:class:`M3GRPOActionHead`, :func:`grpo_group_loss`); this module re-exports
  them so ``from causal_schedule_lab.m3 import grpo`` keeps a single surface.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..counterfactual import CounterfactualResult
from ..training_v1 import group_relative_advantage
from .policy import (
    GRPO_ALPHA_RESIDUAL,
    GRPO_EPS,
    GRPO_G_MIN,
    M3GRPOActionHead,
    grpo_group_loss,
    grpo_group_logp,
    grpo_sample_group,
    resolved_scores,
)
from .selection import future_improvement_value, risk_score

CANONICAL_GRPO = True          # canonical machine is now the default surface
LEGACY_GRPO_LITE = True        # the wiring functions below are legacy, kept for tests

__all__ = [
    "m3_reward",
    "m3_grpo_loss",
    "m3_grpo_reward_for_group",
    "m3_canonical_reward",
    "M3GRPOActionHead",
    "grpo_group_loss",
    "grpo_group_logp",
    "grpo_sample_group",
    "resolved_scores",
    "CANONICAL_GRPO",
    "LEGACY_GRPO_LITE",
]


# ---------------------------------------------------------------------------
# LEGACY GRPO-lite (Patch V2 §14; deprecated for training, kept for tests)
# ---------------------------------------------------------------------------
def m3_reward(
    result: CounterfactualResult,
    *,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0,
) -> float:
    """LEGACY lexicographic reward: w1·(−ΔCmax) + w2·FIV − w3·Risk.

    NOT used by the canonical pipeline (canonical reward is Cmax delta only).
    """
    return (
        w1 * float(-result.delta_cmax)
        + w2 * future_improvement_value(result)
        - w3 * risk_score(result)
    )


def m3_grpo_loss(
    selector,
    state_f: Tensor,
    prop_f: Tensor,
    mem_f: Tensor,
    rewards: Tensor,
    *,
    epsilon: float = 1e-8,
) -> Tensor:
    """LEGACY "GRPO-lite" policy loss (softmax × group-relative advantage).

    No pi_old ratio / clipped surrogate / reference KL.  Kept verbatim for API
    compatibility and tests.  The canonical clipped GRPO is in
    :mod:`causal_schedule_lab.m3.policy` (:func:`grpo_group_loss`).
    """
    rewards = rewards.to(torch.float32)
    k = rewards.numel()
    scores = selector(state_f, prop_f, mem_f).score  # [K]
    logp = torch.log_softmax(scores, dim=-1)
    if k < 2:
        return (logp * 0.0).sum()
    advantage, informative = group_relative_advantage(rewards, epsilon=epsilon)
    if not informative:
        return (logp * 0.0).sum()
    return -(logp * advantage).sum()


def m3_grpo_reward_for_group(
    results: Sequence[CounterfactualResult],
    *,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0,
) -> list[float]:
    """LEGACY rewards for a group of evaluated proposals of one state (§14)."""
    return [m3_reward(r, w1=w1, w2=w2, w3=w3) for r in results]


# ---------------------------------------------------------------------------
# CANONICAL GRPO (Phase 2)
# ---------------------------------------------------------------------------
def m3_canonical_reward(
    base_cmax: float,
    terminal_cmax: float,
) -> float:
    """Canonical Frozen-Local reward: Cmax(S_start) − Cmax(S_terminal).

    >0 means the action improved makespan from the STARTING state.  Appearance
    / FIV / Risk are diagnostics only and intentionally excluded (see the
    canonical training spec: ``m3_reward`` may not default as canonical).
    """
    return float(base_cmax) - float(terminal_cmax)