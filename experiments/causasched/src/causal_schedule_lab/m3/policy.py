"""Canonical M3 GRPO policy (Phase 2).

The SFT scorer (frozen, `m3.scorer`) provides the calibrated proposal base.
For policy learning we add a **bounded residual**:

    score_GRPO(i) = score_SFT(i) + GRPO_ALPHA * tanh(delta_i)   (proposal i)
    score_STOP    = 0            + GRPO_ALPHA * tanh(delta_stop)

`tanh` bounds every residual to [-GRPO_ALPHA, +GRPO_ALPHA] so a broken update
cannot destroy the SFT warm-start (this is the R1 lesson: an uncalibrated raw
residual wasted the base).

Group construction (CANONICAL, NOT the legacy actor-critic):
  - group key = (instance_id, state_hash)
  - actions   = every bounded Proposal at that state + STOP (no primitive ops)
  - G         = min(GRPO_G_MIN, len(actions)); sampled without replacement
  - reward    = Cmax(S_start) - Cmax(S_terminal)   (Frozen-Local one-step,
                CANONICAL reward; appearance/FIV/Risk are NOT used here)
  - advantage A_i = (R_i - mean(R_group)) / (std(R_group) + eps);
                std ≈ 0 -> no signal, group skipped
  - update: maintain logp_old snapshot; ratio = exp(logp_new - logp_old);
    clipped surrogate min(ratio·A, clip(ratio, 1-eps, 1+eps)·A); plus small
    KL to the frozen SFT reference policy.

STOP is a first-class group action sampled like any other (no gate shortcut),
so the policy can learn to stop, not just rank proposals.
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn

from .config import GRPO_ALPHA_RESIDUAL, GRPO_EPS, GRPO_G_MIN

__all__ = [
    "M3GRPOActionHead",
    "grpo_sample_group",
    "grpo_group_logp",
    "grpo_group_loss",
]


class M3GRPOActionHead(nn.Module):
    """Learned residual deltas on top of the frozen SFT scorer.

    Inputs are the per-proposal frozen SFT *rank* scores and the state vector;
    the head emits delta for each proposal and a state-conditional delta for STOP.

    .. math:: \\text{score}_{GRPO}(i) = \\text{rank}_{SFT}(i) + \\alpha \\tanh(\\delta_i)
    """

    def __init__(self, prop_dim: int = 300, state_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.prop_net = nn.Sequential(
            nn.Linear(prop_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.stop_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, 1),
        )
        # init deltas ~ 0 -> score_GRPO starts exactly at score_SFT (warm start)
        nn.init.zeros_(self.prop_net[-1].weight)
        nn.init.zeros_(self.prop_net[-1].bias)
        nn.init.zeros_(self.stop_net[-1].weight)
        nn.init.zeros_(self.stop_net[-1].bias)

    def forward(self, prop_feats, state_feat):
        """prop_feats [N, prop_dim], state_feat [state_dim].
        Returns (delta_prop [N], delta_stop scalar)."""
        delta_prop = self.prop_net(prop_feats).squeeze(-1)
        delta_stop = self.stop_net(state_feat).squeeze(-1)
        return delta_prop, delta_stop


def resolved_scores(sft_rank, delta_prop, delta_stop, state_feat,
                    alpha=GRPO_ALPHA_RESIDUAL):
    """score_GRPO for proposals + STOP from SFT base + bounded residual.

    sft_rank: [N] frozen SFT rank scores.  Returns (prop_scores [N], stop_score)."""
    prop = sft_rank + alpha * torch.tanh(delta_prop)
    stop = 0.0 + alpha * torch.tanh(delta_stop)
    return prop, stop


def grpo_sample_group(n_prop, rng, g_min=GRPO_G_MIN):
    """Sample G distinct action ids from {0..n_prop} (n_prop == STOP).

    STOP id == n_prop.  Sampling without replacement so the group advantage is
    over genuinely distinct actions of one state.  G = min(g_min, n_prop + 1)."""
    n_actions = n_prop + 1
    g = min(g_min, n_actions)
    return rng.sample(range(n_actions), g)


def grpo_group_logp(prop_scores, stop_score, ids):
    """Log-prob of the sampled subset under the current policy.

    prop_scores [N], stop_score scalar -> categorical logits [N+1]."""
    logits = torch.cat([prop_scores, stop_score.reshape(1)], dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    ids = torch.as_tensor(ids, dtype=torch.long)
    return logp[ids], logits


def grpo_group_loss(logp_new, logp_old, rewards, reference_logp=None,
                    epsilon=GRPO_EPS, beta=0.03) -> tuple:
    """Canonical clipped GRPO loss for ONE group.  All tensors are [G].

    Returns (loss scalar, stats dict).  A group whose rewards are constant has
    std≈0 -> no relative signal: returns a differentiable zero and
    `informative=False` (the group is skipped, exactly like the lite version
    but now with an explicit informational mask).

    reference_logp: logp under the frozen SFT reference policy (for KL anchor).
    """
    g = rewards.numel()
    rewards = rewards.to(torch.float32)
    if g < 2:
        return (logp_new * 0.0).sum(), {"informative": False, "reason": "size<2"}
    std = rewards.std()
    if std < 1e-8:
        return (logp_new * 0.0).sum(), {"informative": False, "reason": "const_reward"}
    adv = (rewards - rewards.mean()) / (std + 1e-8)
    # clamp the log-ratio before exp to avoid overflow on large ratio
    ratio_safe = torch.exp((logp_new - logp_old).clamp(min=-20.0, max=20.0))
    clipped = torch.clamp(ratio_safe, 1.0 - epsilon, 1.0 + epsilon)
    surr = torch.min(ratio_safe * adv, clipped * adv)
    pg = -(surr * (adv.abs() > 0).float()).mean()   # 0-advantage terms drop
    stats = {
        "informative": True,
        "advantage_mean": float(adv.mean()),
        "advantage_std": float(adv.std()),
        "clip_fraction": float(((ratio_safe - 1).abs() > epsilon).float().mean()),
        "mean_ratio": float(ratio_safe.mean()),
        "max_ratio": float(ratio_safe.max()),
    }
    loss = pg
    if reference_logp is not None and beta > 0:
        # KL(pi_ref || pi_theta) ~ mean(logp_ref - logp_new); anchor to the SFT policy
        kl = (reference_logp - logp_new).mean()
        loss = pg + beta * kl
        stats["kl_to_ref"] = float(kl)
    return loss, stats