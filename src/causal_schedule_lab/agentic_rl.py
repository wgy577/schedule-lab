"""Hierarchical short-horizon Agentic RL action space.

The policy chooses an operator, closure level, and control action jointly.
Illegal combinations are masked before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .models import AgentAction, CausalInterventionPoint, ControlAction


@dataclass(frozen=True)
class HierarchicalActionSpace:
    actions: tuple[tuple[str, int, ControlAction], ...]

    @classmethod
    def build(cls, operators: tuple[str, ...]) -> "HierarchicalActionSpace":
        controls = (
            ControlAction.RETRY,
            ControlAction.EXPAND,
            ControlAction.NEXT_CIP,
            ControlAction.BACKTRACK,
            ControlAction.FULL_ORACLE,
            ControlAction.STOP_LOCAL,
            ControlAction.STOP_GLOBAL,
        )
        return cls(
            tuple(
                (operator, level, control)
                for operator in operators
                for level in (1, 2, 3)
                for control in controls
            )
        )

    def legal_mask(
        self,
        cip: CausalInterventionPoint,
        *,
        failed_attempts: int,
        remaining_budget: float,
    ) -> Tensor:
        recommended = set(cip.recommended_operators)
        values = []
        for operator, level, control in self.actions:
            legal = operator in recommended and level >= cip.closure.level
            legal &= not (
                control == ControlAction.EXPAND and level <= cip.closure.level
            )
            legal &= not (
                control == ControlAction.RETRY and failed_attempts == 0
            )
            legal &= not (
                control == ControlAction.FULL_ORACLE and remaining_budget <= 0
            )
            values.append(legal)
        return torch.tensor(values, dtype=torch.bool)


class HierarchicalMaskedActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_count)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.body(state)
        logits = self.actor(hidden).masked_fill(
            ~legal_mask.bool(),
            torch.finfo(state.dtype).min,
        )
        return logits, self.critic(hidden).squeeze(-1)


def decode_action(
    action_space: HierarchicalActionSpace,
    index: int,
) -> AgentAction:
    operator, level, control = action_space.actions[index]
    return AgentAction(
        operator=operator,
        closure_level=level,
        control=control,
    )


@dataclass(frozen=True)
class RewardTerms:
    full_improvement: float = 0.0
    best_improvement: float = 0.0
    causal_effect: float = 0.0
    validation_cost: float = 0.0
    closure_size: int = 0
    risk: float = 0.0
    invalid: bool = False
    light_improvement: float = 0.0
    light_causal_effect: float = 0.0


def full_reward(
    terms: RewardTerms,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.25,
    cost_weight: float = 0.05,
    closure_weight: float = 0.01,
    risk_weight: float = 0.25,
    invalid_weight: float = 1.0,
) -> float:
    return (
        alpha * terms.full_improvement
        + beta * terms.best_improvement
        + gamma * terms.causal_effect
        - cost_weight * terms.validation_cost
        - closure_weight * terms.closure_size
        - risk_weight * terms.risk
        - invalid_weight * float(terms.invalid)
    )


def light_reward(
    terms: RewardTerms,
    *,
    alpha: float = 1.0,
    gamma: float = 0.2,
    cost_weight: float = 0.05,
    invalid_weight: float = 1.0,
) -> float:
    return (
        alpha * terms.light_improvement
        + gamma * terms.light_causal_effect
        - cost_weight * terms.validation_cost
        - invalid_weight * float(terms.invalid)
    )


def advantage_weighted_behavior_cloning_loss(
    logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    *,
    temperature: float = 1.0,
    maximum_weight: float = 20.0,
) -> Tensor:
    weights = torch.exp(advantages / temperature).clamp_max(maximum_weight)
    negative_log_likelihood = nn.functional.cross_entropy(
        logits,
        actions,
        reduction="none",
    )
    return (weights.detach() * negative_log_likelihood).mean()
