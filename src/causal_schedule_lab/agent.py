from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .models import AgentAction, CausalInterventionPoint, ControlAction
from .operators import ActionIndex


STATE_DIM = 16


def encode_agent_state(
    cip: CausalInterventionPoint,
    *,
    remaining_budget: float = 1.0,
    failed_attempts: int = 0,
) -> torch.Tensor:
    d = cip.diagnostic
    c = cip.closure
    values = [
        d.magnitude,
        cip.responsible.modifiability,
        len(cip.causal_path.nodes),
        len(c.operation_ids),
        len(c.resource_ids),
        c.level / 3.0,
        c.predicted_outside_risk,
        cip.predicted_improvement,
        cip.predicted_validity,
        cip.predicted_cost,
        cip.uncertainty,
        cip.score,
        float(d.type == "global_sink_gap"),
        float(d.type == "high_wait"),
        max(0.0, remaining_budget),
        float(failed_attempts),
    ]
    scale = torch.tensor(
        [100, 1, 20, 160, 20, 1, 1, 100, 1, 160, 1, 10, 1, 1, 1, 10],
        dtype=torch.float32,
    )
    return torch.tensor(values, dtype=torch.float32) / scale


class MaskedActorCritic(nn.Module):
    def __init__(self, action_count: int, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, action_count)
        self.value = nn.Linear(hidden, 1)

    def forward(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(states)
        logits = self.policy(encoded)
        masked_logits = logits.masked_fill(~masks.bool(), torch.finfo(logits.dtype).min)
        return masked_logits, self.value(encoded).squeeze(-1)


@dataclass
class Transition:
    state: torch.Tensor
    mask: torch.Tensor
    action: int
    old_log_prob: float
    reward: float
    value: float
    done: bool


class MaskedPPOAgent:
    """Short-horizon operator/closure policy from the framework's first version."""

    def __init__(
        self,
        action_index: ActionIndex,
        *,
        seed: int = 0,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
    ) -> None:
        torch.manual_seed(seed)
        self.action_index = action_index
        self.model = MaskedActorCritic(len(action_index.actions))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_ratio = clip_ratio

    def choose(
        self,
        cip: CausalInterventionPoint,
        *,
        remaining_budget: float = 1.0,
        failed_attempts: int = 0,
        deterministic: bool = True,
    ) -> tuple[AgentAction, dict[str, float | int]]:
        state = encode_agent_state(
            cip,
            remaining_budget=remaining_budget,
            failed_attempts=failed_attempts,
        )
        mask = torch.tensor(self.action_index.mask_for(cip), dtype=torch.bool)
        logits, value = self.model(state.unsqueeze(0), mask.unsqueeze(0))
        distribution = torch.distributions.Categorical(logits=logits)
        selected = logits.argmax(-1) if deterministic else distribution.sample()
        index = int(selected.item())
        operator, level = self.action_index.decode(index)
        control = (
            ControlAction.EXPAND
            if operator == "expand_closure"
            else ControlAction.FULL_ORACLE
        )
        action = AgentAction(
            operator=operator,
            closure_level=level,
            control=control,
        )
        return action, {
            "index": index,
            "logProb": float(distribution.log_prob(selected).item()),
            "value": float(value.item()),
        }

    def behavior_clone(
        self,
        demonstrations: Iterable[tuple[CausalInterventionPoint, AgentAction]],
        *,
        epochs: int = 50,
    ) -> list[float]:
        examples = list(demonstrations)
        if not examples:
            return []
        losses = []
        for _ in range(epochs):
            epoch_loss = torch.tensor(0.0)
            for cip, action in examples:
                state = encode_agent_state(cip)
                mask = torch.tensor(self.action_index.mask_for(cip), dtype=torch.bool)
                logits, _ = self.model(state.unsqueeze(0), mask.unsqueeze(0))
                target = torch.tensor(
                    [self.action_index.encode(action.operator, action.closure_level)]
                )
                epoch_loss = epoch_loss + nn.functional.cross_entropy(logits, target)
            epoch_loss = epoch_loss / len(examples)
            self.optimizer.zero_grad()
            epoch_loss.backward()
            self.optimizer.step()
            losses.append(float(epoch_loss.item()))
        return losses

    def update(
        self,
        transitions: Iterable[Transition],
        *,
        gamma: float = 0.95,
        epochs: int = 4,
    ) -> dict[str, float]:
        batch = list(transitions)
        if not batch:
            return {"loss": 0.0}
        returns = []
        running = 0.0
        for transition in reversed(batch):
            running = transition.reward + gamma * running * (not transition.done)
            returns.append(running)
        returns.reverse()
        states = torch.stack([item.state for item in batch])
        masks = torch.stack([item.mask for item in batch])
        actions = torch.tensor([item.action for item in batch])
        old_log_probs = torch.tensor([item.old_log_prob for item in batch])
        target_returns = torch.tensor(returns, dtype=torch.float32)
        advantages = target_returns - torch.tensor([item.value for item in batch])
        final_loss = torch.tensor(0.0)
        for _ in range(epochs):
            logits, values = self.model(states, masks)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            ratios = (log_probs - old_log_probs).exp()
            clipped = ratios.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()
            value_loss = nn.functional.mse_loss(values, target_returns)
            entropy = distribution.entropy().mean()
            final_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            final_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return {"loss": float(final_loss.item())}
