"""Stage-2 training utilities for the Intervention Effect Predictor.

The caller first generates trajectory memory offline, then trains this model.
Formal execution, checkpointing and protocol gates remain external to this
utility and are not started merely by importing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..intervention.effect_predictor_v1 import (
    InterventionEffectPredictor,
    causal_chain_effect_features,
    intervention_effect_loss,
    memory_effect_features,
    proposal_effect_features,
    state_effect_features,
)
from ..memory import PENDING, experience_training_eligible, memory_effect_prior
from .trajectory_generation import trajectory_effect_label


@dataclass(frozen=True)
class EffectTrainingExample:
    key: str
    state_features: tuple[float, ...]
    chain_features: tuple[float, ...]
    proposal_features: tuple[float, ...]
    memory_context: tuple[float, ...]
    future_success: float
    future_gain: float
    failure: float
    immediate_delta_cmax: float
    final_delta_cmax: float
    number_future_steps: int


@dataclass(frozen=True)
class EffectTrainingBatch:
    state_features: Tensor
    chain_features: Tensor
    proposal_features: Tensor
    memory_context: Tensor
    future_success: Tensor
    future_gain: Tensor
    delta_cmax: Tensor
    failure: Tensor


def build_effect_training_examples(store, *, k: int = 5) -> list[EffectTrainingExample]:
    """Build leave-one-out ``(S,P,M)->trajectory outcome`` examples."""
    examples: list[EffectTrainingExample] = []
    for experience in store.experiences():
        if (experience.outcome is None or experience.outcome.classification == PENDING
                or not experience_training_eligible(experience)):
            continue
        label = trajectory_effect_label(experience)
        prior = memory_effect_prior(
            store,
            experience.state,
            experience.proposal,
            k=k,
            exclude_key=experience.key,
        )
        examples.append(EffectTrainingExample(
            key=experience.key,
            state_features=state_effect_features(experience.state),
            chain_features=causal_chain_effect_features(experience.proposal),
            proposal_features=proposal_effect_features(experience.proposal),
            memory_context=memory_effect_features(prior, k=k),
            future_success=label.future_success,
            future_gain=label.future_gain,
            failure=1.0 - label.future_success,
            immediate_delta_cmax=label.immediate_delta_cmax,
            final_delta_cmax=label.final_delta_cmax,
            number_future_steps=label.number_future_steps,
        ))
    return examples


def build_effect_training_batch(
    examples: list[EffectTrainingExample],
) -> EffectTrainingBatch:
    if not examples:
        raise ValueError("effect training batch requires at least one example")
    return EffectTrainingBatch(
        state_features=torch.tensor([e.state_features for e in examples], dtype=torch.float32),
        chain_features=torch.tensor([e.chain_features for e in examples], dtype=torch.float32),
        proposal_features=torch.tensor([e.proposal_features for e in examples], dtype=torch.float32),
        memory_context=torch.tensor([e.memory_context for e in examples], dtype=torch.float32),
        future_success=torch.tensor([e.future_success for e in examples], dtype=torch.float32),
        future_gain=torch.tensor([e.future_gain for e in examples], dtype=torch.float32),
        delta_cmax=torch.tensor([e.immediate_delta_cmax for e in examples], dtype=torch.float32),
        failure=torch.tensor([e.failure for e in examples], dtype=torch.float32),
    )


def train_effect_predictor(
    predictor: InterventionEffectPredictor,
    batch: EffectTrainingBatch,
    *,
    steps: int,
    lr: float = 1e-3,
) -> dict[str, float]:
    """Explicit Stage-2 trainer; never called by the online closure runtime."""
    if steps <= 0:
        raise ValueError("effect training steps must be positive")
    predictor.train()
    device = next(predictor.parameters()).device
    state_features = batch.state_features.to(device)
    chain_features = batch.chain_features.to(device)
    proposal_features = batch.proposal_features.to(device)
    memory_context = batch.memory_context.to(device)
    future_success = batch.future_success.to(device)
    future_gain = batch.future_gain.to(device)
    delta_cmax = batch.delta_cmax.to(device)
    failure = batch.failure.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(steps):
        output = predictor(
            state_features, chain_features, proposal_features, memory_context
        )
        loss = intervention_effect_loss(
            output, delta_cmax, future_success, failure
        )["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return {"loss_first": history[0], "loss_last": history[-1], "steps": float(steps)}


def freeze_effect_predictor(predictor: InterventionEffectPredictor) -> None:
    """Stage-3 boundary: freeze predictor before M3 training."""
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)


__all__ = [
    "EffectTrainingExample",
    "EffectTrainingBatch",
    "build_effect_training_examples",
    "build_effect_training_batch",
    "train_effect_predictor",
    "freeze_effect_predictor",
]
