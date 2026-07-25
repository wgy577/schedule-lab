"""Deterministic trainers for the learned framework components."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .conditional_generator import partial_repair_loss
from .learning import GraphBatch, MultiTaskTargets, multitask_loss


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float | None


def train_cip_model(
    model: nn.Module,
    training: Iterable[tuple[GraphBatch, MultiTaskTargets]],
    *,
    validation: Iterable[tuple[GraphBatch, MultiTaskTargets]] = (),
    epochs: int = 20,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> tuple[EpochMetrics, ...]:
    set_deterministic(seed)
    training = tuple(training)
    validation = tuple(validation)
    if not training:
        raise ValueError("CIP training requires at least one batch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch, target in training:
            prediction = model(batch)
            loss, _ = multitask_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = None
        if validation:
            model.eval()
            with torch.no_grad():
                validation_loss = sum(
                    float(multitask_loss(model(batch), target)[0])
                    for batch, target in validation
                ) / len(validation)
        history.append(
            EpochMetrics(
                epoch=epoch,
                training_loss=sum(losses) / len(losses),
                validation_loss=validation_loss,
            )
        )
    return tuple(history)


def train_partial_repair_model(
    model: nn.Module,
    training: Iterable[tuple[Tensor, Tensor, Tensor, Tensor | None]],
    *,
    epochs: int = 20,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> tuple[float, ...]:
    set_deterministic(seed)
    batches = tuple(training)
    if not batches:
        raise ValueError("partial-repair training requires at least one batch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        model.train()
        losses = []
        for features, modes, starts, mask in batches:
            loss = partial_repair_loss(
                model(features),
                mode_targets=modes,
                start_targets=starts,
                mode_mask=mask,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses) / len(losses))
    return tuple(history)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    configuration: dict[str, object],
    metrics: dict[str, object],
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "configuration": configuration,
            "metrics": metrics,
            "torch_version": torch.__version__,
        },
        target,
    )
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(
            {
                "configuration": configuration,
                "metrics": metrics,
                "torch_version": torch.__version__,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
