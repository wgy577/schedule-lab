"""Conditional partial-schedule generation and repair training utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ir import Problem, Schedule
from .solvers.cp_sat import CPSATResult, solve_cp_sat


@dataclass(frozen=True)
class PartialScheduleExample:
    problem_id: str
    frozen_operations: tuple[str, ...]
    released_operations: tuple[str, ...]
    target_mode_indices: tuple[int, ...]
    target_start_offsets: tuple[float, ...]


def encode_partial_schedule(
    problem: Problem,
    schedule: Schedule,
    *,
    released_operations: Iterable[str],
) -> Tensor:
    """Re-encode the current partial schedule after every intervention."""

    released = set(released_operations)
    assignments = schedule.assignment_map()
    jobs = problem.job_map()
    horizon = max(1, schedule.makespan)
    rows = []
    for operation in sorted(problem.operations, key=lambda item: item.id):
        assignment = assignments[operation.id]
        selected_mode = next(
            index
            for index, mode in enumerate(operation.modes)
            if mode.id == assignment.mode_id
        )
        latest = operation.due or jobs[operation.job_id].due or horizon
        rows.append(
            [
                float(operation.id in released),
                operation.index / max(1, len(problem.operations)),
                assignment.start / horizon,
                assignment.end / horizon,
                max(0, latest - assignment.end) / horizon,
                len(operation.predecessors) / max(1, len(problem.operations)),
                len(operation.modes) / max(1, len(problem.resources)),
                selected_mode / max(1, len(operation.modes) - 1),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def destroy_schedule(
    problem: Problem,
    schedule: Schedule,
    *,
    fraction: float,
    seed: int,
) -> PartialScheduleExample:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    operation_ids = sorted(item.id for item in problem.operations)
    released = set(
        rng.sample(
            operation_ids,
            k=max(1, round(len(operation_ids) * fraction)),
        )
    )
    assignments = schedule.assignment_map()
    operation_map = problem.operation_map()
    targets = [assignments[item] for item in sorted(released)]
    return PartialScheduleExample(
        problem_id=problem.id,
        frozen_operations=tuple(item for item in operation_ids if item not in released),
        released_operations=tuple(sorted(released)),
        target_mode_indices=tuple(
            next(
                index
                for index, mode in enumerate(operation_map[item.operation_id].modes)
                if mode.id == item.mode_id
            )
            for item in targets
        ),
        target_start_offsets=tuple(
            item.start / max(1, schedule.makespan) for item in targets
        ),
    )


class PartialRepairPolicy(nn.Module):
    """Small conditional decoder used before deterministic solver repair.

    It predicts a mode and normalized start preference for each released
    operation.  Predictions are hints only; CP-SAT and the Oracle retain final
    authority.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, max_modes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mode_head = nn.Linear(hidden_dim, max_modes)
        self.start_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, operation_features: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(operation_features)
        return self.mode_head(encoded), self.start_head(encoded).squeeze(-1)


def partial_repair_loss(
    prediction: tuple[Tensor, Tensor],
    *,
    mode_targets: Tensor,
    start_targets: Tensor,
    mode_mask: Tensor | None = None,
) -> Tensor:
    mode_logits, start = prediction
    if mode_mask is not None:
        mode_logits = mode_logits.masked_fill(
            ~mode_mask.bool(),
            torch.finfo(mode_logits.dtype).min,
        )
    return F.cross_entropy(mode_logits, mode_targets) + F.smooth_l1_loss(
        start,
        start_targets,
    )


class SolverBackedConditionalGenerator:
    """Multi-candidate generator with reliable deterministic completion."""

    def __init__(
        self,
        *,
        deterministic_time: float = 1.0,
        seeds: Iterable[int] = (0,),
    ) -> None:
        self.deterministic_time = deterministic_time
        self.seeds = tuple(seeds)

    def generate(
        self,
        problem: Problem,
        incumbent: Schedule,
        *,
        released_operations: Iterable[str],
    ) -> tuple[CPSATResult, ...]:
        released = set(released_operations)
        frozen = {item.id for item in problem.operations} - released
        return tuple(
            solve_cp_sat(
                problem,
                incumbent=incumbent,
                frozen_operation_ids=frozen,
                seed=seed,
                workers=1,
                max_deterministic_time=self.deterministic_time,
            )
            for seed in self.seeds
        )
