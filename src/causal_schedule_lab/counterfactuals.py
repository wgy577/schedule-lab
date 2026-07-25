"""Offline counterfactual collection and causal-control experiment plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .benchmarks import benchmark_suite
from .ir import Problem, Schedule
from .solvers.dispatching import solve_dispatching


@dataclass(frozen=True)
class IncumbentVariant:
    problem: Problem
    schedule: Schedule
    rule: str
    seed: int


def generate_incumbents(
    *,
    families: Iterable[str] = ("jsp", "fsp", "fjsp", "hfsp"),
    instance_seeds: Iterable[int] = (0, 1, 2),
    rules: Iterable[str] = ("earliest_finish", "spt", "lpt", "balanced"),
) -> tuple[IncumbentVariant, ...]:
    variants = []
    for seed in instance_seeds:
        for problem in benchmark_suite(
            families=families,
            seed=seed,
            instances_per_family=2,
        ).values():
            for rule in rules:
                variants.append(
                    IncumbentVariant(
                        problem=problem,
                        schedule=solve_dispatching(problem, rule=rule),
                        rule=rule,
                        seed=seed,
                    )
                )
    return tuple(variants)


@dataclass(frozen=True)
class CausalControl:
    name: str
    preserve_operator: bool
    preserve_position: bool
    preserve_closure_size: bool


CAUSAL_CONTROLS = (
    CausalControl("causal", True, True, True),
    CausalControl("same_operator_random_point", True, False, False),
    CausalControl("same_point_random_operator", False, True, False),
    CausalControl("same_closure_size_random_region", False, False, True),
)
