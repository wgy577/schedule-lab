"""Family-stratified experiment and ablation runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .core_validation import validate_schedule
from .ir import Problem, Schedule
from .metrics import schedule_metrics


@dataclass(frozen=True)
class MethodResult:
    method: str
    family: str
    instance_id: str
    seed: int
    feasible: bool
    metrics: dict[str, object]
    runtime_seconds: float
    metadata: dict[str, object]


Method = Callable[[Problem, Schedule, int], Schedule]


def run_methods(
    cases: Iterable[tuple[Problem, Schedule]],
    methods: dict[str, Method],
    *,
    seeds: Iterable[int],
) -> tuple[MethodResult, ...]:
    results = []
    for problem, incumbent in cases:
        for seed in seeds:
            for name, method in methods.items():
                started = time.perf_counter()
                candidate = method(problem, incumbent, seed)
                validation = validate_schedule(problem, candidate)
                results.append(
                    MethodResult(
                        method=name,
                        family=problem.kind,
                        instance_id=problem.id,
                        seed=seed,
                        feasible=validation.feasible,
                        metrics=schedule_metrics(problem, candidate, incumbent),
                        runtime_seconds=time.perf_counter() - started,
                        metadata={"validation": validation.as_dict()},
                    )
                )
    return tuple(results)


def save_results(results: Iterable[MethodResult], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True)
            for result in results
        )
        + "\n",
        encoding="utf-8",
    )
    return target


ABLATIONS = {
    "full": (),
    "no_project_semantics": ("semantic_compiler",),
    "no_causal_path": ("causal_path_head",),
    "no_closure_learning": ("closure_head",),
    "no_agentic_rl": ("hierarchical_policy",),
    "no_multi_fidelity": ("light_oracle", "posterior_bias"),
    "no_counterfactual_controls": ("causal_controls",),
    "solver_only": (
        "semantic_compiler",
        "cip_model",
        "closure_head",
        "hierarchical_policy",
    ),
}
