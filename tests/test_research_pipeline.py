from pathlib import Path

import torch

from causal_schedule_lab.agentic_rl import HierarchicalActionSpace
from causal_schedule_lab.benchmarks import example_problems
from causal_schedule_lab.conditional_generator import (
    destroy_schedule,
    encode_partial_schedule,
)
from causal_schedule_lab.graph import build_scheduling_graph
from causal_schedule_lab.cip import CausalCoreDiscoverer
from causal_schedule_lab.models import ControlAction
from causal_schedule_lab.semantics import load_semantics
from causal_schedule_lab.solvers.dispatching import solve_dispatching
from causal_schedule_lab.statistics import holm_adjust, paired_comparison
from causal_schedule_lab.training_pipeline import (
    DEPENDENCIES,
    TrainingStage,
)
from causal_schedule_lab.tensorization import NUMERIC_FEATURES, tensorize_graph


ROOT = Path(__file__).resolve().parents[1]


def test_partial_schedule_destruction_is_reproducible() -> None:
    problem = example_problems()["hfsp"]
    schedule = solve_dispatching(problem, rule="balanced")
    left = destroy_schedule(problem, schedule, fraction=0.25, seed=4)
    right = destroy_schedule(problem, schedule, fraction=0.25, seed=4)
    assert left == right
    assert set(left.frozen_operations).isdisjoint(left.released_operations)
    assert set(left.frozen_operations) | set(left.released_operations) == {
        item.id for item in problem.operations
    }
    encoding = encode_partial_schedule(
        problem,
        schedule,
        released_operations=left.released_operations,
    )
    assert encoding.shape == (len(problem.operations), 8)


def test_hierarchical_action_mask_covers_control_actions() -> None:
    semantics = load_semantics(ROOT / "configs" / "base_semantics.json")
    problem = example_problems()["fsp"]
    schedule = solve_dispatching(problem, rule="lpt")
    graph = build_scheduling_graph(
        problem,
        schedule,
        project_id="generic-scheduling-project",
    )
    cip = CausalCoreDiscoverer(semantics).discover(
        problem=problem,
        schedule=schedule,
        graph=graph,
        top_k=1,
    )[0]
    batch = tensorize_graph(graph, (cip,))
    assert batch.node_features.shape[1] == len(NUMERIC_FEATURES)
    space = HierarchicalActionSpace.build(tuple(semantics.allowed_interventions))
    mask = space.legal_mask(
        cip,
        failed_attempts=1,
        remaining_budget=1.0,
    )
    assert mask.dtype == torch.bool
    assert mask.any()
    controls = {
        space.actions[index][2]
        for index in torch.nonzero(mask).flatten().tolist()
    }
    assert ControlAction.FULL_ORACLE in controls


def test_statistics_and_pipeline_dependencies() -> None:
    comparison = paired_comparison(
        [10, 12, 14, 16],
        [9, 11, 12, 15],
        seed=0,
    )
    assert comparison.mean_difference > 0
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert all(0 <= value <= 1 for value in adjusted.values())
    assert TrainingStage.SEMANTIC_COMPILATION in DEPENDENCIES[
        TrainingStage.COUNTERFACTUAL_COLLECTION
    ]
