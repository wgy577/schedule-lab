# TEST-TAGS: modules=C,E,F,G; capabilities=controller,end_to_end,rollback; level=system; cost=medium
from pathlib import Path

from causal_schedule_lab.agent import MaskedPPOAgent
from causal_schedule_lab.benchmarks import example_problems
from causal_schedule_lab.cip import CausalCoreDiscoverer
from causal_schedule_lab.controller import AgenticImprovementController
from causal_schedule_lab.operators import ActionIndex
from causal_schedule_lab.repair import GenericCPSATRepairGenerator
from causal_schedule_lab.semantics import load_semantics
from causal_schedule_lab.solvers.dispatching import solve_dispatching
from causal_schedule_lab.validation import MultiFidelityValidator


ROOT = Path(__file__).resolve().parents[1]


def test_controller_runs_without_external_project() -> None:
    semantics = load_semantics(ROOT / "configs" / "base_semantics.json")
    problem = example_problems()["jsp"]
    incumbent = solve_dispatching(problem, rule="lpt")
    action_index = ActionIndex.from_semantics(semantics)
    controller = AgenticImprovementController(
        project_id="generic-scheduling-project",
        discoverer=CausalCoreDiscoverer(semantics),
        action_index=action_index,
        policy=MaskedPPOAgent(action_index, seed=0),
        generator=GenericCPSATRepairGenerator(
            seed=0,
            deterministic_time=0.1,
        ),
        validator=MultiFidelityValidator(
            semantics=semantics,
            project_root=str(ROOT),
        ),
    )
    result = controller.run(
        problem=problem,
        incumbent=incumbent,
        max_iterations=1,
        candidate_budget=2,
    )
    assert result.best_schedule.problem_id == problem.id

    no_full = controller.run(
        problem=problem,
        incumbent=incumbent,
        max_iterations=1,
        candidate_budget=1,
        full_oracle_budget=0,
    )
    assert no_full.best_schedule == incumbent
    assert all(
        verification.fidelity.value != "full"
        for record in no_full.records
        for verification in record.verifications
    )
