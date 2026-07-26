# TEST-TAGS: modules=C,F; capabilities=canonical_ir,dispatching,cp_sat,hard_constraints; level=integration; cost=medium
from causal_schedule_lab.benchmarks import benchmark_suite, example_problems
from causal_schedule_lab.core_validation import validate_schedule
from causal_schedule_lab.ir import ConstraintSpec
from causal_schedule_lab.solvers.cp_sat import solve_cp_sat
from causal_schedule_lab.solvers.dispatching import solve_dispatching


def test_all_supported_families_have_valid_deterministic_baselines() -> None:
    for family, problem in example_problems().items():
        first = solve_dispatching(problem, rule="lpt")
        second = solve_dispatching(problem, rule="lpt")
        assert first == second, family
        assert validate_schedule(problem, first).feasible, family


def test_cp_sat_repairs_a_bounded_neighborhood() -> None:
    problem = example_problems()["fjsp"]
    incumbent = solve_dispatching(problem, rule="lpt")
    released = {item.id for item in problem.operations if item.job_id == "J2"}
    result = solve_cp_sat(
        problem,
        incumbent=incumbent,
        frozen_operation_ids={item.id for item in problem.operations} - released,
        seed=7,
        workers=1,
        max_deterministic_time=0.1,
    )
    assert result.schedule is not None
    assert validate_schedule(problem, result.schedule).feasible
    before = incumbent.assignment_map()
    after = result.schedule.assignment_map()
    for operation in set(before) - released:
        assert (
            before[operation].mode_id,
            before[operation].start,
            before[operation].end,
        ) == (
            after[operation].mode_id,
            after[operation].start,
            after[operation].end,
        )

    replay = solve_cp_sat(
        problem,
        incumbent=incumbent,
        frozen_operation_ids={item.id for item in problem.operations} - released,
        seed=7,
        workers=1,
        max_deterministic_time=0.1,
    )
    assert replay.schedule == result.schedule


def test_seeded_benchmark_suite_is_reproducible() -> None:
    left = benchmark_suite(seed=11, instances_per_family=2)
    right = benchmark_suite(seed=11, instances_per_family=2)
    assert left == right


def test_time_window_and_no_wait_constraints_are_solver_encoded() -> None:
    base = example_problems()["jsp"]
    first_job = sorted(
        (item for item in base.operations if item.job_id == "J1"),
        key=lambda item: item.index,
    )
    problem = base.model_copy(
        update={
            "id": "constrained-jsp",
            "constraints": (
                ConstraintSpec(
                    id="window",
                    kind="time_window",
                    scope=(first_job[0].id,),
                    parameters={"earliest_start": 2, "latest_start": 4},
                ),
                ConstraintSpec(
                    id="no-wait",
                    kind="no_wait",
                    scope=(first_job[0].id, first_job[1].id),
                ),
            ),
        }
    )
    result = solve_cp_sat(
        problem,
        seed=0,
        workers=1,
        max_deterministic_time=0.1,
    )
    assert result.schedule is not None
    report = validate_schedule(problem, result.schedule)
    assert report.feasible
    values = result.schedule.assignment_map()
    assert 2 <= values[first_job[0].id].start <= 4
    assert values[first_job[1].id].start == values[first_job[0].id].end
