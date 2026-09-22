"""FJSP target-project adapter: instances -> IR, faithful solver, FJSP Oracle.

This module bridges the third-party FJSP-DRL/DANIEL repository
(`external_cases/fjsp_dan/repository`) to the standalone ``ir.Problem`` /
``ir.Schedule`` without modifying the target project's source files.

It provides three things the second-stage pilot needs:

* :func:`load_fjsp_problem` reads a Brandimarte ``.fjs`` instance and builds a
  canonical FJSP :class:`~causal_schedule_lab.ir.Problem` via
  :func:`~causal_schedule_lab.benchmarks.build_fjsp`.
* :class:`FJSPProjectSolver` is a faithful re-implementation of the project's
  ``ortools_solver.fjsp_solver`` CP-SAT model that *keeps* the full assignment
  (the original discards it) and records the solve status and best bound, so it
  closes technical-debt items #5 (independent FJSP validator) and #6 (CP-SAT
  status not recorded). It is the project-solver incumbent producer.
* :func:`cross_check_oracle` independently re-solves with the lab's own
  closure-aware :func:`~causal_schedule_lab.solvers.cp_sat.solve_cp_sat` from
  scratch (no incumbent hint) and reports the objective bound, as a Full-Oracle
  cross check; :func:`validate_fjsp` wires the generic hard-constraint validator.

The second-stage external-case pipeline additionally needs a *fixed, reusable*
conversion entry that turns a project inference run into canonical, auditable
artifacts without ever trusting the rendered PNG or an unreviewed LLM prior:

* :func:`schedule_from_fjsp_drl_payload` maps a project schedule artifact onto
  the canonical :class:`~causal_schedule_lab.ir.Schedule`, recovering the mode
  only from the full ``Problem`` and rejecting any inconsistency fail-closed.
* :func:`build_serialized_gantt` emits an image-independent Gantt serialization
  keeping every operation's stable identifiers, machine/job sequences and the
  independent validation record.
* :func:`build_unified_case_representation` reuses the project's single
  :class:`~causal_schedule_lab.unified_representation.UnifiedSchedulingRepresentation`
  (problem + Gantt overlay + UTSEG + family/capability + feasibility masks +
  operator masks) and wraps it with source/derived provenance, never a second
  schema.
* :func:`convert_fjsp_drl_case` is the fixed entry point: new projects only
  swap paths/parameters and receive the three artifacts plus a report.

Everything operates on the family-agnostic IR; nothing here is FJSP-specific
beyond the instance parser and the project's modelling choices.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from ..benchmarks import build_fjsp
from ..core_validation import ValidationReport, validate_schedule
from ..ir import Assignment, Problem, Schedule
from ..models import FailureLabel, Fidelity, VerificationResult
from ..solvers.cp_sat import solve_cp_sat
from ..unified_representation import (
    UnifiedSchedulingRepresentation,
    build_unified_representation,
)

_MACHINE_TAG = "fjsp_machine"

# Every conversion artifact carries this so downstream audits know the payload
# was produced by the fixed entry point, not hand-edited. The label is about
# provenance of the *conversion*, never a claim that any LLM artifact was
# human-approved.
CONVERSION_GENERATOR = "causal_schedule_lab.ir_adapters.fjsp_drl.convert_fjsp_drl_case"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes used for output hashing and writing."""

    return json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=False
    ).encode("utf-8")


def parse_fjs_file(path: str | Path) -> tuple[list[list[list[tuple[str, int]]]], int]:
    """Parse a standard Brandimarte ``.fjs`` text instance.

    Returns ``(alternatives, num_machines)`` where ``alternatives`` matches
    :func:`~causal_schedule_lab.benchmarks.build_fjsp`: a per-job, per-operation
    list of ``(machine_id, processing_time)`` alternatives. Machine ids in the
    file are 1-based and are normalised to ``M<int>`` resource ids.
    """

    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    numbers = [int(token) for token in re.findall(r"\d+", text[0])]
    if len(numbers) < 2:
        raise ValueError(f"{path}: first line must start with n_jobs n_machines")
    num_jobs, num_machines = numbers[0], numbers[1]
    alternatives: list[list[list[tuple[str, int]]]] = []
    for job_index in range(num_jobs):
        line = numbers_line(text, job_index + 2)
        if line is None:
            raise ValueError(f"{path}: missing job line {job_index + 1}")
        tokens = [int(token) for token in re.findall(r"\d+", line)]
        if not tokens:
            raise ValueError(f"{path}: empty job line {job_index + 1}")
        cursor = 0
        operation_count = tokens[cursor]
        cursor += 1
        job_ops: list[list[tuple[str, int]]] = []
        for _operation in range(operation_count):
            if cursor >= len(tokens):
                break
            eligible = tokens[cursor]
            cursor += 1
            choices: list[tuple[str, int]] = []
            for _ in range(eligible):
                if cursor + 1 >= len(tokens):
                    break
                machine_id = f"M{tokens[cursor]}"
                duration = tokens[cursor + 1]
                choices.append((machine_id, duration))
                cursor += 2
            if not choices:
                raise ValueError(f"{path}: operation with no eligible machine")
            job_ops.append(choices)
        alternatives.append(job_ops)
    return alternatives, num_machines


def numbers_line(text: list[str], index: int) -> str | None:
    """Return the ``index``-th non-empty line (1-based)."""

    seen = 0
    for line in text:
        if line.strip():
            seen += 1
            if seen == index:
                return line
    return None


def load_fjsp_problem(path: str | Path, *, problem_id: str | None = None) -> Problem:
    """Build a canonical FJSP :class:`Problem` from a Brandimarte ``.fjs`` file."""

    alternatives, num_machines = parse_fjs_file(path)
    problem = build_fjsp(alternatives, problem_id=problem_id or Path(path).stem)
    return problem.model_copy(
        update={
            "resources": tuple(
                resource.model_copy(update={"family": "machine", "tags": (_MACHINE_TAG,)})
                for resource in problem.resources
            )
        }
    )


@dataclass(frozen=True)
class FJSPSolverResult:
    schedule: Schedule | None
    makespan: int | None
    status: str
    objective_bound: float | None
    conflicts: int
    branches: int
    wall_time: float
    time_limit: float


class FJSPProjectSolver:
    """Faithful project CP-SAT solver that keeps the full assignment.

    Re-implements ``ortools_solver.fjsp_solver`` (alternative intervals per
    machine, ``exactly_one`` presence, ``no_overlap`` per machine, minimise
    makespan) but returns a complete :class:`Schedule` together with the solver
    status and best bound that the original discards.
    """

    def __init__(self, *, time_limit: float = 30.0, seed: int = 0) -> None:
        self.time_limit = time_limit
        self.seed = seed

    def solve(self, problem: Problem) -> FJSPSolverResult:
        model = cp_model.CpModel()
        horizon = sum(
            max(mode.duration for mode in operation.modes)
            for operation in problem.operations
        ) or 1

        starts: dict[str, cp_model.IntVar] = {}
        ends: dict[str, cp_model.IntVar] = {}
        presences: dict[str, cp_model.BoolVar] = {}
        intervals_per_machine: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)
        job_ends: list[cp_model.IntVar] = []

        jobs = problem.job_map()
        for operation in problem.operations:
            job = jobs[operation.job_id]
            release = max(operation.release, job.release)
            start = model.new_int_var(release, horizon, f"start:{operation.id}")
            end = model.new_int_var(release + 1, horizon, f"end:{operation.id}")
            starts[operation.id] = start
            ends[operation.id] = end
            for mode in operation.modes:
                presence = model.new_bool_var(f"presence:{mode.id}")
                presences[mode.id] = presence
                interval = model.new_optional_interval_var(
                    start, mode.duration, end, presence, f"interval:{mode.id}"
                )
                for resource in mode.resources:
                    intervals_per_machine[resource].append(interval)
            model.add_exactly_one([presences[mode.id] for mode in operation.modes])
            for predecessor in operation.predecessors:
                model.add(start >= ends[predecessor])
            job_ends.append(end)

        for machine in problem.resources:
            intervals = intervals_per_machine.get(machine.id, [])
            if len(intervals) > 1:
                model.add_no_overlap(intervals)

        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, job_ends)
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.random_seed = self.seed
        # Single worker for deterministic, reproducible research runs.
        solver.parameters.num_search_workers = 1
        status_code = solver.solve(model)
        status = solver.status_name(status_code)

        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return FJSPSolverResult(
                schedule=None,
                makespan=None,
                status=status,
                objective_bound=solver.best_objective_bound,
                conflicts=solver.num_conflicts,
                branches=solver.num_branches,
                wall_time=solver.wall_time,
                time_limit=self.time_limit,
            )

        assignments: list[Assignment] = []
        for operation in problem.operations:
            chosen = next(
                mode
                for mode in operation.modes
                if solver.boolean_value(presences[mode.id])
            )
            assignments.append(
                Assignment(
                    operation_id=operation.id,
                    mode_id=chosen.id,
                    start=solver.value(starts[operation.id]),
                    end=solver.value(ends[operation.id]),
                    provenance="fjsp-project-cp-sat",
                )
            )
        schedule = Schedule(
            problem_id=problem.id,
            assignments=tuple(assignments),
            metadata={
                "solver": "fjsp-project-cp-sat",
                "seed": self.seed,
                "time_limit": self.time_limit,
                "status": status,
            },
        )
        return FJSPSolverResult(
            schedule=schedule,
            makespan=schedule.makespan,
            status=status,
            objective_bound=solver.best_objective_bound,
            conflicts=solver.num_conflicts,
            branches=solver.num_branches,
            wall_time=solver.wall_time,
            time_limit=self.time_limit,
        )


def validate_fjsp(problem: Problem, schedule: Schedule) -> ValidationReport:
    """Independent FJSP feasibility gate (wraps the generic hard validator)."""

    return validate_schedule(problem, schedule)


def _exact_integer(value: Any, *, field: str, operation_id: str) -> int:
    """Return an exact integer time or reject lossy schedule conversion."""

    if isinstance(value, bool):
        raise ValueError(f"{operation_id}: {field} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{operation_id}: {field}={value!r} is not an exact integer")


def schedule_from_fjsp_drl_payload(
    problem: Problem,
    payload: dict[str, Any],
    *,
    provenance: str = "fjsp-drl-hgnn-ppo",
    metadata: dict[str, Any] | None = None,
    require_complete: bool = True,
) -> Schedule:
    """Convert an FJSP-DRL schedule artifact into the canonical Schedule IR.

    The original artifact names the selected machine but not the canonical mode.
    This function recovers the mode only from the full ``Problem`` parsed from the
    source ``.fjs`` file. Every assignment is checked fail-closed against the
    canonical instance:

    * the operation id must exist and be listed at most once;
    * any declared ``job_id`` / ``operation_index`` must match the canonical
      operation (the artifact uses 1-based operation numbers, the IR is 0-based);
    * the named machine must be eligible for the operation;
    * ``start`` / ``end`` must be exact integers and any declared ``duration``
      must agree with them;
    * the realised duration must equal the source mode duration.

    With ``require_complete`` (the default) every canonical operation must be
    scheduled exactly once. The generic hard-constraint validator then runs as an
    independent feasibility gate. Thus a chosen schedule can neither silently
    redefine the instance's machine alternatives / processing times nor quietly
    drop operations that a rendered image might hide.
    """

    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("FJSP-DRL payload must contain an assignments list")
    operations = problem.operation_map()
    canonical: list[Assignment] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_assignments):
        if not isinstance(raw, dict):
            raise ValueError(f"assignment[{index}] must be an object")
        operation_id = str(raw.get("operation_id", ""))
        machine_id = str(raw.get("machine_id", ""))
        operation = operations.get(operation_id)
        if operation is None:
            raise ValueError(f"unknown operation in FJSP-DRL schedule: {operation_id!r}")
        if operation_id in seen:
            raise ValueError(f"operation scheduled more than once: {operation_id!r}")
        seen.add(operation_id)
        declared_job = raw.get("job_id")
        if declared_job is not None and str(declared_job) != operation.job_id:
            raise ValueError(
                f"{operation_id}: declared job_id {declared_job!r} does not match "
                f"canonical job {operation.job_id!r}"
            )
        declared_index = raw.get("operation_index")
        if declared_index is not None and int(declared_index) != operation.index + 1:
            raise ValueError(
                f"{operation_id}: declared operation_index {declared_index!r} does "
                f"not match canonical 1-based index {operation.index + 1}"
            )
        eligible = {
            mode.resources[0]: mode
            for mode in operation.modes
            if len(mode.resources) == 1
        }
        mode = eligible.get(machine_id)
        if mode is None:
            raise ValueError(
                f"{operation_id}: machine {machine_id!r} is not eligible; "
                f"expected one of {sorted(eligible)}"
            )
        start = _exact_integer(raw.get("start"), field="start", operation_id=operation_id)
        end = _exact_integer(raw.get("end"), field="end", operation_id=operation_id)
        if "duration" in raw:
            declared_duration = _exact_integer(
                raw.get("duration"), field="duration", operation_id=operation_id
            )
            if declared_duration != end - start:
                raise ValueError(
                    f"{operation_id}: declared duration {declared_duration} does not "
                    f"match end-start {end - start}"
                )
        if end - start != mode.duration:
            raise ValueError(
                f"{operation_id}: scheduled duration {end - start} does not match "
                f"source mode {mode.id} duration {mode.duration}"
            )
        canonical.append(
            Assignment(
                operation_id=operation_id,
                mode_id=mode.id,
                start=start,
                end=end,
                provenance=provenance,
                metadata={
                    "source_machine_id": machine_id,
                    "source_global_operation_index": raw.get("global_operation_index"),
                },
            )
        )
    if require_complete:
        missing = sorted(set(operations) - seen)
        if missing:
            raise ValueError(
                f"FJSP-DRL schedule is incomplete: {len(missing)} operation(s) "
                f"unscheduled, e.g. {missing[:5]}"
            )
    schedule = Schedule(
        problem_id=problem.id,
        assignments=tuple(canonical),
        metadata={
            "source_schema_version": payload.get("schema_version"),
            "source_makespan": payload.get("makespan"),
            "source_gantt_valid": payload.get("gantt_valid"),
            "source_precedence_violations": payload.get("precedence_violations"),
            **(metadata or {}),
        },
    )
    report = validate_fjsp(problem, schedule)
    if not report.feasible:
        raise ValueError(f"canonical FJSP schedule is infeasible: {report.as_dict()['errors']}")
    return schedule


def build_serialized_gantt(
    problem: Problem,
    schedule: Schedule,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete, image-independent Gantt serialization.

    Every operation appears once in ``assignments`` and is also indexed by job
    and resource. Each row carries a stable ``operation_number`` (1-based
    ``J{job}.O{index+1}`` position) so short operations whose text labels may be
    hidden in a rendered PNG still have a deterministic, machine-readable
    identity. The machine and job sequences are deterministic and retain exact
    canonical IDs. The independent feasibility report is embedded, and the source
    makespan is cross-checked when present.
    """

    report = validate_fjsp(problem, schedule)
    if not report.feasible:
        raise ValueError(f"cannot serialize infeasible schedule: {report.as_dict()['errors']}")
    operation_map = problem.operation_map()
    mode_map = problem.mode_map()
    rows: list[dict[str, Any]] = []
    for assignment in sorted(
        schedule.assignments,
        key=lambda item: (item.start, item.end, item.operation_id),
    ):
        operation = operation_map[assignment.operation_id]
        _, mode = mode_map[assignment.mode_id]
        rows.append(
            {
                "operation_id": operation.id,
                "operation_number": f"{operation.job_id}.O{operation.index + 1}",
                "job_id": operation.job_id,
                "operation_index": operation.index,
                "mode_id": mode.id,
                "machine_id": mode.resources[0],
                "start": assignment.start,
                "end": assignment.end,
                "duration": assignment.end - assignment.start,
                "predecessors": list(operation.predecessors),
                "provenance": assignment.provenance,
                "source_machine_id": assignment.metadata.get("source_machine_id"),
                "source_global_operation_index": assignment.metadata.get(
                    "source_global_operation_index"
                ),
            }
        )
    # Machine sequence: exact per-resource order, keeping start/end so hidden
    # short bars remain fully reconstructable without the image.
    by_machine: dict[str, list[dict[str, Any]]] = {item.id: [] for item in problem.resources}
    by_job: dict[str, list[str]] = {item.id: [] for item in problem.jobs}
    for row in rows:
        by_machine[row["machine_id"]].append(
            {
                "operation_id": row["operation_id"],
                "start": row["start"],
                "end": row["end"],
                "duration": row["duration"],
            }
        )
    for row in sorted(rows, key=lambda item: (item["job_id"], item["operation_index"])):
        by_job[row["job_id"]].append(row["operation_id"])
    makespan = schedule.makespan
    source_makespan = schedule.metadata.get("source_makespan")
    source_makespan_matches: bool | None = None
    if source_makespan is not None:
        source_makespan_matches = float(source_makespan) == float(makespan)
    return {
        "schema_version": "canonical-gantt-1.0",
        "problem_id": problem.id,
        "problem_family": problem.kind,
        "time_scale": problem.time_scale,
        "makespan": makespan,
        "source_makespan": source_makespan,
        "source_makespan_matches": source_makespan_matches,
        "operation_count": len(problem.operations),
        "assignment_count": len(rows),
        "complete": len(rows) == len(problem.operations),
        "image_independent": True,
        "validation": report.as_dict(),
        "assignments": rows,
        "sequences": {"by_machine": by_machine, "by_job": by_job},
        "metadata": metadata or {},
    }


def cross_check_oracle(
    problem: Problem,
    schedule: Schedule,
    *,
    max_deterministic_time: float = 2.0,
    seed: int = 0,
) -> VerificationResult:
    """Full-Oracle cross check via an independent from-scratch CP-SAT solve.

    Re-solves the same IR with the lab's closure-aware solver (no incumbent
    hint) and compares its objective bound against the incumbent makespan. The
    feasibility of ``schedule`` itself is also checked.
    """

    feasibility = validate_fjsp(problem, schedule)
    failures: list[FailureLabel] = []
    if not feasibility.feasible:
        failures.append(FailureLabel.PRECEDENCE_VIOLATION)
    independent = solve_cp_sat(
        problem,
        incumbent=None,
        seed=seed,
        max_deterministic_time=max_deterministic_time,
    )
    incumbent_makespan = schedule.makespan / max(1, problem.time_scale)
    oracle_bound = None
    if independent.schedule is not None:
        oracle_bound = independent.schedule.makespan / max(1, problem.time_scale)
    elif independent.objective_bound is not None:
        oracle_bound = independent.objective_bound / max(1, problem.time_scale)
    return VerificationResult(
        fidelity=Fidelity.FULL,
        passed=feasibility.feasible,
        proxy_delta=None,
        true_delta=oracle_bound - incumbent_makespan if oracle_bound is not None else None,
        objective=oracle_bound,
        failures=tuple(failures),
        details={
            "incumbent_makespan": incumbent_makespan,
            "oracle_bound": oracle_bound,
            "oracle_status": independent.status,
            "oracle_schedule_produced": independent.schedule is not None,
            "feasibility_errors": feasibility.as_dict()["errors"],
        },
        runtime_seconds=float(independent.deterministic_time),
    )


def load_semantic_prior(path: str | Path) -> dict[str, Any]:
    """Load an LLM semantic-compilation artifact as a *non-authoritative* prior.

    The returned record only reports the artifact's own ``review_status`` and a
    hash. ``approved`` is ``True`` only if the artifact itself already carries an
    explicit ``approved`` review status; a ``pending_human_review`` artifact is
    never silently promoted. Nothing here overrides the parsed instance, the
    canonical IR or the independent validator.
    """

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    status = str(raw.get("review_status", "unknown"))
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": raw.get("schema_version"),
        "review_status": status,
        "approved": status == "approved",
        "role": "semantic_prior",
        "authoritative": False,
        "note": (
            "Semantic prior only. Does not override code, parsed instance, "
            "canonical IR or the independent validator."
        ),
    }


# Which unified-representation fields are direct source facts and which are
# deterministically derived by the lab. This lets an auditor see that the LLM
# prior contributed nothing to the payload's authoritative structure.
UNIFIED_FIELD_PROVENANCE: dict[str, dict[str, str]] = {
    "problem": {
        "kind": "source_fact",
        "origin": "parsed from the Brandimarte-format .fjs instance",
    },
    "schedule": {
        "kind": "source_fact",
        "origin": "the project's own inference run (machine + start/end per operation)",
    },
    "graph": {
        "kind": "derived",
        "origin": "UTSEG built deterministically from problem+schedule (graph.build_utseg)",
    },
    "family": {
        "kind": "derived",
        "origin": "capability signature inferred from the canonical problem",
    },
    "feasibility": {
        "kind": "derived",
        "origin": "deterministic state masks at decision_time (no enumeration of schedules)",
    },
    "operators": {
        "kind": "derived",
        "origin": "family/capability labels combined with current-Gantt preconditions",
    },
}


def build_unified_case_representation(
    problem: Problem,
    schedule: Schedule,
    *,
    project_id: str,
    decision_time: int = 0,
    assignment_order_seed: int = 0,
    semantic_prior: dict[str, Any] | None = None,
) -> tuple[UnifiedSchedulingRepresentation, dict[str, Any]]:
    """Build the case's unified representation and its provenance envelope.

    The representation is the project's single
    :class:`UnifiedSchedulingRepresentation` (problem + Gantt overlay + UTSEG +
    family/capability + feasibility masks + operator masks); no second schema is
    introduced. The envelope documents that this is a ``decision_time`` complete
    incumbent snapshot, records which fields are source facts vs derived, and
    carries the (non-authoritative) semantic-prior status.
    """

    representation = build_unified_representation(
        problem,
        schedule,
        project_id=project_id,
        decision_time=decision_time,
        assignment_order_seed=assignment_order_seed,
    )
    envelope = {
        "artifact_kind": "unified_scheduling_representation_case",
        "representation_schema_version": representation.schema_version,
        "project_id": project_id,
        "snapshot": {
            "decision_time": decision_time,
            "description": (
                "Complete incumbent snapshot at decision_time="
                f"{decision_time}; every operation is scheduled and frozen, so "
                "the feasibility masks describe the full-Gantt state rather than "
                "a partial-construction state."
            ),
            "released_operations": [],
        },
        "field_provenance": UNIFIED_FIELD_PROVENANCE,
        "semantic_prior": semantic_prior,
    }
    return representation, envelope


@dataclass(frozen=True)
class ConversionArtifacts:
    problem: Problem
    schedule: Schedule
    gantt: dict[str, Any]
    unified: UnifiedSchedulingRepresentation
    report: dict[str, Any]
    gantt_path: Path
    unified_path: Path
    report_path: Path


CONVERSION_BOUNDARIES: tuple[str, ...] = (
    "The rendered PNG is never a source of truth; all operations come from the "
    "parsed instance and the project schedule JSON.",
    "The full Problem (all jobs, operations, precedence, every eligible machine "
    "and its duration) is recovered from the .fjs instance, not reverse-engineered "
    "from the selected schedule.",
    "Feasibility is decided by the lab's independent hard-constraint validator, "
    "not by the project's self-reported gantt_valid flag.",
    "The UTSEG graph is O(n^2) in eligible-operation resource competition, so the "
    "unified representation is large for dense flexible instances; this is an "
    "inherent property of the canonical graph, not a defect.",
    "Any LLM semantic artifact is a non-authoritative prior; its review status is "
    "reported verbatim and it is never treated as human-approved.",
)


def _write_json_artifact(path: Path, payload: Any) -> tuple[Path, str, int]:
    data = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, _sha256_bytes(data), len(data)


def convert_fjsp_drl_case(
    *,
    instance_path: str | Path,
    schedule_path: str | Path,
    gantt_output: str | Path,
    unified_output: str | Path,
    report_output: str | Path,
    problem_id: str | None = None,
    semantic_prior_path: str | Path | None = None,
    run_log_path: str | Path | None = None,
    gantt_image_path: str | Path | None = None,
    decision_time: int = 0,
    assignment_order_seed: int = 0,
    cross_check: bool = True,
    cross_check_time: float = 3.0,
) -> ConversionArtifacts:
    """Fixed, reusable conversion entry for an FJSP-DRL external case.

    A new project only swaps the paths/parameters. The function:

    1. recovers the full canonical :class:`Problem` from the ``.fjs`` instance;
    2. maps the project schedule JSON onto a canonical :class:`Schedule`,
       fail-closed against the instance;
    3. writes an image-independent Gantt serialization;
    4. writes the project's single unified representation with a provenance
       envelope (source facts vs derived, non-authoritative semantic prior);
    5. writes a conversion report with input/output hashes, counts, validation
       and boundaries.

    The rendered PNG (``gantt_image_path``) is only hashed for provenance; it is
    never parsed as a source of truth.
    """

    instance_path = Path(instance_path)
    schedule_path = Path(schedule_path)
    resolved_problem_id = problem_id or instance_path.stem

    problem = load_fjsp_problem(instance_path, problem_id=resolved_problem_id)
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))

    instance_sha = _sha256_file(instance_path)
    declared_instance_sha = payload.get("instance_sha256")
    instance_hash_matches: bool | None = None
    if isinstance(declared_instance_sha, str):
        instance_hash_matches = declared_instance_sha == instance_sha

    schedule = schedule_from_fjsp_drl_payload(
        problem,
        payload,
        metadata={
            "source_instance": str(instance_path),
            "source_schedule": str(schedule_path),
        },
    )
    feasibility = validate_fjsp(problem, schedule)

    gantt = build_serialized_gantt(
        problem,
        schedule,
        metadata={
            "generator": CONVERSION_GENERATOR,
            "source_instance": str(instance_path),
            "source_instance_sha256": instance_sha,
            "source_schedule": str(schedule_path),
            "source_schedule_sha256": _sha256_file(schedule_path),
            "source_gantt_image": str(gantt_image_path) if gantt_image_path else None,
            "source_gantt_image_sha256": (
                _sha256_file(gantt_image_path) if gantt_image_path else None
            ),
            "image_is_not_source_of_truth": True,
        },
    )

    semantic_prior = (
        load_semantic_prior(semantic_prior_path) if semantic_prior_path else None
    )
    unified, unified_envelope = build_unified_case_representation(
        problem,
        schedule,
        project_id=resolved_problem_id,
        decision_time=decision_time,
        assignment_order_seed=assignment_order_seed,
        semantic_prior=semantic_prior,
    )
    unified_document = {
        "provenance": {
            "generator": CONVERSION_GENERATOR,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_instance": str(instance_path),
            "source_instance_sha256": instance_sha,
            "source_schedule": str(schedule_path),
            "source_schedule_sha256": _sha256_file(schedule_path),
            **unified_envelope,
        },
        "representation": unified.model_dump(mode="json"),
    }

    oracle_summary: dict[str, Any] | None = None
    if cross_check:
        oracle = cross_check_oracle(
            problem, schedule, max_deterministic_time=cross_check_time
        )
        oracle_summary = {
            "fidelity": oracle.fidelity.value,
            "passed": oracle.passed,
            "incumbent_makespan": oracle.details.get("incumbent_makespan"),
            "oracle_bound": oracle.details.get("oracle_bound"),
            "oracle_status": oracle.details.get("oracle_status"),
            "true_delta": oracle.true_delta,
            "note": (
                "Independent from-scratch CP-SAT bound. A bound below the "
                "incumbent only means the project schedule is not optimal, not "
                "that it is infeasible."
            ),
        }

    gantt_path, gantt_sha, gantt_bytes = _write_json_artifact(Path(gantt_output), gantt)
    unified_path, unified_sha, unified_bytes = _write_json_artifact(
        Path(unified_output), unified_document
    )

    report = {
        "schema_version": "fjsp-drl-conversion-report-1.0",
        "generator": CONVERSION_GENERATOR,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "instance": {
                "path": str(instance_path),
                "sha256": instance_sha,
                "declared_sha256": declared_instance_sha,
                "declared_sha256_matches": instance_hash_matches,
            },
            "schedule": {
                "path": str(schedule_path),
                "sha256": _sha256_file(schedule_path),
                "source_schema_version": payload.get("schema_version"),
                "source_repository_commit": payload.get("source_repository_commit"),
                "source_makespan": payload.get("makespan"),
                "source_gantt_valid": payload.get("gantt_valid"),
                "source_precedence_violations": payload.get("precedence_violations"),
            },
            "gantt_image": (
                {
                    "path": str(gantt_image_path),
                    "sha256": _sha256_file(gantt_image_path),
                    "role": "reference_only_not_source_of_truth",
                }
                if gantt_image_path
                else None
            ),
            "run_log": (
                {"path": str(run_log_path), "sha256": _sha256_file(run_log_path)}
                if run_log_path
                else None
            ),
        },
        "problem": {
            "id": problem.id,
            "kind": problem.kind,
            "jobs": len(problem.jobs),
            "resources": len(problem.resources),
            "operations": len(problem.operations),
            "modes": sum(len(op.modes) for op in problem.operations),
        },
        "schedule": {
            "assignments": len(schedule.assignments),
            "makespan": schedule.makespan,
            "source_makespan_matches": gantt["source_makespan_matches"],
        },
        "validation": {
            "independent_validator_feasible": feasibility.feasible,
            "errors": feasibility.as_dict()["errors"],
            "warnings": feasibility.as_dict()["warnings"],
        },
        "cross_check_oracle": oracle_summary,
        "semantic_prior": semantic_prior,
        "family": {
            "declared": unified.family.declared_family,
            "inferred": unified.family.inferred_family,
            "consistent": unified.family.consistent,
        },
        "outputs": {
            "gantt": {
                "path": str(gantt_path),
                "sha256": gantt_sha,
                "bytes": gantt_bytes,
            },
            "unified_representation": {
                "path": str(unified_path),
                "sha256": unified_sha,
                "bytes": unified_bytes,
                "graph_nodes": len(unified.graph.nodes),
                "graph_edges": len(unified.graph.edges),
                "graph_constraints": len(unified.graph.constraints),
            },
        },
        "boundaries": list(CONVERSION_BOUNDARIES),
    }
    report_path, report_sha, _ = _write_json_artifact(Path(report_output), report)
    report["self_sha256"] = report_sha

    return ConversionArtifacts(
        problem=problem,
        schedule=schedule,
        gantt=gantt,
        unified=unified,
        report=report,
        gantt_path=gantt_path,
        unified_path=unified_path,
        report_path=report_path,
    )


__all__ = [
    "parse_fjs_file",
    "load_fjsp_problem",
    "FJSPProjectSolver",
    "FJSPSolverResult",
    "validate_fjsp",
    "schedule_from_fjsp_drl_payload",
    "build_serialized_gantt",
    "load_semantic_prior",
    "build_unified_case_representation",
    "convert_fjsp_drl_case",
    "ConversionArtifacts",
    "cross_check_oracle",
]
