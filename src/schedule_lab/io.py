from __future__ import annotations

from pathlib import Path

from .model import Problem, Schedule


def load_problem(path: str | Path) -> Problem:
    return Problem.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_schedule(path: str | Path) -> Schedule:
    return Schedule.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_problem(problem: Problem, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(problem.model_dump_json(indent=2), encoding="utf-8")


def save_schedule(schedule: Schedule, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(schedule.model_dump_json(indent=2), encoding="utf-8")
