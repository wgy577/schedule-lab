from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import ProjectSemantics
from .semantics import load_semantics


class ProjectManifest(BaseModel):
    """Declarative entry point for any scheduling project.

    Paths are resolved relative to the manifest file.  A custom adapter is a
    normal Python ``module:object`` plugin and therefore does not require a
    change to the core controller.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    project_id: str
    adapter: str = "schedule-lab-json"
    semantics: str
    problem: str | None = None
    schedule: str | None = None
    adapter_options: dict[str, Any] = Field(default_factory=dict)
    generator: str = "generic-cp-sat"
    generator_options: dict[str, Any] = Field(default_factory=dict)
    domain_oracle: str | None = None
    domain_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ProjectContext:
    manifest_path: Path
    manifest: ProjectManifest
    problem: Any
    incumbent: Any
    semantics: ProjectSemantics
    project_root: Path


class SchedulingProjectAdapter(Protocol):
    def load(
        self,
        manifest: ProjectManifest,
        *,
        manifest_path: Path,
    ) -> tuple[Any, Any]: ...


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


class CanonicalJSONAdapter:
    """Load canonical Problem/Schedule JSON without family-specific assumptions."""

    def load(
        self,
        manifest: ProjectManifest,
        *,
        manifest_path: Path,
    ) -> tuple[Any, Any]:
        base = manifest_path.parent
        problem_path = _resolve(base, manifest.problem)
        schedule_path = _resolve(base, manifest.schedule)
        if problem_path is None or schedule_path is None:
            raise ValueError("canonical-json requires problem and schedule paths")
        from .io import load_problem, load_schedule

        problem = load_problem(problem_path)
        schedule = load_schedule(schedule_path)
        if schedule.problem_id != problem.id:
            raise ValueError("schedule problem_id does not match problem id")
        return problem, schedule


class BuiltinExampleAdapter:
    """Small deterministic fixtures used by regression tests and tutorials."""

    def load(
        self,
        manifest: ProjectManifest,
        *,
        manifest_path: Path,
    ) -> tuple[Any, Any]:
        from .benchmarks import example_problems
        from .solvers.dispatching import solve_dispatching

        family = str(manifest.adapter_options.get("family", "fjsp")).lower()
        rule = str(manifest.adapter_options.get("baseline_rule", "lpt"))
        problems = example_problems()
        if family not in problems:
            raise ValueError(f"unknown built-in family: {family}")
        problem = problems[family]
        return problem, solve_dispatching(problem, rule=rule)


BUILTIN_ADAPTERS: dict[str, SchedulingProjectAdapter] = {
    "canonical-json": CanonicalJSONAdapter(),
    "schedule-lab-json": CanonicalJSONAdapter(),
    "builtin-example": BuiltinExampleAdapter(),
}


def load_object(specification: str) -> Any:
    if ":" not in specification:
        raise ValueError("plugin specification must be module:object")
    module_name, object_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def resolve_adapter(name: str) -> SchedulingProjectAdapter:
    if name in BUILTIN_ADAPTERS:
        return BUILTIN_ADAPTERS[name]
    candidate = load_object(name)
    return candidate() if isinstance(candidate, type) else candidate


def load_project(path: str | Path) -> ProjectContext:
    manifest_path = Path(path).expanduser().resolve()
    manifest = ProjectManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    adapter = resolve_adapter(manifest.adapter)
    problem, incumbent = adapter.load(
        manifest,
        manifest_path=manifest_path,
    )
    semantics_path = _resolve(manifest_path.parent, manifest.semantics)
    assert semantics_path is not None
    semantics = load_semantics(semantics_path)
    if semantics.project_id not in {manifest.project_id, "generic-scheduling-project"}:
        raise ValueError(
            "manifest and semantic project ids differ; use generic-scheduling-project "
            "only for the reusable base semantics"
        )
    if problem.kind not in semantics.problem_families:
        raise ValueError(
            f"problem family {problem.kind} is not declared by project semantics"
        )
    return ProjectContext(
        manifest_path=manifest_path,
        manifest=manifest,
        problem=problem,
        incumbent=incumbent,
        semantics=semantics,
        project_root=manifest_path.parent,
    )
