"""Bounded, resumable DeepSeek-assisted trajectory harvesting runner V1."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..ir import Problem, Schedule
from ..memory import DIRECT_SUCCESS, DELAYED_SUCCESS, FAILURE, ExperienceStore, encode_state
from ..sg_sct_model_v5 import run_m2_v5_schedule
from ..symptom_pruning import diagnose_and_prune
from ..validation import schedule_hash
from .llm_experience_generator_v1 import (
    PROMPT_VERSION,
    LLMAssistedTrajectoryPipeline,
    LLMExperienceGenerator,
)


@dataclass(frozen=True)
class HarvestGenerationConfig:
    enabled: bool = True
    provider: str = "deepseek"
    candidates_per_state: int = 10
    max_instances_per_run: int = 100
    temperature: float = 0.2
    retry: int = 2
    dry_run: bool = True
    solver_time: float = 1.0
    seed: int = 0
    prioritize_appearance: str = "A2"
    operator_target_distribution: Mapping[str, float] = field(default_factory=lambda: {
        "routing": 0.6, "sequencing": 0.2, "insertion": 0.1, "timing": 0.1,
    })
    run_id: str = "deepseek-harvest-v1"

    def validate(self) -> None:
        if self.provider.lower() != "deepseek":
            raise ValueError("Trajectory Harvester V1 requires provider=deepseek")
        if self.candidates_per_state < 1 or self.max_instances_per_run < 1:
            raise ValueError("harvesting candidate/instance limits must be positive")
        if not 0.0 <= self.temperature <= 2.0 or self.retry < 1:
            raise ValueError("invalid harvesting temperature/retry")
        if self.solver_time <= 0:
            raise ValueError("experience_harvesting.solver_time must be > 0")
        expected = {"routing", "sequencing", "insertion", "timing"}
        if set(self.operator_target_distribution) != expected:
            raise ValueError("operator target distribution must cover four operators")
        if abs(sum(float(v) for v in self.operator_target_distribution.values()) - 1.0) > 1e-9:
            raise ValueError("operator target distribution must sum to 1")
        if any(float(value) < 0 for value in self.operator_target_distribution.values()):
            raise ValueError("operator target distribution cannot be negative")


def _repository_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "hyperparameters.yaml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


def load_harvest_generation_config(path: str | Path | None = None) -> HarvestGenerationConfig:
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or (_repository_root() / "configs" / "hyperparameters.yaml")
    ).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    section = payload.get("experience_harvesting")
    if not isinstance(section, Mapping):
        raise ValueError("missing experience_harvesting hyperparameters")
    config = HarvestGenerationConfig(
        enabled=bool(section.get("enabled", True)),
        provider=str(section.get("provider", "deepseek")),
        candidates_per_state=int(section.get("candidates_per_state", 10)),
        max_instances_per_run=int(section.get("max_instances_per_run", 100)),
        temperature=float(section.get("temperature", 0.2)),
        retry=int(section.get("retry", 2)),
        dry_run=bool(section.get("dry_run", True)),
        solver_time=float(section.get("solver_time", 1.0)),
        seed=int(section.get("seed", 0)),
        prioritize_appearance=str(section.get("prioritize_appearance", "A2")),
        operator_target_distribution={
            str(key): float(value)
            for key, value in section.get("operator_target_distribution", {}).items()
        },
    )
    config.validate()
    return config


@dataclass(frozen=True)
class HarvestInstance:
    instance_id: str
    problem: Problem
    schedule: Schedule
    split_id: str = "train"
    appearance: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarvestReport:
    run_id: str
    manifest_path: str
    failure_path: str
    candidate_path: str
    memory_path: str | None
    dry_run: bool
    statistics: Mapping[str, Any]
    status: str


M2Inference = Callable[[Problem, Schedule, Mapping[str, Any], int], Any]


def _default_m2_inference(
    problem: Problem, schedule: Schedule, appearance: Mapping[str, Any], seed: int
) -> Any:
    import torch
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        output, _model, _bundle = run_m2_v5_schedule(
            problem, schedule, appearance, case_id=problem.id
        )
    return output


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _appearance_rows(appearance: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in appearance.get("blocks", ()) if isinstance(row, Mapping)]


def _appearance_kind(row: Mapping[str, Any]) -> str:
    block = row.get("block", row)
    rules = block.get("appearance_rules", ()) if isinstance(block, Mapping) else ()
    return str(rules[0]) if rules else "unknown"


def _appearance_id(row: Mapping[str, Any]) -> str:
    block = row.get("block", row)
    return str(block.get("block_id", "")) if isinstance(block, Mapping) else ""


def _select_appearance(
    appearance: Mapping[str, Any], preferred: str
) -> dict[str, Any] | None:
    rows = [row for row in _appearance_rows(appearance)
            if str(row.get("keep_or_prune", "keep")) == "keep"]
    rows.sort(key=lambda row: (
        0 if _appearance_kind(row) == preferred else 1,
        -float(row.get("priority", row.get("score", 0.0)) or 0.0),
        _appearance_id(row),
    ))
    return rows[0] if rows else None


class TrajectoryHarvesterV1:
    """One bounded LLM call per TRAIN instance, with atomic resume artifacts."""

    def __init__(
        self,
        generator: LLMExperienceGenerator,
        store: ExperienceStore,
        *,
        output_dir: str | Path,
        m2_inference: M2Inference | None = None,
    ) -> None:
        self.generator = generator
        self.store = store
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.m2_inference = m2_inference or _default_m2_inference

    def harvest(
        self,
        instances: Iterable[HarvestInstance],
        generation_config: HarvestGenerationConfig | None = None,
    ) -> HarvestReport:
        config = generation_config or load_harvest_generation_config()
        config.validate()
        self.generator.operator_target_distribution = dict(config.operator_target_distribution)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "experience_generation_manifest.json"
        failure_path = self.output_dir / "harvest_failures.jsonl"
        candidate_path = self.output_dir / "harvest_candidates.jsonl"
        manifest = self._initial_manifest(config)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_id") != config.run_id or bool(existing.get("dry_run")) != config.dry_run:
                raise ValueError("existing harvest manifest identity/mode mismatch")
            manifest = existing
        if not config.enabled:
            manifest["status"] = "disabled"
            _atomic_json(manifest_path, manifest)
            return self._report(manifest, manifest_path, failure_path, candidate_path)

        completed = set(manifest.get("completed_state_keys", ()))
        selected_instances = sorted(instances, key=lambda item: item.instance_id)[
            : config.max_instances_per_run
        ]
        pipeline = LLMAssistedTrajectoryPipeline(
            self.generator, self.store, solver_time=config.solver_time, seed=config.seed
        )
        for instance in selected_instances:
            if instance.split_id.lower() == "test" or bool(instance.problem.metadata.get("formal_test", False)):
                self._failure(manifest, failure_path, instance.instance_id, "input", "formal_test_forbidden")
                self._checkpoint(manifest, manifest_path)
                continue
            problem_hash = hashlib.sha256(instance.problem.model_dump_json().encode()).hexdigest()
            incumbent_hash = schedule_hash(instance.schedule)
            state_key = f"{instance.instance_id}:{incumbent_hash}"
            if state_key in completed:
                continue
            manifest["statistics"]["instance_count"] += 1
            try:
                appearance = dict(instance.appearance or diagnose_and_prune(
                    instance.problem, instance.schedule
                ).model_dump(mode="json"))
                selected = _select_appearance(appearance, config.prioritize_appearance)
                if selected is None:
                    raise RuntimeError("no_retained_appearance")
                output = self.m2_inference(
                    instance.problem, instance.schedule, appearance, config.seed
                )
                manifest.setdefault("m2_training_statuses", {})[
                    str(getattr(output, "training_status", "unknown"))
                ] = manifest.setdefault("m2_training_statuses", {}).get(
                    str(getattr(output, "training_status", "unknown")), 0
                ) + 1
                context = self._generation_context(instance, selected, output)
                if not context["root_candidates"] or not context["legal_constraints"]:
                    raise RuntimeError("m2_no_actionable_legal_context")
                if config.dry_run:
                    batch = self.generator.generate(
                        context["state"], context["causal_chains"],
                        context["root_candidates"], context["operator_candidates"],
                        context["legal_constraints"],
                    )
                    valid, rejected = self.generator.validate_candidates(
                        batch, state=context["state"], causal_chains=context["causal_chains"],
                        root_candidates=context["root_candidates"],
                        operator_candidates=context["operator_candidates"],
                        legal_constraints=context["legal_constraints"],
                    ) if not batch.failure_reason else ((), ())
                    dispositions = tuple(rejected)
                    persisted_keys: tuple[str, ...] = ()
                    legal_count = len(valid)
                else:
                    result = pipeline.run(
                        instance.problem, instance.schedule,
                        state=context["state"], causal_chains=context["causal_chains"],
                        root_candidates=context["root_candidates"],
                        operator_candidates=context["operator_candidates"],
                        legal_constraints=context["legal_constraints"],
                    )
                    batch = result.generation
                    dispositions = result.dispositions
                    persisted_keys = result.persisted_keys
                    legal_count = sum(
                        item.status == "persisted" or item.reason == "counterfactual_validation_failed"
                        for item in dispositions
                    )
                self._record_batch(
                    manifest, failure_path, candidate_path, instance,
                    problem_hash, incumbent_hash, context, batch,
                    dispositions, persisted_keys, legal_count,
                )
            except Exception as error:
                self._failure(
                    manifest, failure_path, instance.instance_id,
                    "runtime", f"{type(error).__name__}:{error}",
                )
            completed.add(state_key)
            manifest["completed_state_keys"] = sorted(completed)
            self._checkpoint(manifest, manifest_path)
        manifest["status"] = "dry_run_complete" if config.dry_run else "production_complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._checkpoint(manifest, manifest_path)
        return self._report(manifest, manifest_path, failure_path, candidate_path)

    def _generation_context(self, instance, selected, output) -> dict[str, Any]:
        appearance_id = _appearance_id(selected)
        appearance_kind = _appearance_kind(selected)
        proposals = tuple(
            proposal for proposal in getattr(output, "proposals", ())
            if str(getattr(proposal, "appearance_id", "")) == appearance_id
        )
        chains = tuple(
            chain for chain in getattr(output, "causal_explanation_chains", ())
            if str(getattr(chain, "appearance_id", "")) == appearance_id
        )
        roots_by_id = {}
        for proposal in proposals:
            for site in getattr(proposal, "root_decisions", ()):
                roots_by_id[str(site.site_id)] = site
        if not roots_by_id:
            actionable = set(getattr(output, "actionable_root_ids", ()))
            roots_by_id = {
                str(site.site_id): site for site in getattr(output, "decision_sites", ())
                if site.site_id in actionable
            }
        relevant_ops = {
            edit.operation_id for proposal in proposals for edit in getattr(proposal, "edits", ())
        }
        relevant_ops.update(
            str(node) for chain in chains for node in getattr(chain, "nodes", ())
            if not str(node).startswith("machine:")
        )
        legal = tuple(
            edit for edit in getattr(output, "legal_edits", ())
            if edit.operation_id in relevant_ops
        )
        if not legal:
            legal = tuple(edit for proposal in proposals for edit in proposal.edits)
        base = encode_state(
            instance.problem, instance.schedule,
            appearance_type=appearance_kind,
            appearance_score=float(selected.get("priority", selected.get("score", 0.0)) or 0.0),
        )
        block = selected.get("block", selected)
        machines = tuple(block.get("machines", ())) if isinstance(block, Mapping) else ()
        state = replace(base, overloaded_machine=(str(machines[0]) if machines else None))
        return {
            "appearance_id": appearance_id,
            "appearance_kind": appearance_kind,
            "state": state,
            "causal_chains": chains,
            "root_candidates": tuple(roots_by_id.values()),
            "operator_candidates": proposals,
            "legal_constraints": tuple(dict.fromkeys(legal)),
        }

    def _initial_manifest(self, config: HarvestGenerationConfig) -> dict[str, Any]:
        return {
            "schema": "deepseek_trajectory_harvesting_manifest_v1",
            "run_id": config.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": config.provider,
            "model": self.generator.adapter.provider.model,
            "prompt_version": PROMPT_VERSION,
            "dry_run": config.dry_run,
            "formal_training": False,
            "optimizer_steps": 0,
            "test_access": 0,
            "identified": False,
            "config": asdict(config),
            "completed_state_keys": [],
            "m2_training_statuses": {},
            "statistics": {
                "instance_count": 0, "api_call_count": 0, "api_attempt_count": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "candidate_count": 0, "legal_count": 0, "executed_count": 0,
                "memory_write_count": 0, "discard_count": 0,
                "success_count": 0, "failure_count": 0, "partial_success_count": 0,
                "operator_distribution": {}, "appearance_distribution": {},
            },
            "failure_count": 0,
            "status": "running",
        }

    def _record_batch(
        self, manifest, failure_path, candidate_path, instance,
        problem_hash, incumbent_hash, context, batch, dispositions,
        persisted_keys, legal_count,
    ) -> None:
        stats = manifest["statistics"]
        stats["api_call_count"] += 1
        stats["api_attempt_count"] += int(batch.attempts)
        stats["input_tokens"] += int(batch.input_tokens)
        stats["output_tokens"] += int(batch.output_tokens)
        stats["total_tokens"] += int(batch.total_tokens)
        stats["candidate_count"] += len(batch.candidates)
        stats["legal_count"] += int(legal_count)
        stats["executed_count"] += sum(bool(item.solver_status) for item in dispositions)
        stats["memory_write_count"] += len(persisted_keys)
        stats["discard_count"] += sum(item.status == "discarded" for item in dispositions)
        if batch.failure_reason:
            self._failure(
                manifest, failure_path, instance.instance_id,
                "json_parse" if batch.failure_reason.startswith("invalid_json") else "api",
                batch.failure_reason,
            )
        by_index = {item.index: item for item in dispositions}
        for index, candidate in enumerate(batch.candidates):
            disposition = by_index.get(index)
            candidate_id = _canonical_hash({
                "instance_id": instance.instance_id,
                "prompt_hash": batch.prompt_hash,
                "candidate": candidate.model_dump(mode="json"),
            })[:24]
            row = {
                "candidate_id": candidate_id,
                "instance_id": instance.instance_id,
                "problem_hash": problem_hash,
                "schedule_hash": incumbent_hash,
                "appearance": context["appearance_kind"],
                "appearance_id": context["appearance_id"],
                "prompt_hash": batch.prompt_hash,
                "prompt_version": batch.prompt_version,
                "candidate": candidate.model_dump(mode="json"),
                "status": disposition.status if disposition else "legal_dry_run",
                "reason": disposition.reason if disposition else "schema_and_legal_validated",
                "memory_key": disposition.memory_key if disposition else "",
                "solver_status": disposition.solver_status if disposition else "",
                "delta_cmax": disposition.delta_cmax if disposition else None,
            }
            _append_jsonl(candidate_path, row)
            if disposition and disposition.status == "discarded":
                stage = "solver" if disposition.solver_status else "legal_validation"
                self._failure(
                    manifest, failure_path, instance.instance_id, stage,
                    disposition.reason, candidate_id=candidate_id,
                )
        operators = Counter()
        appearances = Counter()
        for key in persisted_keys:
            record = self.store.get(key)
            if record is None or record.outcome is None:
                continue
            operators[record.proposal.operator_type or "unknown"] += 1
            appearances[record.proposal.appearance_type or "unknown"] += 1
            if record.outcome.classification == DIRECT_SUCCESS:
                stats["success_count"] += 1
            elif record.outcome.classification == DELAYED_SUCCESS:
                stats["partial_success_count"] += 1
            elif record.outcome.classification == FAILURE:
                stats["failure_count"] += 1
        for key, value in operators.items():
            stats["operator_distribution"][key] = stats["operator_distribution"].get(key, 0) + value
        for key, value in appearances.items():
            stats["appearance_distribution"][key] = stats["appearance_distribution"].get(key, 0) + value

    @staticmethod
    def _failure(manifest, path, instance_id, stage, reason, *, candidate_id="") -> None:
        manifest["failure_count"] = int(manifest.get("failure_count", 0)) + 1
        _append_jsonl(path, {
            "candidate_id": candidate_id,
            "instance_id": instance_id,
            "stage": stage,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @staticmethod
    def _checkpoint(manifest, path) -> None:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(path, manifest)

    def _report(self, manifest, manifest_path, failure_path, candidate_path) -> HarvestReport:
        return HarvestReport(
            run_id=str(manifest["run_id"]),
            manifest_path=str(manifest_path), failure_path=str(failure_path),
            candidate_path=str(candidate_path),
            memory_path=(str(self.store.path) if self.store.path else None),
            dry_run=bool(manifest["dry_run"]),
            statistics=dict(manifest["statistics"]),
            status=str(manifest["status"]),
        )


__all__ = [
    "HarvestGenerationConfig", "HarvestInstance", "HarvestReport",
    "TrajectoryHarvesterV1", "load_harvest_generation_config",
]
