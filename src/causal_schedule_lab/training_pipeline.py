"""Resumable stage 0–8 training pipeline specified by the framework."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable


class TrainingStage(IntEnum):
    SEMANTIC_COMPILATION = 0
    COUNTERFACTUAL_COLLECTION = 1
    CIP_MULTITASK_PRETRAINING = 2
    CAUSAL_PATH_AND_CLOSURE_TRAINING = 3
    CONDITIONAL_GENERATOR_TRAINING = 4
    AGENT_BEHAVIOR_CLONING = 5
    OFFLINE_POLICY_TRAINING = 6
    CONTROLLED_ONLINE_PPO = 7
    LLM_REFLECTION_AND_DISTILLATION = 8


@dataclass(frozen=True)
class StageResult:
    stage: TrainingStage
    status: str
    started_at: float
    completed_at: float
    artifacts: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""


StageRunner = Callable[[Path], StageResult]


DEPENDENCIES: dict[TrainingStage, tuple[TrainingStage, ...]] = {
    TrainingStage.SEMANTIC_COMPILATION: (),
    TrainingStage.COUNTERFACTUAL_COLLECTION: (
        TrainingStage.SEMANTIC_COMPILATION,
    ),
    TrainingStage.CIP_MULTITASK_PRETRAINING: (
        TrainingStage.COUNTERFACTUAL_COLLECTION,
    ),
    TrainingStage.CAUSAL_PATH_AND_CLOSURE_TRAINING: (
        TrainingStage.COUNTERFACTUAL_COLLECTION,
    ),
    TrainingStage.CONDITIONAL_GENERATOR_TRAINING: (
        TrainingStage.COUNTERFACTUAL_COLLECTION,
    ),
    TrainingStage.AGENT_BEHAVIOR_CLONING: (
        TrainingStage.CIP_MULTITASK_PRETRAINING,
        TrainingStage.CAUSAL_PATH_AND_CLOSURE_TRAINING,
    ),
    TrainingStage.OFFLINE_POLICY_TRAINING: (
        TrainingStage.AGENT_BEHAVIOR_CLONING,
    ),
    TrainingStage.CONTROLLED_ONLINE_PPO: (
        TrainingStage.OFFLINE_POLICY_TRAINING,
        TrainingStage.CONDITIONAL_GENERATOR_TRAINING,
    ),
    TrainingStage.LLM_REFLECTION_AND_DISTILLATION: (
        TrainingStage.CONTROLLED_ONLINE_PPO,
    ),
}


class PipelineState:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "pipeline_state.json"
        self.results: dict[TrainingStage, StageResult] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for item in payload.get("stages", []):
                result = StageResult(
                    stage=TrainingStage(item["stage"]),
                    status=item["status"],
                    started_at=item["started_at"],
                    completed_at=item["completed_at"],
                    artifacts=tuple(item.get("artifacts", [])),
                    metrics=dict(item.get("metrics", {})),
                    message=item.get("message", ""),
                )
                self.results[result.stage] = result

    def completed(self, stage: TrainingStage) -> bool:
        return self.results.get(stage, None) is not None and (
            self.results[stage].status == "completed"
        )

    def save(self, result: StageResult) -> None:
        self.results[result.stage] = result
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "updated_at": time.time(),
                    "stages": [
                        {
                            "stage": int(item.stage),
                            "name": item.stage.name,
                            "status": item.status,
                            "started_at": item.started_at,
                            "completed_at": item.completed_at,
                            "artifacts": list(item.artifacts),
                            "metrics": item.metrics,
                            "message": item.message,
                        }
                        for item in sorted(
                            self.results.values(),
                            key=lambda value: value.stage,
                        )
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class TrainingPipeline:
    def __init__(self, state: PipelineState) -> None:
        self.state = state
        self.runners: dict[TrainingStage, StageRunner] = {}

    def register(self, stage: TrainingStage, runner: StageRunner) -> None:
        self.runners[stage] = runner

    def runnable(self, stage: TrainingStage) -> bool:
        return all(self.state.completed(item) for item in DEPENDENCIES[stage])

    def run(
        self,
        *,
        through: TrainingStage = TrainingStage.LLM_REFLECTION_AND_DISTILLATION,
        resume: bool = True,
    ) -> tuple[StageResult, ...]:
        results = []
        for stage in TrainingStage:
            if stage > through:
                break
            if resume and self.state.completed(stage):
                results.append(self.state.results[stage])
                continue
            if not self.runnable(stage):
                missing = [
                    item.name
                    for item in DEPENDENCIES[stage]
                    if not self.state.completed(item)
                ]
                raise RuntimeError(f"{stage.name} missing dependencies: {missing}")
            if stage not in self.runners:
                raise RuntimeError(
                    f"no runner registered for {stage.name}; "
                    "the pipeline will not pretend this training stage completed"
                )
            result = self.runners[stage](self.state.directory)
            if result.stage != stage:
                raise ValueError("stage runner returned a mismatched result")
            self.state.save(result)
            results.append(result)
            if result.status != "completed":
                break
        return tuple(results)
