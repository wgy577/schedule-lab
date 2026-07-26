# TEST-TAGS: modules=B,H; capabilities=llm_trace,token_accounting,error_audit; level=unit; cost=low
from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_schedule_lab.providers.base import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from causal_schedule_lab.providers.tracing import LLMRunTrace


class FakeProvider:
    name = "fake-provider"
    model = "fake-model"

    def complete(self, request: ModelRequest) -> ModelResponse:
        if request.user == "fail":
            raise RuntimeError("controlled failure")
        return ModelResponse(
            content='{"ok":true}',
            provider=self.name,
            requested_model=self.model,
            response_model="fake-model-v1",
            usage=TokenUsage(
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
            ),
            raw_metadata={"total_cost_usd": 0.01},
        )


def _events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_trace_marks_model_stage_batch_round_and_completion(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    provider = LLMRunTrace(path).wrap(FakeProvider(), label="analyst")
    provider.complete(
        ModelRequest(
            system="system",
            user="ok",
            metadata={
                "task": "semantic_batch",
                "batch": "constraints",
                "round": 2,
                "repair": 1,
            },
        )
    )
    events = _events(path)
    assert [item["status"] for item in events] == ["started", "completed"]
    assert events[0]["call_id"] == "analyst-0001"
    assert events[0]["model"] == "fake-model"
    assert events[0]["batch"] == "constraints"
    assert events[0]["round"] == 2
    assert events[1]["total_tokens"] == 5
    assert "content_sha256" in events[1]
    assert "content" not in events[1]


def test_trace_records_provider_failure_before_reraising(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    provider = LLMRunTrace(path).wrap(FakeProvider(), label="navigator")
    with pytest.raises(RuntimeError, match="controlled failure"):
        provider.complete(ModelRequest(system="system", user="fail"))
    events = _events(path)
    assert events[-1]["status"] == "error"
    assert events[-1]["error_type"] == "RuntimeError"
    assert events[-1]["error_category"] == "unclassified"
    assert "进一步诊断" in str(events[-1]["likely_cause"])
    assert events[-1]["call_id"] == "navigator-0001"
