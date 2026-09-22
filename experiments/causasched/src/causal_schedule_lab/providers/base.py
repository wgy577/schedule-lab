"""Provider-neutral LLM request and response contracts."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class TokenUsage(FrozenModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelRequest(FrozenModel):
    system: str
    user: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, ge=128)
    require_json: bool = True
    thinking_mode: Literal["enabled", "disabled", "adaptive"] | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(FrozenModel):
    content: str
    provider: str
    requested_model: str
    response_model: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=1)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...
