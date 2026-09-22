"""Adapter from experience generation to the project's existing LLM provider.

This module intentionally contains no HTTP client, SDK import, API-key parsing,
or provider-specific completion logic.  It reuses ``ModelProvider`` and the
existing ``.env`` configuration loader.  DeepSeek is the default provider
prefix, not a separate integration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Literal, Mapping

from ..providers import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleProvider,
    load_provider_configuration,
)
from ..providers.anthropic_compatible import (
    AnthropicCompatibleProvider,
    AnthropicConfiguration,
)


@dataclass(frozen=True)
class ExperienceGenerationConfig:
    llm_provider: str = "deepseek"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    candidates_per_state: int = 10
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    thinking_mode: Literal["enabled", "disabled", "adaptive"] = "disabled"

    def validate(self) -> None:
        if not self.llm_provider.strip():
            raise ValueError("experience_generation.llm_provider must be non-empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("experience_generation.temperature must be in [0,2]")
        if self.max_tokens < 128:
            raise ValueError("experience_generation.max_tokens must be >= 128")
        if self.candidates_per_state < 1:
            raise ValueError("experience_generation.candidates_per_state must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("experience_generation.timeout_seconds must be > 0")
        if self.max_attempts < 1:
            raise ValueError("experience_generation.max_attempts must be >= 1")
        if self.thinking_mode not in {"enabled", "disabled", "adaptive"}:
            raise ValueError("invalid experience_generation.thinking_mode")


def _repository_root() -> Path:
    configured = os.environ.get("CAUSAL_SCHEDULE_LAB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "configs" / "hyperparameters.yaml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


def load_experience_generation_config(
    path: str | Path | None = None,
) -> ExperienceGenerationConfig:
    source = Path(
        path
        or os.environ.get("CAUSAL_SCHEDULE_LAB_HYPERPARAMETERS", "")
        or (_repository_root() / "configs" / "hyperparameters.yaml")
    ).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    section = payload.get("experience_generation")
    if not isinstance(section, Mapping):
        raise ValueError("missing experience_generation hyperparameters")
    config = ExperienceGenerationConfig(
        llm_provider=str(section.get("llm_provider", "deepseek")),
        model=str(section.get("model") or ""),
        temperature=float(section.get("temperature", 0.2)),
        max_tokens=int(section.get("max_tokens", 4096)),
        candidates_per_state=int(section.get("candidates_per_state", 10)),
        timeout_seconds=float(section.get("timeout_seconds", 120.0)),
        max_attempts=int(section.get("max_attempts", 3)),
        thinking_mode=str(section.get("thinking_mode", "disabled")),
    )
    config.validate()
    return config


class LLMProviderAdapter:
    """Thin provider-neutral completion adapter for experience generation."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        *,
        config: ExperienceGenerationConfig | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        self.config = config or load_experience_generation_config()
        self.config.validate()
        if provider is None:
            source = Path(env_file or (_repository_root() / ".env"))
            provider_config = load_provider_configuration(
                env_file=source,
                prefix=self.config.llm_provider.strip().upper(),
                timeout_seconds=self.config.timeout_seconds,
                max_attempts=self.config.max_attempts,
            )
            if self.config.model:
                provider_config = replace(provider_config, model=self.config.model)
            provider_name = self.config.llm_provider.strip().lower()
            # The existing DeepSeek relay may expose either OpenAI chat
            # completions or Anthropic messages. Select the matching project
            # provider instead of constructing a second API integration.
            if "/anthropic" in provider_config.base_url.rstrip("/").lower():
                provider = AnthropicCompatibleProvider(
                    AnthropicConfiguration(
                        api_key=provider_config.api_key,
                        base_url=provider_config.base_url,
                        model=provider_config.model,
                        timeout_seconds=provider_config.timeout_seconds,
                        max_attempts=provider_config.max_attempts,
                    ),
                    provider_name=provider_name,
                )
            else:
                provider = OpenAICompatibleProvider(
                    provider_config,
                    provider_name=provider_name,
                )
        self.provider = provider

    def generate_completion(
        self,
        prompt: str,
        model_config: ExperienceGenerationConfig | None = None,
    ) -> ModelResponse:
        """Generate strict JSON through the existing provider management layer."""
        config = model_config or self.config
        config.validate()
        return self.provider.complete(ModelRequest(
            system=(
                "You propose scheduling intervention candidates only. Return one "
                "strict JSON object matching the supplied schema. Never invent a "
                "reward, success label, solver result, or future gain."
            ),
            user=prompt,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            require_json=True,
            thinking_mode=config.thinking_mode,
            metadata={
                "task": "llm_assisted_trajectory_experience_generation_v1",
                "candidate_limit": config.candidates_per_state,
            },
        ))


__all__ = [
    "ExperienceGenerationConfig",
    "LLMProviderAdapter",
    "load_experience_generation_config",
]
