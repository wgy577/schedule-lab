"""Anthropic Messages-compatible provider for Claude relay endpoints."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .base import ModelRequest, ModelResponse, TokenUsage
from .openai_compatible import ProviderError, read_env_file


@dataclass(frozen=True)
class AnthropicConfiguration:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    max_attempts: int = 3

    def __repr__(self) -> str:
        return (
            "AnthropicConfiguration(api_key='***', "
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_attempts={self.max_attempts!r})"
        )


def load_anthropic_configuration(
    *,
    env_file: str | Path,
    timeout_seconds: float = 120.0,
    max_attempts: int = 3,
) -> AnthropicConfiguration:
    file_values = read_env_file(env_file)

    def value(name: str) -> str:
        return os.environ.get(name) or file_values.get(name, "")

    api_key = value("ANTHROPIC_AUTH_TOKEN") or value("ANTHROPIC_API_KEY")
    base_url = value("ANTHROPIC_BASE_URL")
    model = value("CLAUDE_MODEL") or value("ANTHROPIC_MODEL")
    missing = [
        name
        for name, item in (
            ("ANTHROPIC_AUTH_TOKEN", api_key),
            ("ANTHROPIC_BASE_URL", base_url),
            ("CLAUDE_MODEL", model),
        )
        if not item
    ]
    if missing:
        raise ProviderError(
            "missing Anthropic provider configuration: " + ", ".join(missing)
        )
    if not base_url.startswith("https://"):
        raise ProviderError("Anthropic base URL must use HTTPS")
    return AnthropicConfiguration(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )


Transport = Callable[
    [str, dict[str, str], bytes, float],
    tuple[int, dict[str, str], bytes],
]


def _default_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                dict(response.headers.items()),
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _thinking_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        str(item.get("thinking", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "thinking"
    )


class AnthropicCompatibleProvider:
    def __init__(
        self,
        configuration: AnthropicConfiguration,
        *,
        provider_name: str = "anthropic-compatible",
        transport: Transport | None = None,
    ) -> None:
        self.configuration = configuration
        self._name = provider_name
        self.transport = transport or _default_transport

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self.configuration.model

    def _endpoint(self) -> str:
        base = self.configuration.base_url
        return base if base.endswith("/v1/messages") else base + "/v1/messages"

    def complete(self, request: ModelRequest) -> ModelResponse:
        json_instruction = (
            "\n\nReturn exactly one JSON object without Markdown fences."
            if request.require_json
            else ""
        )
        payload = {
            "model": self.model,
            "system": request.system + json_instruction,
            "messages": [{"role": "user", "content": request.user}],
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.thinking_mode is not None:
            if self.model.lower().startswith("claude-opus-4-8"):
                payload["thinking"] = {"type": "adaptive"}
            else:
                payload["thinking"] = {"type": request.thinking_mode}
        else:
            payload["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            payload["output_config"] = {
                "effort": request.reasoning_effort,
            }
        headers = {
            "x-api-key": self.configuration.api_key,
            "Authorization": f"Bearer {self.configuration.api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "causal-schedule-lab/0.5.0",
        }
        started = time.perf_counter()
        last_error = "unknown provider error"
        for attempt in range(1, self.configuration.max_attempts + 1):
            try:
                status, response_headers, body = self.transport(
                    self._endpoint(),
                    headers,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    self.configuration.timeout_seconds,
                )
            except (TimeoutError, OSError, urllib.error.URLError) as error:
                last_error = (
                    f"transport error: {type(error).__name__}: {str(error)[:500]}"
                )
                if attempt < self.configuration.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                break
            text = body.decode("utf-8", errors="replace")
            if 200 <= status < 300:
                try:
                    data = json.loads(text)
                    content = _content_text(data.get("content"))
                    thinking_content = _thinking_text(data.get("content"))
                    if not content:
                        raise ValueError("response contains no text block")
                    usage = data.get("usage") or {}
                    input_tokens = int(usage.get("input_tokens") or 0)
                    output_tokens = int(usage.get("output_tokens") or 0)
                    return ModelResponse(
                        content=content,
                        provider=self.name,
                        requested_model=self.model,
                        response_model=data.get("model"),
                        request_id=(
                            data.get("id")
                            or response_headers.get("request-id")
                            or response_headers.get("x-request-id")
                        ),
                        finish_reason=data.get("stop_reason"),
                        usage=TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                        ),
                        latency_seconds=time.perf_counter() - started,
                        attempts=attempt,
                        raw_metadata={
                            "thinking_mode": (
                                payload.get("thinking", {}).get("type")
                            ),
                            "reasoning_effort": request.reasoning_effort,
                            "reasoning_characters": len(thinking_content),
                            "thinking_tokens": (
                                usage.get("output_tokens_details", {})
                                .get("thinking_tokens")
                                or 0
                            ),
                        },
                    )
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as error:
                    last_error = f"invalid provider response: {error}"
            else:
                try:
                    error_payload = json.loads(text)
                    last_error = json.dumps(
                        error_payload.get("error", error_payload),
                        ensure_ascii=False,
                    )[:1000]
                except json.JSONDecodeError:
                    last_error = text[:1000] or f"HTTP {status}"
                if status not in {408, 409, 429, 500, 502, 503, 504, 529}:
                    break
            if attempt < self.configuration.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
        raise ProviderError(
            f"{self.name} request failed after "
            f"{self.configuration.max_attempts} attempt(s): {last_error}"
        )
