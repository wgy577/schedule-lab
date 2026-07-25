"""Minimal OpenAI-compatible provider with safe retries and JSON support."""

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


class ProviderError(RuntimeError):
    """A redacted provider failure safe to show in CLI output."""


@dataclass(frozen=True)
class ProviderConfiguration:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    max_attempts: int = 3

    def __repr__(self) -> str:
        return (
            "ProviderConfiguration(api_key='***', "
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_attempts={self.max_attempts!r})"
        )


def read_env_file(path: str | Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without executing shell expressions."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        return {}
    values: dict[str, str] = {}
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def load_provider_configuration(
    *,
    env_file: str | Path,
    prefix: str = "SEED",
    timeout_seconds: float = 120.0,
    max_attempts: int = 3,
) -> ProviderConfiguration:
    file_values = read_env_file(env_file)

    def value(name: str) -> str:
        return os.environ.get(name) or file_values.get(name, "")

    api_key = value(f"{prefix}_API") or value(f"{prefix}_API_KEY")
    base_url = value(f"{prefix}_BASE_URL_OPENAI") or value(
        f"{prefix}_BASE_URL"
    )
    model = value(f"{prefix}_MODEL")
    missing = [
        name
        for name, item in (
            (f"{prefix}_API", api_key),
            (f"{prefix}_BASE_URL_OPENAI", base_url),
            (f"{prefix}_MODEL", model),
        )
        if not item
    ]
    if missing:
        raise ProviderError(
            "missing provider configuration: " + ", ".join(missing)
        )
    if not base_url.startswith("https://"):
        raise ProviderError("provider base URL must use HTTPS")
    return ProviderConfiguration(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )


Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]


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
        payload = error.read()
        return error.code, dict(error.headers.items()), payload


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return ""


class OpenAICompatibleProvider:
    def __init__(
        self,
        configuration: ProviderConfiguration,
        *,
        provider_name: str = "volcengine-coding-plan",
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
        return (
            base
            if base.endswith("/chat/completions")
            else base + "/chat/completions"
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        model_name = self.model.lower()
        thinking_model = (
            request.thinking_mode in {"enabled", "adaptive"}
            or any(
                marker in model_name
                for marker in ("deepseek-v4", "kimi-k3", "mimo-v2.5")
            )
        )
        effective_effort = request.reasoning_effort
        if (
            "mimo-v2.5" in model_name
            and effective_effort in {"xhigh", "max"}
        ):
            effective_effort = "high"
        effective_temperature = 1.0 if thinking_model else request.temperature
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
        }
        if "mimo-v2.5" in model_name:
            payload["max_completion_tokens"] = request.max_output_tokens
        else:
            payload["max_tokens"] = request.max_output_tokens
        if not thinking_model:
            payload["temperature"] = effective_temperature
        if effective_effort is not None:
            payload["reasoning_effort"] = effective_effort
        if request.thinking_mode is not None and "kimi-k3" not in model_name:
            payload["thinking"] = {"type": request.thinking_mode}
        if request.require_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.configuration.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "causal-schedule-lab/0.5.0",
        }
        last_error = "unknown provider error"
        started = time.perf_counter()
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
                    choice = data["choices"][0]
                    message = choice["message"]
                    content = _message_content(message.get("content"))
                    reasoning_content = _message_content(
                        message.get("reasoning_content")
                    )
                    if not reasoning_content and isinstance(
                        message.get("reasoning_content"), str
                    ):
                        reasoning_content = message["reasoning_content"]
                    if not content:
                        raise ValueError(
                            "response message content is empty; "
                            f"finish_reason={choice.get('finish_reason')}; "
                            "reasoning_characters="
                            f"{len(_message_content(message.get('reasoning_content')))}"
                        )
                    usage = data.get("usage") or {}
                    return ModelResponse(
                        content=content,
                        provider=self.name,
                        requested_model=self.model,
                        response_model=data.get("model"),
                        request_id=(
                            data.get("id")
                            or response_headers.get("x-request-id")
                            or response_headers.get("X-Request-Id")
                        ),
                        finish_reason=choice.get("finish_reason"),
                        usage=TokenUsage(
                            input_tokens=int(
                                usage.get("prompt_tokens")
                                or usage.get("input_tokens")
                                or 0
                            ),
                            output_tokens=int(
                                usage.get("completion_tokens")
                                or usage.get("output_tokens")
                                or 0
                            ),
                            total_tokens=int(usage.get("total_tokens") or 0),
                        ),
                        latency_seconds=time.perf_counter() - started,
                        attempts=attempt,
                        raw_metadata={
                            "system_fingerprint": data.get(
                                "system_fingerprint"
                            ),
                            "requested_temperature": request.temperature,
                            "effective_temperature": (
                                None if thinking_model else effective_temperature
                            ),
                            "thinking_mode": (
                                "always_on"
                                if "kimi-k3" in model_name
                                else request.thinking_mode
                            ),
                            "requested_reasoning_effort": request.reasoning_effort,
                            "reasoning_effort": effective_effort,
                            "reasoning_characters": len(reasoning_content),
                            "thinking_tokens": (
                                usage.get("completion_tokens_details", {})
                                .get("reasoning_tokens")
                                or usage.get("output_tokens_details", {})
                                .get("thinking_tokens")
                                or 0
                            ),
                        },
                    )
                except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
                    last_error = f"invalid provider response: {error}"
            else:
                try:
                    error_payload = json.loads(text)
                    detail = error_payload.get("error", error_payload)
                    last_error = json.dumps(detail, ensure_ascii=False)[:1000]
                except json.JSONDecodeError:
                    last_error = text[:1000] or f"HTTP {status}"
                # Some compatible gateways do not support response_format.
                if (
                    status == 400
                    and "response_format" in payload
                    and "response_format" in last_error
                ):
                    payload.pop("response_format", None)
                    continue
                if status not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            if attempt < self.configuration.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
        raise ProviderError(
            f"{self.name} request failed after "
            f"{self.configuration.max_attempts} attempt(s): {last_error}"
        )
