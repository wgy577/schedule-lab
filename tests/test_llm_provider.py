import json

from causal_schedule_lab.providers.base import ModelRequest
from causal_schedule_lab.providers.anthropic_compatible import (
    AnthropicCompatibleProvider,
    AnthropicConfiguration,
    load_anthropic_configuration,
)
from causal_schedule_lab.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderConfiguration,
    load_provider_configuration,
)


def test_configuration_loads_seed_values_without_exposing_key(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            (
                "SEED_API=top-secret",
                "SEED_BASE_URL_OPENAI=https://example.invalid/v3",
                "SEED_MODEL=ark-code-latest",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    configuration = load_provider_configuration(env_file=env)
    assert configuration.api_key == "top-secret"
    assert configuration.model == "ark-code-latest"
    assert "top-secret" not in repr(configuration)


def test_openai_compatible_provider_parses_json_response() -> None:
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "payload": json.loads(body),
                "timeout": timeout,
            }
        )
        response = {
            "id": "request-1",
            "model": "actual-model",
            "choices": [
                {
                    "message": {"content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
        return 200, {}, json.dumps(response).encode()

    provider = OpenAICompatibleProvider(
        ProviderConfiguration(
            api_key="secret",
            base_url="https://example.invalid/v3",
            model="requested-model",
        ),
        transport=transport,
    )
    result = provider.complete(
        ModelRequest(system="system", user="user", require_json=True)
    )
    assert captured["url"].endswith("/v3/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert result.content == '{"answer":"ok"}'
    assert result.response_model == "actual-model"
    assert result.usage.total_tokens == 14


def test_kimi_k3_uses_max_reasoning_without_thinking_or_temperature() -> None:
    captured = {}

    def transport(url, headers, body, timeout):
        captured["payload"] = json.loads(body)
        response = {
            "model": "kimi-k3",
            "choices": [
                {
                    "message": {"content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        return 200, {}, json.dumps(response).encode()

    provider = OpenAICompatibleProvider(
        ProviderConfiguration(
            api_key="secret",
            base_url="https://example.invalid/v1",
            model="kimi-k3",
        ),
        transport=transport,
    )
    result = provider.complete(
        ModelRequest(
            system="system",
            user="user",
            temperature=0,
            require_json=True,
            thinking_mode="enabled",
            reasoning_effort="max",
        )
    )
    assert "temperature" not in captured["payload"]
    assert "thinking" not in captured["payload"]
    assert captured["payload"]["reasoning_effort"] == "max"
    assert result.raw_metadata["requested_temperature"] == 0.0
    assert result.raw_metadata["effective_temperature"] is None


def test_openai_provider_retries_transport_timeout() -> None:
    calls = 0

    def transport(url, headers, body, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("read timed out")
        response = {
            "model": "actual",
            "choices": [
                {
                    "message": {"content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        return 200, {}, json.dumps(response).encode()

    provider = OpenAICompatibleProvider(
        ProviderConfiguration(
            api_key="secret",
            base_url="https://example.invalid/v1",
            model="requested",
            max_attempts=2,
        ),
        transport=transport,
    )
    result = provider.complete(
        ModelRequest(system="system", user="user", require_json=True)
    )
    assert calls == 2
    assert result.attempts == 2


def test_mimo_pro_enables_deep_thinking_with_completion_budget() -> None:
    captured = {}

    def transport(url, headers, body, timeout):
        captured["payload"] = json.loads(body)
        response = {
            "model": "mimo-v2.5-pro",
            "choices": [
                {
                    "message": {
                        "content": '{"answer":"ok"}',
                        "reasoning_content": "careful reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        return 200, {}, json.dumps(response).encode()

    provider = OpenAICompatibleProvider(
        ProviderConfiguration(
            api_key="secret",
            base_url="https://example.invalid/v1",
            model="mimo-v2.5-pro",
        ),
        transport=transport,
    )
    result = provider.complete(
        ModelRequest(
            system="system",
            user="user",
            max_output_tokens=24000,
            thinking_mode="enabled",
            reasoning_effort="max",
        )
    )
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    assert captured["payload"]["max_completion_tokens"] == 24000
    assert captured["payload"]["reasoning_effort"] == "high"
    assert "temperature" not in captured["payload"]
    assert result.raw_metadata["requested_reasoning_effort"] == "max"
    assert result.raw_metadata["reasoning_effort"] == "high"
    assert result.raw_metadata["reasoning_characters"] == len(
        "careful reasoning"
    )


def test_anthropic_configuration_and_response(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            (
                "ANTHROPIC_AUTH_TOKEN=top-secret",
                "ANTHROPIC_BASE_URL=https://relay.example",
                "CLAUDE_MODEL=claude-opus-test",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    configuration = load_anthropic_configuration(env_file=env)
    assert configuration.model == "claude-opus-test"
    assert "top-secret" not in repr(configuration)
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "payload": json.loads(body),
            }
        )
        response = {
            "id": "msg-1",
            "model": "claude-opus-actual",
            "content": [{"type": "text", "text": '{"answer":"ok"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 5},
        }
        return 200, {}, json.dumps(response).encode()

    provider = AnthropicCompatibleProvider(
        AnthropicConfiguration(
            api_key="secret",
            base_url="https://relay.example",
            model="claude-opus-test",
        ),
        transport=transport,
    )
    result = provider.complete(
        ModelRequest(system="system", user="user", require_json=True)
    )
    assert captured["url"] == "https://relay.example/v1/messages"
    assert captured["headers"]["x-api-key"] == "secret"
    assert "Return exactly one JSON object" in captured["payload"]["system"]
    assert result.response_model == "claude-opus-actual"
    assert result.usage.total_tokens == 17


def test_opus_48_uses_adaptive_thinking_at_max_effort() -> None:
    captured = {}

    def transport(url, headers, body, timeout):
        captured["payload"] = json.loads(body)
        response = {
            "model": "claude-opus-4-8",
            "content": [
                {"type": "thinking", "thinking": "deep reasoning"},
                {"type": "text", "text": '{"answer":"ok"}'},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        return 200, {}, json.dumps(response).encode()

    provider = AnthropicCompatibleProvider(
        AnthropicConfiguration(
            api_key="secret",
            base_url="https://relay.example",
            model="claude-opus-4-8",
        ),
        transport=transport,
    )
    result = provider.complete(
        ModelRequest(
            system="system",
            user="user",
            thinking_mode="enabled",
            reasoning_effort="max",
        )
    )
    assert captured["payload"]["thinking"] == {"type": "adaptive"}
    assert captured["payload"]["output_config"] == {"effort": "max"}
    assert "temperature" not in captured["payload"]
    assert result.raw_metadata["reasoning_characters"] == len("deep reasoning")
