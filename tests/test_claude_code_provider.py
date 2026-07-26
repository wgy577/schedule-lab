# TEST-TAGS: modules=B; capabilities=claude_code_provider,skill_routing; level=unit; cost=low
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from causal_schedule_lab.providers.base import ModelRequest
from causal_schedule_lab.providers.claude_code_cli import (
    ClaudeCodeCLIProvider,
    ClaudeCodeConfiguration,
    load_claude_code_configuration,
)


def test_claude_code_loader_uses_dedicated_profile_and_official_node(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TOKEN=test-token",
                "BASE_URL=https://api.sssaicode.com",
                "CLAUDE_MODEL=claude-opus-5",
            ]
        ),
        encoding="utf-8",
    )
    configuration = load_claude_code_configuration(env_file=env_file)
    assert configuration.api_key == "test-token"
    assert configuration.model == "claude-opus-5"
    assert configuration.base_url == "https://node-cf.sssaicodeapi.com/api"
    assert "test-token" not in repr(configuration)


def test_skill_mode_records_read_only_tools(monkeypatch) -> None:
    stdout = "\n".join(
        [
            '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Grep","input":{"pattern":"step"}}]}}',
            '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"env.py"}}]}}',
            '{"type":"result","subtype":"success","is_error":false,"result":"{\\"ok\\":true}","session_id":"s1","total_cost_usd":0.02,"usage":{"input_tokens":10,"output_tokens":4}}',
        ]
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    provider = ClaudeCodeCLIProvider(
        ClaudeCodeConfiguration(
            api_key="token",
            base_url="https://example.test/api",
                model="claude-opus-5",
                enable_project_skill=True,
                enable_read_tools=True,
        )
    )
    response = provider.complete(
        ModelRequest(system="contract", user="inspect", reasoning_effort="medium")
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert "stream-json" in command
    assert "Read,Glob,Grep" in command
    assert command[command.index("--effort") + 1] == "medium"
    assert "/scheduling-code-semantics" in command[2]
    assert response.raw_metadata["tools_used"] == ("Grep", "Read")
    assert response.raw_metadata["tool_call_count"] == 2
