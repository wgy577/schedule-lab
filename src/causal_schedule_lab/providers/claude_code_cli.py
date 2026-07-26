"""Official Claude Code CLI adapter for channels that reject generic clients."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ModelRequest, ModelResponse, TokenUsage
from .openai_compatible import ProviderError, read_env_file


@dataclass(frozen=True)
class ClaudeCodeConfiguration:
    api_key: str
    base_url: str
    model: str
    executable: str = "claude"
    timeout_seconds: float = 600.0
    max_budget_usd: float = 8.0
    enable_project_skill: bool = False
    enable_read_tools: bool = False
    skill_name: str = "scheduling-code-semantics"
    read_budget: int = 18

    def __repr__(self) -> str:
        return (
            "ClaudeCodeConfiguration(api_key='***', "
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"executable={self.executable!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_budget_usd={self.max_budget_usd!r}, "
            f"enable_project_skill={self.enable_project_skill!r}, "
            f"enable_read_tools={self.enable_read_tools!r}, "
            f"skill_name={self.skill_name!r}, read_budget={self.read_budget!r})"
        )


def load_claude_code_configuration(
    *,
    env_file: str | Path,
    timeout_seconds: float = 600.0,
    max_budget_usd: float = 8.0,
    base_url_override: str | None = None,
    enable_project_skill: bool = False,
    enable_read_tools: bool = False,
    skill_name: str = "scheduling-code-semantics",
    read_budget: int = 18,
) -> ClaudeCodeConfiguration:
    values = read_env_file(env_file)

    def value(name: str) -> str:
        return os.environ.get(name) or values.get(name, "")

    api_key = value("CLAUDE_CODE_TOKEN") or value("TOKEN")
    base_url = (
        base_url_override
        or value("CLAUDE_CODE_BASE_URL")
        or value("BASE_URL")
    )
    model = value("CLAUDE_CODE_MODEL") or value("CLAUDE_MODEL")
    missing = [
        key
        for key, item in (
            ("TOKEN", api_key),
            ("BASE_URL", base_url),
            ("CLAUDE_MODEL", model),
        )
        if not item
    ]
    if missing:
        raise ProviderError(
            "missing Claude Code configuration: " + ", ".join(missing)
        )
    if not base_url.startswith("https://"):
        raise ProviderError("Claude Code base URL must use HTTPS")
    # The public selector page is Cloudflare-protected.  Its Claude channel
    # exposes an official-client node; callers can always override this through
    # CLAUDE_CODE_BASE_URL or the CLI flag.
    if base_url.rstrip("/") == "https://api.sssaicode.com":
        base_url = "https://node-cf.sssaicodeapi.com/api"
    return ClaudeCodeConfiguration(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_seconds=timeout_seconds,
        max_budget_usd=max_budget_usd,
        enable_project_skill=enable_project_skill,
        enable_read_tools=enable_read_tools,
        skill_name=skill_name,
        read_budget=read_budget,
    )


def _token_usage(payload: dict[str, Any]) -> TokenUsage:
    direct = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = int(direct.get("input_tokens") or 0)
    output_tokens = int(direct.get("output_tokens") or 0)
    if not input_tokens and not output_tokens:
        model_usage = payload.get("modelUsage")
        if isinstance(model_usage, dict):
            for item in model_usage.values():
                if not isinstance(item, dict):
                    continue
                input_tokens += int(item.get("inputTokens") or 0)
                output_tokens += int(item.get("outputTokens") or 0)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


class ClaudeCodeCLIProvider:
    """Call Opus through the installed official Claude Code executable."""

    def __init__(
        self,
        configuration: ClaudeCodeConfiguration,
        *,
        provider_name: str = "claude-code",
    ) -> None:
        self.configuration = configuration
        self._name = provider_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self.configuration.model

    def complete(self, request: ModelRequest) -> ModelResponse:
        # This relay authenticates the official client partly through Claude
        # Code's native system envelope.  Replacing it via --system-prompt makes
        # a genuine CLI invocation look like a generic client, so the task
        # contract is carried inside the user message instead.
        skill_instruction = ""
        if self.configuration.enable_project_skill:
            read_instruction = (
                "可以使用只读工具定位代码。"
                if self.configuration.enable_read_tools
                else
                "本次代码复读由外层 Harness ReadGate 统一执行；"
                "不得自行调用文件工具。证据不足时只能按输出 schema 提交 read_requests。"
            )
            skill_instruction = (
                f"/{self.configuration.skill_name}\n\n"
                f"必须使用项目 Skill /{self.configuration.skill_name}。"
                + read_instruction
                + f"本次最多进行 {self.configuration.read_budget} 次文件读取或检索。"
                "最终仍必须服从下面的任务契约与输出 schema。\n\n"
            )
        instruction = (
            skill_instruction
            +
            "以下为不可忽略的任务契约：\n"
            + request.system
            + "\n\n以下为本次输入：\n"
            + request.user
        )
        if request.require_json:
            instruction += (
                "\n\n只返回一个 JSON 对象，不要 Markdown 代码围栏，不要解释。"
            )
        output_format = (
            "stream-json" if self.configuration.enable_read_tools else "json"
        )
        command = [
            self.configuration.executable,
            "-p",
            instruction,
            "--output-format",
            output_format,
            "--model",
            self.model,
            "--no-session-persistence",
            "--max-budget-usd",
            str(self.configuration.max_budget_usd),
        ]
        if request.reasoning_effort is not None:
            command.extend(["--effort", request.reasoning_effort])
        if self.configuration.enable_read_tools:
            command.extend(
                [
                    "--verbose",
                    "--tools",
                    "Read,Glob,Grep",
                    "--allowedTools",
                    "Read,Glob,Grep",
                    "--permission-mode",
                    "dontAsk",
                    "--setting-sources",
                    "project",
                ]
            )
        else:
            command.extend(["--tools", "", "--safe-mode"])
            if self.configuration.enable_project_skill:
                command.extend(["--setting-sources", "project"])
        environment = os.environ.copy()
        environment.update(
            {
                "ANTHROPIC_BASE_URL": self.configuration.base_url,
                "ANTHROPIC_AUTH_TOKEN": self.configuration.api_key,
                "ANTHROPIC_MODEL": self.model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        )
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                input="",
                text=True,
                capture_output=True,
                env=environment,
                timeout=self.configuration.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise ProviderError(
                f"Claude Code executable not found: {self.configuration.executable}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise ProviderError(
                "Claude Code request timed out after "
                f"{self.configuration.timeout_seconds:.0f}s"
            ) from error
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "no output")[:1200]
            raise ProviderError(
                f"Claude Code exited with status {process.returncode}: {detail}"
            )
        tools_used: list[str] = []
        if self.configuration.enable_read_tools:
            payload: dict[str, Any] | None = None
            for line in process.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = event.get("message")
                blocks = (
                    message.get("content", [])
                    if isinstance(message, dict)
                    else []
                )
                for block in blocks if isinstance(blocks, list) else []:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and isinstance(block.get("name"), str)
                    ):
                        tools_used.append(block["name"])
                if event.get("type") == "result":
                    payload = event
            if payload is None:
                raise ProviderError(
                    "Claude Code stream contains no final result event"
                )
        else:
            try:
                payload = json.loads(process.stdout)
            except json.JSONDecodeError as error:
                raise ProviderError(
                    "Claude Code returned invalid transport JSON: "
                    f"{process.stdout[:800]}"
                ) from error
        if payload.get("is_error") or payload.get("subtype") not in {None, "success"}:
            raise ProviderError(
                "Claude Code request failed: "
                f"{str(payload.get('result') or payload)[:1200]}"
            )
        content = payload.get("structured_output") or payload.get("result")
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Claude Code response contains no result text")
        return ModelResponse(
            content=content,
            provider=self.name,
            requested_model=self.model,
            response_model=self.model,
            request_id=payload.get("session_id"),
            finish_reason=payload.get("subtype"),
            usage=_token_usage(payload),
            latency_seconds=time.perf_counter() - started,
            attempts=1,
            raw_metadata={
                "duration_ms": payload.get("duration_ms"),
                "duration_api_ms": payload.get("duration_api_ms"),
                "num_turns": payload.get("num_turns"),
                "total_cost_usd": payload.get("total_cost_usd"),
                "official_client": True,
                "project_skill": (
                    self.configuration.skill_name
                    if self.configuration.enable_project_skill
                    else None
                ),
                "read_tools": self.configuration.enable_project_skill,
                "tools_used": tuple(tools_used),
                "tool_call_count": len(tools_used),
            },
        )
