"""Provider-neutral, content-free LLM call tracing."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import ModelProvider, ModelRequest, ModelResponse


def _diagnose_error(error: Exception) -> tuple[str, str]:
    message = str(error).lower()
    if "technical_completion_budget_failure" in message:
        return (
            "completion_budget",
            "reasoning 用尽 completion budget，未留下最终结构化输出预算",
        )
    if "schema" in message or "validation" in message or "batch_id" in message:
        return (
            "structured_output",
            "模型返回内容不符合固定 JSON schema 或不可变字段要求",
        )
    if "timed out" in message or "timeout" in message:
        return ("timeout", "模型或中转在当前超时预算内没有完成响应")
    if "forbidden" in message or "status 403" in message:
        return ("authentication_or_route", "凭据、账户组或中转路由拒绝访问")
    if "official" in message or "官方客户端" in message:
        return ("client_compatibility", "专用通道没有识别出受支持的官方客户端请求")
    if "not supported" in message and "model" in message:
        return ("model_route", "所选账户组或中转节点不支持请求的模型")
    if "transport" in message or "connection" in message:
        return ("transport", "本机到模型服务的网络传输失败")
    return ("unclassified", "需要结合该 call_id 的阶段和原始错误进一步诊断")


class LLMRunTrace:
    """Share a monotonic call sequence and JSONL event log across providers."""

    def __init__(self, path: str | Path, *, reset: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")
        self._counter = 0
        self._lock = threading.Lock()

    def _next_id(self, label: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{label}-{self._counter:04d}"

    def _write(self, event: dict[str, object]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def wrap(self, provider: ModelProvider, *, label: str) -> "TracingProvider":
        return TracingProvider(provider, trace=self, label=label)


class TracingProvider:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        trace: LLMRunTrace,
        label: str,
    ) -> None:
        self.provider = provider
        self.trace = trace
        self.label = label

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def complete(self, request: ModelRequest) -> ModelResponse:
        call_id = self.trace._next_id(self.label)
        metadata = request.metadata
        configuration = getattr(self.provider, "configuration", None)
        project_skill = (
            getattr(configuration, "skill_name", None)
            if getattr(configuration, "enable_project_skill", False)
            else None
        )
        context = {
            "task": metadata.get("task", "-"),
            "batch": metadata.get("batch", metadata.get("shard", "-")),
            "round": metadata.get("round", "-"),
            "repair": metadata.get("repair", 0),
            "skill": project_skill or "-",
        }
        started = time.perf_counter()
        start_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "call_id": call_id,
            "provider": self.name,
            "model": self.model,
            **context,
        }
        self.trace._write(start_event)
        print(
            "[LLM][START] "
            f"call={call_id} provider={self.name} model={self.model} "
            f"task={context['task']} batch={context['batch']} "
            f"round={context['round']} repair={context['repair']} "
            f"skill={context['skill']}",
            file=sys.stderr,
            flush=True,
        )
        try:
            response = self.provider.complete(request)
        except Exception as error:
            elapsed = time.perf_counter() - started
            message = str(error)[:1000]
            category, likely_cause = _diagnose_error(error)
            self.trace._write(
                {
                    **start_event,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "elapsed_seconds": round(elapsed, 3),
                    "error_type": type(error).__name__,
                    "error": message,
                    "error_category": category,
                    "likely_cause": likely_cause,
                }
            )
            print(
                "[LLM][ERROR] "
                f"call={call_id} provider={self.name} model={self.model} "
                f"task={context['task']} batch={context['batch']} "
                f"round={context['round']} repair={context['repair']} "
                f"skill={context['skill']} "
                f"elapsed={elapsed:.1f}s category={category} "
                f"cause={likely_cause} error={type(error).__name__}: {message}",
                file=sys.stderr,
                flush=True,
            )
            raise
        elapsed = time.perf_counter() - started
        content_hash = hashlib.sha256(response.content.encode("utf-8")).hexdigest()
        self.trace._write(
            {
                **start_event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "elapsed_seconds": round(elapsed, 3),
                "response_model": response.response_model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "attempts": response.attempts,
                "content_sha256": content_hash,
                "cost_usd": response.raw_metadata.get("total_cost_usd"),
                "project_skill": response.raw_metadata.get("project_skill"),
                "read_tools": response.raw_metadata.get("read_tools"),
                "tools_used": response.raw_metadata.get("tools_used"),
                "tool_call_count": response.raw_metadata.get("tool_call_count"),
            }
        )
        print(
            "[LLM][DONE] "
            f"call={call_id} provider={self.name} model={self.model} "
            f"task={context['task']} batch={context['batch']} "
            f"round={context['round']} repair={context['repair']} "
            f"skill={context['skill']} "
            f"elapsed={elapsed:.1f}s tokens={response.usage.total_tokens} "
            f"cost_usd={response.raw_metadata.get('total_cost_usd', '-')} "
            f"tools={response.raw_metadata.get('tools_used', ())}",
            file=sys.stderr,
            flush=True,
        )
        return response
