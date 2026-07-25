"""Blind paper-label versus code-only semantic recognition harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pypdf import PdfReader

from .llm_semantics import (
    LLMSemanticCompilation,
    SemanticAnalysis,
    choice_library,
    compile_project_semantics_with_llm,
    extract_json_object,
)
from .providers.base import ModelProvider, ModelRequest, ModelResponse, TokenUsage
from .semantic_compiler import index_repository


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HarnessCase(FrozenModel):
    schema_version: Literal["1.0"]
    case_id: str
    title: str
    paper_url: str
    repository_url: str
    repository_ref: str
    paper_path: str
    repository_path: str
    excluded_code_paths: tuple[str, ...] = ()
    label_status: Literal["draft_llm_label", "human_verified_label"]
    notes: str = ""


class RepositoryFile(FrozenModel):
    path: str
    suffix: str
    bytes: int = Field(ge=0)
    symbols: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class RepositoryReadPlan(FrozenModel):
    schema_version: Literal["1.0"]
    focus_questions: tuple[
        Literal[
            "project_type",
            "problem_family",
            "environment",
            "objective",
            "constraints",
            "decisions",
            "oracles",
            "training",
            "evaluation",
        ],
        ...,
    ] = Field(min_length=1)
    selected_files: tuple[str, ...] = Field(min_length=1, max_length=24)
    rationale: str = Field(min_length=1, max_length=2000)
    expected_gaps: tuple[str, ...] = ()
    paper_used: Literal[False]


class PaperSemanticLabel(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    label_status: Literal["draft_llm_label", "human_verified_label"]
    paper_sha256: str
    semantics: SemanticAnalysis
    evidence_pass_rate: float = Field(ge=0.0, le=1.0)
    evidence_violations: tuple[str, ...] = ()
    metadata: dict[str, Any]


class MetricScore(FrozenModel):
    name: str
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    exact: bool
    expected: tuple[str, ...]
    predicted: tuple[str, ...]


class HarnessEvaluation(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    label_status: str
    blind: bool
    metrics: tuple[MetricScore, ...]
    macro_f1: float = Field(ge=0.0, le=1.0)
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    evidence_pass_rate: float = Field(ge=0.0, le=1.0)
    prediction_review_status: str
    conclusion: str


class SemanticHarnessResult(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case: HarnessCase
    read_plan: RepositoryReadPlan
    planner_metadata: dict[str, Any]
    label: PaperSemanticLabel
    prediction: LLMSemanticCompilation
    evaluation: HarnessEvaluation


def load_case(path: str | Path) -> tuple[HarnessCase, Path, Path]:
    source = Path(path).expanduser().resolve()
    case = HarnessCase.model_validate_json(source.read_text(encoding="utf-8"))
    paper = (source.parent / case.paper_path).resolve()
    repository = (source.parent / case.repository_path).resolve()
    if not paper.is_file():
        raise FileNotFoundError(f"paper not found: {paper}")
    if not repository.is_dir():
        raise FileNotFoundError(f"repository not found: {repository}")
    return case, paper, repository


def extract_paper_text(path: str | Path, *, max_characters: int = 50000) -> str:
    source = Path(path).expanduser().resolve()
    reader = PdfReader(str(source))
    pieces = []
    size = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        block = f"\n--- PAGE {page_number} ---\n{text.strip()}\n"
        if size + len(block) > max_characters:
            block = block[: max(0, max_characters - size)]
        pieces.append(block)
        size += len(block)
        if size >= max_characters:
            break
    result = "".join(pieces).strip()
    if not result:
        raise ValueError(f"paper contains no extractable text: {source}")
    return result


T = TypeVar("T", bound=BaseModel)


def _structured_call(
    provider: ModelProvider,
    *,
    system: str,
    user: str,
    model_type: type[T],
    max_output_tokens: int,
    max_repairs: int = 1,
) -> tuple[T, tuple[ModelResponse, ...]]:
    responses = []
    prompt = user
    for repair in range(max_repairs + 1):
        response = provider.complete(
            ModelRequest(
                system=system,
                user=prompt,
                temperature=0.0,
                max_output_tokens=max_output_tokens,
                require_json=True,
                thinking_mode="enabled",
                reasoning_effort="max",
            )
        )
        responses.append(response)
        try:
            if model_type is SemanticAnalysis:
                payload = extract_json_object(response.content)
            else:
                payload = _extract_generic_json_object(response.content)
            return model_type.model_validate(payload), tuple(responses)
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            if repair >= max_repairs:
                raise ValueError(
                    f"{model_type.__name__} failed after {repair} repair(s): {error}"
                ) from error
            prompt = (
                "上一次答题卡没有通过校验。只修复 JSON 和选项，不要增加无证据事实。"
                "\n错误：\n"
                + str(error)[:4000]
                + "\n上次输出：\n"
                + response.content[:24000]
                + "\n目标 Schema：\n"
                + json.dumps(
                    model_type.model_json_schema(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    raise AssertionError("unreachable")


def _extract_generic_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError("model output does not contain a JSON object")
    return max(candidates, key=lambda item: len(json.dumps(item)))


def _paper_citations(analysis: SemanticAnalysis):
    for item in analysis.environments:
        yield from item.evidence
    for collection in (
        analysis.objectives,
        analysis.constraints,
        analysis.decisions,
        analysis.oracles,
    ):
        for item in collection:
            yield from item.evidence


def _audit_paper_label_evidence(
    analysis: SemanticAnalysis,
    *,
    paper_name: str,
) -> tuple[float, tuple[str, ...]]:
    citations = tuple(_paper_citations(analysis))
    violations = tuple(
        f"{citation.file}:{citation.symbol or '-'}"
        for citation in citations
        if citation.file != paper_name or citation.symbol is not None
    )
    passed = len(citations) - len(violations)
    rate = passed / len(citations) if citations else 0.0
    return rate, violations


def _metadata(responses: tuple[ModelResponse, ...]) -> dict[str, Any]:
    last = responses[-1]
    usage = TokenUsage(
        input_tokens=sum(item.usage.input_tokens for item in responses),
        output_tokens=sum(item.usage.output_tokens for item in responses),
        total_tokens=sum(item.usage.total_tokens for item in responses),
    )
    return {
        "provider": last.provider,
        "requested_model": last.requested_model,
        "response_model": last.response_model,
        "request_id": last.request_id,
        "usage": usage.model_dump(),
        "latency_seconds": sum(item.latency_seconds for item in responses),
        "calls": len(responses),
    }


def build_paper_label(
    case: HarnessCase,
    paper_path: str | Path,
    *,
    provider: ModelProvider,
    max_characters: int = 50000,
) -> PaperSemanticLabel:
    source = Path(paper_path).resolve()
    paper_text = extract_paper_text(source, max_characters=max_characters)
    system = (
        "你是调度研究论文标签编译器。只根据论文正文提取问题定义、环境、目标、"
        "硬约束、决策和 Oracle。所有分类只能从选项库选择。每项证据的 file 必须"
        f"填写 {source.name!r}，symbol 必须为 null，detail 写页码和内容依据。"
        "结果是待人工审核标签，不允许参考代码。只输出一个完整 JSON 根对象。"
    )
    user = (
        "选项库：\n"
        + json.dumps(choice_library(), ensure_ascii=False, indent=2)
        + "\nSchema：\n"
        + json.dumps(
            SemanticAnalysis.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n论文文本：\n"
        + paper_text
    )
    semantics, responses = _structured_call(
        provider,
        system=system,
        user=user,
        model_type=SemanticAnalysis,
        max_output_tokens=24000,
    )
    evidence_pass_rate, evidence_violations = _audit_paper_label_evidence(
        semantics,
        paper_name=source.name,
    )
    return PaperSemanticLabel(
        case_id=case.case_id,
        label_status=case.label_status,
        paper_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        semantics=semantics,
        evidence_pass_rate=evidence_pass_rate,
        evidence_violations=evidence_violations,
        metadata=_metadata(responses),
    )


def repository_inventory(
    root: str | Path,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> tuple[RepositoryFile, ...]:
    base = Path(root).resolve()
    symbols_by_file: dict[str, list[str]] = {}
    tags_by_file: dict[str, set[str]] = {}
    for item in index_repository(base):
        symbols_by_file.setdefault(item.file, []).append(item.symbol)
        tags_by_file.setdefault(item.file, set()).update(item.tags)
    result = []
    for path in base.rglob("*"):
        if not path.is_file() or any(
            part in {".git", ".venv", "__pycache__", "node_modules"}
            for part in path.relative_to(base).parts
        ):
            continue
        relative = path.relative_to(base).as_posix()
        if any(
            relative == prefix.strip("/")
            or relative.startswith(prefix.strip("/") + "/")
            for prefix in excluded_prefixes
        ):
            continue
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
        }:
            continue
        result.append(
            RepositoryFile(
                path=relative,
                suffix=path.suffix.lower(),
                bytes=path.stat().st_size,
                symbols=tuple(symbols_by_file.get(relative, ()))[:40],
                tags=tuple(sorted(tags_by_file.get(relative, ()))),
            )
        )
    return tuple(sorted(result, key=lambda item: item.path))


def plan_code_reading(
    repository: str | Path,
    *,
    excluded_prefixes: tuple[str, ...],
    provider: ModelProvider,
) -> tuple[RepositoryReadPlan, dict[str, Any]]:
    inventory = repository_inventory(
        repository,
        excluded_prefixes=excluded_prefixes,
    )
    allowed = {item.path for item in inventory}
    system = (
        "你是 code-only 项目理解 Agent 的证据规划器。论文是隐藏标签，禁止请求、"
        "引用或推测论文内容。你只能从 inventory 中选择最多 24 个文件，用于判断"
        "项目类型、问题族、环境、目标、硬约束、决策、Oracle、训练和评估。"
        "优先选择实现、配置和测试，不要只看 README。只输出 JSON。"
    )
    user = (
        "Schema：\n"
        + json.dumps(
            RepositoryReadPlan.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        + "\nInventory：\n"
        + json.dumps(
            [item.model_dump(mode="json") for item in inventory],
            ensure_ascii=False,
        )
    )
    plan, responses = _structured_call(
        provider,
        system=system,
        user=user,
        model_type=RepositoryReadPlan,
        max_output_tokens=16000,
    )
    invalid = sorted(set(plan.selected_files) - allowed)
    if invalid:
        raise ValueError(f"read plan selected files outside inventory: {invalid}")
    return plan, _metadata(responses)


def _items(analysis: SemanticAnalysis, field: str) -> set[str]:
    if field == "project_type":
        return {analysis.project_type.value}
    if field == "problem_families":
        return {item.value for item in analysis.problem_families}
    if field == "environments":
        return {item.kind.value for item in analysis.environments}
    if field == "objectives":
        return {f"{item.kind.value}:{item.sense.value}" for item in analysis.objectives}
    if field == "constraints":
        return {
            f"{item.kind.value}:{item.scope.value}:{str(item.hard).lower()}"
            for item in analysis.constraints
        }
    if field == "decisions":
        return {
            f"{item.kind.value}:{str(item.modifiable).lower()}"
            for item in analysis.decisions
        }
    if field == "oracles":
        return {f"{item.kind.value}:{str(item.required).lower()}" for item in analysis.oracles}
    if field == "allowed_interventions":
        return {item.value for item in analysis.allowed_interventions}
    raise KeyError(field)


def _score(name: str, expected: set[str], predicted: set[str]) -> MetricScore:
    intersection = len(expected & predicted)
    precision = intersection / len(predicted) if predicted else float(not expected)
    recall = intersection / len(expected) if expected else float(not predicted)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return MetricScore(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        exact=expected == predicted,
        expected=tuple(sorted(expected)),
        predicted=tuple(sorted(predicted)),
    )


def evaluate_prediction(
    case: HarnessCase,
    label: PaperSemanticLabel,
    prediction: LLMSemanticCompilation,
) -> HarnessEvaluation:
    fields = (
        "project_type",
        "problem_families",
        "environments",
        "objectives",
        "constraints",
        "decisions",
        "oracles",
        "allowed_interventions",
    )
    metrics = tuple(
        _score(
            field,
            _items(label.semantics, field),
            _items(prediction.analysis, field),
        )
        for field in fields
    )
    macro_f1 = sum(item.f1 for item in metrics) / len(metrics)
    exact_rate = sum(item.exact for item in metrics) / len(metrics)
    return HarnessEvaluation(
        case_id=case.case_id,
        label_status=label.label_status,
        blind=True,
        metrics=metrics,
        macro_f1=macro_f1,
        exact_match_rate=exact_rate,
        evidence_pass_rate=prediction.evidence_pass_rate,
        prediction_review_status=prediction.review_status,
        conclusion=(
            "draft comparison only; paper label and code prediction both require human review"
            if label.label_status == "draft_llm_label"
            else "comparison against a human-verified paper label"
        ),
    )


def run_blind_harness(
    case_path: str | Path,
    *,
    provider: ModelProvider,
    max_paper_characters: int = 40000,
    max_code_characters: int = 60000,
) -> SemanticHarnessResult:
    case, paper, repository = load_case(case_path)
    label = build_paper_label(
        case,
        paper,
        provider=provider,
        max_characters=max_paper_characters,
    )
    plan, plan_metadata = plan_code_reading(
        repository,
        excluded_prefixes=case.excluded_code_paths,
        provider=provider,
    )
    prediction = compile_project_semantics_with_llm(
        repository,
        provider=provider,
        max_input_characters=max_code_characters,
        max_output_tokens=24000,
        include_paths=plan.selected_files,
        excluded_prefixes=case.excluded_code_paths,
    )
    prediction = prediction.model_copy(
        update={
            "packet": prediction.packet.model_copy(
                update={
                    "project_name": prediction.packet.project_name
                    + ":blind-code-only"
                }
            )
        }
    )
    result = SemanticHarnessResult(
        case=case,
        read_plan=plan,
        planner_metadata=plan_metadata,
        label=label,
        prediction=prediction,
        evaluation=evaluate_prediction(case, label, prediction),
    )
    return result


def run_code_only_harness(
    case_path: str | Path,
    *,
    label: PaperSemanticLabel,
    provider: ModelProvider,
    max_code_characters: int = 60000,
) -> SemanticHarnessResult:
    """Evaluate one provider against a fixed paper label without reading paper."""

    case, _paper, repository = load_case(case_path)
    if label.case_id != case.case_id:
        raise ValueError(
            f"label case {label.case_id!r} does not match {case.case_id!r}"
        )
    plan, plan_metadata = plan_code_reading(
        repository,
        excluded_prefixes=case.excluded_code_paths,
        provider=provider,
    )
    prediction = compile_project_semantics_with_llm(
        repository,
        provider=provider,
        max_input_characters=max_code_characters,
        max_output_tokens=24000,
        include_paths=plan.selected_files,
        excluded_prefixes=case.excluded_code_paths,
    )
    prediction = prediction.model_copy(
        update={
            "packet": prediction.packet.model_copy(
                update={
                    "project_name": prediction.packet.project_name
                    + ":blind-code-only"
                }
            )
        }
    )
    return SemanticHarnessResult(
        case=case,
        read_plan=plan,
        planner_metadata=plan_metadata,
        label=label,
        prediction=prediction,
        evaluation=evaluate_prediction(case, label, prediction),
    )
