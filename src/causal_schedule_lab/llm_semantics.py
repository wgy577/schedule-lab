"""LLM-first, evidence-grounded semantic compilation for unknown projects."""

from __future__ import annotations

import ast
import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .constraint_impact import (
    ConstraintImpactReport,
    assess_constraint_impacts_with_llm,
)
from .providers.base import ModelProvider, ModelRequest, ModelResponse, TokenUsage
from .semantic_compiler import index_repository
from .semantic_knowledge import (
    EngineeringPatternHit,
    KnowledgeHit,
    SchedulingKnowledgeBase,
    compact_knowledge_context,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectType(StrEnum):
    SCHEDULING_OPTIMIZATION = "scheduling_optimization"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    MACHINE_LEARNING = "machine_learning"
    DATA_PIPELINE = "data_pipeline"
    WEB_APPLICATION = "web_application"
    SCIENTIFIC_COMPUTING = "scientific_computing"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"


class ProblemFamily(StrEnum):
    JSP = "JSP"
    FSP = "FSP"
    FJSP = "FJSP"
    HFSP = "HFSP"
    RCPSP = "RCPSP"
    JSSP = "JSSP"
    VRP = "VRP"
    OTHER = "other"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ObjectiveKind(StrEnum):
    MAKESPAN = "makespan"
    TOTAL_TARDINESS = "total_tardiness"
    MAX_TARDINESS = "max_tardiness"
    FLOW_TIME = "flow_time"
    COST = "cost"
    ENERGY = "energy"
    THROUGHPUT = "throughput"
    ROBUSTNESS = "robustness"
    LEXICOGRAPHIC = "lexicographic"
    WEIGHTED_MULTI_OBJECTIVE = "weighted_multi_objective"
    FEASIBILITY_ONLY = "feasibility_only"
    OTHER = "other"
    UNKNOWN = "unknown"


class ObjectiveSense(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    SATISFY = "satisfy"
    UNKNOWN = "unknown"


class EnvironmentKind(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    DISCRETE_TIME = "discrete_time"
    CONTINUOUS_TIME = "continuous_time"
    SIMULATION_BACKED = "simulation_backed"
    REAL_TIME = "real_time"
    OFFLINE = "offline"
    MULTI_RESOURCE = "multi_resource"
    MULTI_AGENT = "multi_agent"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConstraintKind(StrEnum):
    PRECEDENCE = "precedence"
    RESOURCE_CAPACITY = "resource_capacity"
    RESOURCE_ELIGIBILITY = "resource_eligibility"
    NO_OVERLAP = "no_overlap"
    RELEASE_TIME = "release_time"
    DUE_DATE = "due_date"
    TIME_WINDOW = "time_window"
    CALENDAR = "calendar"
    SETUP_TIME = "setup_time"
    TRANSPORT = "transport"
    ROUTE_CONTINUITY = "route_continuity"
    COLLISION_AVOIDANCE = "collision_avoidance"
    BUFFER = "buffer"
    BLOCKING = "blocking"
    NO_WAIT = "no_wait"
    BATCHING = "batching"
    SYNCHRONIZATION = "synchronization"
    CHOICE_BINDING = "choice_binding"
    DOMAIN_ORACLE = "domain_oracle"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConstraintScope(StrEnum):
    GLOBAL = "global"
    PROJECT = "project"
    JOB = "job"
    OPERATION = "operation"
    RESOURCE = "resource"
    ROUTE = "route"
    TIME = "time"
    OTHER = "other"
    UNKNOWN = "unknown"


class DecisionKind(StrEnum):
    SELECT = "select"
    ASSIGN_RESOURCE = "assign_resource"
    SEQUENCE = "sequence"
    START_TIME = "start_time"
    RELEASE = "release"
    ROUTE = "route"
    BATCH = "batch"
    ACCEPT_REJECT = "accept_reject"
    OTHER = "other"
    UNKNOWN = "unknown"


class OracleKind(StrEnum):
    STATIC_VALIDATOR = "static_validator"
    EXACT_SOLVER = "exact_solver"
    HEURISTIC_SOLVER = "heuristic_solver"
    SIMULATOR = "simulator"
    DIGITAL_TWIN = "digital_twin"
    EXTERNAL_SERVICE = "external_service"
    HUMAN_REVIEW = "human_review"
    NONE_FOUND = "none_found"
    OTHER = "other"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceCitation(FrozenModel):
    file: str
    symbol: str | None = None
    detail: str = Field(min_length=1, max_length=500)


class ObjectiveFinding(FrozenModel):
    id: str = Field(pattern=r"^objective_[0-9]+$")
    kind: ObjectiveKind
    sense: ObjectiveSense
    priority: int = Field(ge=1, le=10)
    statement: str = Field(min_length=1, max_length=1000)
    evidence: tuple[EvidenceCitation, ...] = Field(min_length=1)
    confidence: Confidence


class ConstraintFinding(FrozenModel):
    id: str = Field(pattern=r"^constraint_[0-9]+$")
    kind: ConstraintKind
    scope: ConstraintScope
    hard: bool
    statement: str = Field(min_length=1, max_length=1200)
    evidence: tuple[EvidenceCitation, ...] = Field(min_length=1)
    confidence: Confidence


class DecisionFinding(FrozenModel):
    id: str = Field(pattern=r"^decision_[0-9]+$")
    kind: DecisionKind
    modifiable: bool
    statement: str = Field(min_length=1, max_length=1000)
    evidence: tuple[EvidenceCitation, ...] = Field(min_length=1)
    confidence: Confidence


class OracleFinding(FrozenModel):
    id: str = Field(pattern=r"^oracle_[0-9]+$")
    kind: OracleKind
    required: bool
    statement: str = Field(min_length=1, max_length=1000)
    evidence: tuple[EvidenceCitation, ...] = ()
    confidence: Confidence


class EnvironmentFinding(FrozenModel):
    kind: EnvironmentKind
    statement: str = Field(min_length=1, max_length=1000)
    evidence: tuple[EvidenceCitation, ...] = Field(min_length=1)
    confidence: Confidence


class UnknownFinding(FrozenModel):
    category: Literal[
        "project_type",
        "problem_family",
        "objective",
        "environment",
        "constraint",
        "decision",
        "oracle",
        "data",
        "other",
    ]
    question: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)
    suggested_evidence: str = Field(min_length=1, max_length=1000)


class SemanticAnalysis(FrozenModel):
    schema_version: Literal["1.0"]
    language: Literal["zh-CN"]
    summary: str = Field(min_length=1, max_length=2000)
    project_type: ProjectType
    problem_families: tuple[ProblemFamily, ...] = Field(min_length=1)
    environments: tuple[EnvironmentFinding, ...] = Field(min_length=1)
    objectives: tuple[ObjectiveFinding, ...] = Field(min_length=1)
    constraints: tuple[ConstraintFinding, ...] = Field(min_length=1)
    decisions: tuple[DecisionFinding, ...] = Field(min_length=1)
    oracles: tuple[OracleFinding, ...] = ()
    allowed_interventions: tuple[DecisionKind, ...] = ()
    unknowns: tuple[UnknownFinding, ...] = ()
    overall_confidence: Confidence

    @model_validator(mode="after")
    def unique_ids(self) -> "SemanticAnalysis":
        identifiers = [
            *(item.id for item in self.objectives),
            *(item.id for item in self.constraints),
            *(item.id for item in self.decisions),
            *(item.id for item in self.oracles),
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("finding IDs must be unique")
        return self


class FileEvidence(FrozenModel):
    path: str
    sha256: str
    excerpt: str


class RepositoryEvidencePacket(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    project_name: str
    files: tuple[FileEvidence, ...]
    omitted_file_count: int = Field(default=0, ge=0)
    character_count: int = Field(default=0, ge=0)
    packet_sha256: str


class EvidenceCheck(FrozenModel):
    finding_id: str
    file: str
    symbol: str | None = None
    valid: bool
    reason: str


class LLMCompilationMetadata(FrozenModel):
    provider: str
    requested_model: str
    response_model: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage
    latency_seconds: float = Field(ge=0.0)
    provider_attempts: int = Field(ge=1)
    schema_repairs: int = Field(ge=0)


class LLMSemanticCompilation(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    review_status: Literal["pending_human_review"] = "pending_human_review"
    packet: RepositoryEvidencePacket
    knowledge_hits: tuple[KnowledgeHit, ...] = ()
    engineering_pattern_hits: tuple[EngineeringPatternHit, ...] = ()
    analysis: SemanticAnalysis
    constraint_impact: ConstraintImpactReport | None = None
    evidence_checks: tuple[EvidenceCheck, ...]
    metadata: LLMCompilationMetadata

    @property
    def evidence_pass_rate(self) -> float:
        if not self.evidence_checks:
            return 0.0
        return sum(item.valid for item in self.evidence_checks) / len(
            self.evidence_checks
        )


_SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "outputs",
    "checkpoints",
}
_TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}


def _priority(path: Path) -> tuple[int, str]:
    relative = path.as_posix().lower()
    name = path.name.lower()
    if name in {"readme.md", "pyproject.toml", "package.json"}:
        return 0, relative
    if "manifest" in name or "config" in relative:
        return 1, relative
    if "/src/" in f"/{relative}" or relative.startswith("src/"):
        return 2, relative
    if "/test" in f"/{relative}" or relative.startswith("test"):
        return 3, relative
    if relative.startswith("docs/"):
        return 4, relative
    return 5, relative


def _numbered(lines: list[str], indexes: Iterable[int]) -> str:
    chosen = sorted({index for index in indexes if 0 <= index < len(lines)})
    return "\n".join(f"{index + 1:04d}: {lines[index]}" for index in chosen)


def _python_excerpt(text: str, *, max_chars: int) -> str:
    lines = text.splitlines()
    indexes = set(range(min(45, len(lines))))
    try:
        tree = ast.parse(text)
        definitions = [
            node
            for node in tree.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ]
        for node in definitions[:14]:
            indexes.update(
                range(
                    max(0, node.lineno - 2),
                    min(len(lines), node.lineno + 20),
                )
            )
    except SyntaxError:
        indexes.update(range(min(100, len(lines))))
    return _numbered(lines, indexes)[:max_chars]


def _text_excerpt(path: Path, text: str, *, max_chars: int) -> str:
    if path.suffix.lower() == ".py":
        return _python_excerpt(text, max_chars=max_chars)
    lines = text.splitlines()
    return _numbered(lines, range(min(len(lines), 180)))[:max_chars]


def build_repository_evidence_packet(
    root: str | Path,
    *,
    max_characters: int = 60000,
    max_files: int = 48,
    max_characters_per_file: int = 5000,
    include_paths: Iterable[str] | None = None,
    excluded_prefixes: Iterable[str] = (),
) -> RepositoryEvidencePacket:
    base = Path(root).expanduser().resolve()
    included = None if include_paths is None else set(include_paths)
    excluded = tuple(
        item.strip("/")
        for item in excluded_prefixes
        if item.strip("/")
    )
    candidates = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        relative = path.relative_to(base)
        relative_name = relative.as_posix()
        if (
            any(part in _SKIP_PARTS for part in relative.parts)
            or path.name == ".env"
            or path.name.startswith(".env.")
            or (
                included is not None
                and relative_name not in included
            )
            or any(
                relative_name == prefix
                or relative_name.startswith(prefix + "/")
                for prefix in excluded
            )
        ):
            continue
        candidates.append(path)
    candidates.sort(key=lambda path: _priority(path.relative_to(base)))
    files = []
    character_count = 0
    for path in candidates:
        if len(files) >= max_files or character_count >= max_characters:
            break
        text = path.read_text(encoding="utf-8", errors="replace")
        remaining = max_characters - character_count
        excerpt = _text_excerpt(
            path,
            text,
            max_chars=min(max_characters_per_file, remaining),
        )
        if not excerpt.strip():
            continue
        relative = path.relative_to(base).as_posix()
        files.append(
            FileEvidence(
                path=relative,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                excerpt=excerpt,
            )
        )
        character_count += len(excerpt)
    normalized = json.dumps(
        [item.model_dump(mode="json") for item in files],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return RepositoryEvidencePacket(
        project_name=base.name,
        files=tuple(files),
        omitted_file_count=max(0, len(candidates) - len(files)),
        character_count=character_count,
        packet_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def choice_library() -> dict[str, list[str]]:
    return {
        "project_type": [item.value for item in ProjectType],
        "problem_family": [item.value for item in ProblemFamily],
        "objective_kind": [item.value for item in ObjectiveKind],
        "objective_sense": [item.value for item in ObjectiveSense],
        "environment_kind": [item.value for item in EnvironmentKind],
        "constraint_kind": [item.value for item in ConstraintKind],
        "constraint_scope": [item.value for item in ConstraintScope],
        "decision_kind": [item.value for item in DecisionKind],
        "oracle_kind": [item.value for item in OracleKind],
        "confidence": [item.value for item in Confidence],
    }


_SYSTEM_PROMPT = """\
你是“项目语义编译 Agent”，负责理解一个完整软件项目。
你的任务不是写代码，而是从给定项目证据包中识别项目类型、调度问题族、环境、
目标、硬约束、决策变量、可修改项、Oracle 和未知信息。

严格规则：
1. 所有分类字段只能从给出的选项库选择，绝不创造新枚举值。
2. 无法可靠归类时选择 other 或 unknown，并在 statement/unknowns 中解释。
3. 每个目标、约束、决策和环境判断必须引用证据包中真实存在的 file；
   symbol 只在证据明确出现时填写，否则为 null。
4. 不能从文件名臆测业务事实，不能把计划文档当作已运行结果。
5. 区分硬约束、软目标、实现限制和研究计划。
6. 不得把 LLM 判断标记为 verified；输出将进入程序证据审计和人工审核。
7. 输出必须是一个 JSON 对象，不要 Markdown，不要代码围栏，不要额外解释。
8. 使用简体中文描述，但枚举值、schema_version 和 language 必须保持指定格式。
9. 必须返回一个且只有一个根 JSON 对象，不能逐条返回多个对象。
10. 调度知识库只提供常见问题族先验，不能作为当前项目证据；优先复用已知的经典
    定义，把主要注意力放在项目相对经典问题新增、删除或修改的变体语义。
11. 论文对比算法、benchmark、训练硬件和作者性能主张不是项目调度硬约束。
"""


_ROOT_KEYS = (
    "schema_version",
    "language",
    "summary",
    "project_type",
    "problem_families",
    "environments",
    "objectives",
    "constraints",
    "decisions",
    "oracles",
    "allowed_interventions",
    "unknowns",
    "overall_confidence",
)


def _answer_sheet_template() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "language": "zh-CN",
        "summary": "填写项目摘要",
        "project_type": "unknown",
        "problem_families": ["unknown"],
        "environments": [],
        "objectives": [],
        "constraints": [],
        "decisions": [],
        "oracles": [],
        "allowed_interventions": [],
        "unknowns": [],
        "overall_confidence": "low",
    }


def _analysis_prompt(
    packet: RepositoryEvidencePacket,
    knowledge_hits: tuple[KnowledgeHit, ...] = (),
    engineering_hits: tuple[EngineeringPatternHit, ...] = (),
) -> str:
    schema = SemanticAnalysis.model_json_schema()
    return (
        "请完成以下选择题式项目语义分析。\n\n"
        "选项库：\n"
        + json.dumps(choice_library(), ensure_ascii=False, indent=2)
        + "\n\n必须完整复制并填写下面这一张根答题卡，不能拆成多个 JSON：\n"
        + json.dumps(_answer_sheet_template(), ensure_ascii=False, indent=2)
        + "\n\n输出 JSON Schema：\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\n检索到的经典调度知识先验（不可作为项目 evidence 引用）：\n"
        + compact_knowledge_context(knowledge_hits, engineering_hits)
        + "\n\n项目证据包：\n"
        + packet.model_dump_json(indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
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
        if isinstance(value, dict) and set(_ROOT_KEYS) <= set(value):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates = []
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            if isinstance(value, dict):
                score = len(set(value) & set(_ROOT_KEYS))
                candidates.append((score, len(json.dumps(value)), value))
        except json.JSONDecodeError:
            continue
    if candidates:
        score, _, value = max(candidates, key=lambda item: (item[0], item[1]))
        if score == len(_ROOT_KEYS):
            return value
        raise ValueError(
            "model output does not contain a complete SemanticAnalysis root "
            f"object; best candidate has {score}/{len(_ROOT_KEYS)} root keys"
        )
    raise ValueError("model output does not contain a JSON object")


def _repair_prompt(content: str, error: Exception) -> str:
    return (
        "上一次输出未通过 JSON Schema。请只修复格式和不合法选项，保持有证据的"
        "语义判断，不要增加选项库之外的值。\n\n校验错误：\n"
        + str(error)[:5000]
        + "\n\n上一次输出：\n"
        + content[:30000]
        + "\n\n必须返回的完整根答题卡：\n"
        + json.dumps(_answer_sheet_template(), ensure_ascii=False, indent=2)
        + "\n\n目标 JSON Schema：\n"
        + json.dumps(
            SemanticAnalysis.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
    )


def _citations(
    analysis: SemanticAnalysis,
) -> Iterable[tuple[str, EvidenceCitation]]:
    for item in analysis.objectives:
        for citation in item.evidence:
            yield item.id, citation
    for item in analysis.constraints:
        for citation in item.evidence:
            yield item.id, citation
    for item in analysis.decisions:
        for citation in item.evidence:
            yield item.id, citation
    for index, item in enumerate(analysis.environments, start=1):
        for citation in item.evidence:
            yield f"environment_{index}", citation
    for item in analysis.oracles:
        for citation in item.evidence:
            yield item.id, citation


def audit_llm_evidence(
    root: str | Path,
    packet: RepositoryEvidencePacket,
    analysis: SemanticAnalysis,
) -> tuple[EvidenceCheck, ...]:
    base = Path(root).expanduser().resolve()
    packet_files = {item.path for item in packet.files}
    symbols = {
        (item.file, item.symbol)
        for item in index_repository(base)
    }
    # LLMs naturally cite class methods as ``ClassName.method`` while the
    # lightweight AST index stores the method node as ``method``.  Accept that
    # qualified spelling only when the terminal symbol exists in the same file.
    qualified_symbols = {
        (file, symbol.rsplit(".", 1)[-1])
        for file, symbol in (
            (citation.file, citation.symbol)
            for _, citation in _citations(analysis)
            if citation.symbol is not None and "." in citation.symbol
        )
    }
    checks = []
    for finding_id, citation in _citations(analysis):
        candidate = (base / citation.file).resolve()
        valid_path = (
            citation.file in packet_files
            and candidate.is_relative_to(base)
            and candidate.is_file()
        )
        valid_symbol = (
            citation.symbol is None
            or (citation.file, citation.symbol) in symbols
            or (
                (citation.file, citation.symbol.rsplit(".", 1)[-1])
                in symbols & qualified_symbols
            )
        )
        valid = valid_path and valid_symbol
        reason = (
            "file and symbol are present in the indexed evidence"
            if valid
            else "file is absent from packet or project"
            if not valid_path
            else "symbol is absent from the indexed file"
        )
        checks.append(
            EvidenceCheck(
                finding_id=finding_id,
                file=citation.file,
                symbol=citation.symbol,
                valid=valid,
                reason=reason,
            )
        )
    return tuple(checks)


def _aggregate_metadata(
    responses: list[ModelResponse],
    *,
    schema_repairs: int,
) -> LLMCompilationMetadata:
    last = responses[-1]
    return LLMCompilationMetadata(
        provider=last.provider,
        requested_model=last.requested_model,
        response_model=last.response_model,
        request_id=last.request_id,
        finish_reason=last.finish_reason,
        usage=TokenUsage(
            input_tokens=sum(item.usage.input_tokens for item in responses),
            output_tokens=sum(item.usage.output_tokens for item in responses),
            total_tokens=sum(item.usage.total_tokens for item in responses),
        ),
        latency_seconds=sum(item.latency_seconds for item in responses),
        provider_attempts=sum(item.attempts for item in responses),
        schema_repairs=schema_repairs,
    )


def compile_project_semantics_with_llm(
    root: str | Path,
    *,
    provider: ModelProvider,
    max_input_characters: int = 60000,
    max_output_tokens: int = 6000,
    max_schema_repairs: int = 1,
    thinking_mode: Literal["enabled", "disabled", "adaptive"] | None = "enabled",
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "max",
    include_paths: Iterable[str] | None = None,
    excluded_prefixes: Iterable[str] = (),
    knowledge_base_path: str | Path | None = None,
    impact_provider: ModelProvider | None = None,
    impact_max_output_tokens: int = 12000,
) -> LLMSemanticCompilation:
    packet = build_repository_evidence_packet(
        root,
        max_characters=max_input_characters,
        include_paths=include_paths,
        excluded_prefixes=excluded_prefixes,
    )
    if not packet.files:
        raise ValueError("project evidence packet is empty")
    knowledge_base = SchedulingKnowledgeBase.load(knowledge_base_path)
    packet_text = "\n".join(
        f"{item.path}\n{item.excerpt}" for item in packet.files
    )
    initial_hits = knowledge_base.retrieve(packet_text, top_k=2)
    initial_engineering_hits = knowledge_base.retrieve_engineering_patterns(
        packet_text,
        top_k=8,
    )
    responses = [
        provider.complete(
            ModelRequest(
                system=_SYSTEM_PROMPT,
                user=_analysis_prompt(
                    packet,
                    initial_hits,
                    initial_engineering_hits,
                ),
                temperature=0.0,
                max_output_tokens=max_output_tokens,
                require_json=True,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
                metadata={"task": "project_semantic_compilation"},
            )
        )
    ]
    schema_repairs = 0
    while True:
        try:
            analysis = SemanticAnalysis.model_validate(
                extract_json_object(responses[-1].content)
            )
            break
        except (ValueError, ValidationError) as error:
            if schema_repairs >= max_schema_repairs:
                raise ValueError(
                    "LLM semantic output failed schema validation after "
                    f"{schema_repairs} repair(s): {error}"
                ) from error
            schema_repairs += 1
            responses.append(
                provider.complete(
                    ModelRequest(
                        system=_SYSTEM_PROMPT,
                        user=_repair_prompt(responses[-1].content, error),
                        temperature=0.0,
                        max_output_tokens=max_output_tokens,
                        require_json=True,
                        thinking_mode=thinking_mode,
                        reasoning_effort=reasoning_effort,
                        metadata={
                            "task": "project_semantic_schema_repair",
                            "repair": schema_repairs,
                        },
                    )
                )
            )
    final_hits = knowledge_base.retrieve(
        packet_text,
        family_hints=tuple(
            item.value for item in analysis.problem_families
        ),
        top_k=4,
    )
    final_engineering_hits = knowledge_base.retrieve_engineering_patterns(
        packet_text,
        family_hints=tuple(
            item.value for item in analysis.problem_families
        ),
        top_k=8,
    )
    impact_report = (
        assess_constraint_impacts_with_llm(
            analysis,
            final_hits,
            provider=impact_provider,
            engineering_hits=final_engineering_hits,
            max_output_tokens=impact_max_output_tokens,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        if impact_provider is not None
        else None
    )
    return LLMSemanticCompilation(
        packet=packet,
        knowledge_hits=final_hits,
        engineering_pattern_hits=final_engineering_hits,
        analysis=analysis,
        constraint_impact=impact_report,
        evidence_checks=audit_llm_evidence(root, packet, analysis),
        metadata=_aggregate_metadata(
            responses,
            schema_repairs=schema_repairs,
        ),
    )
