"""Staged, tool-mediated semantic analysis for unfamiliar software projects.

The cheap navigator builds a repository map.  A stronger analyst then works in
small semantic batches and may request bounded, auditable re-reads of related
symbols.  The program, not the model, decides whether each read is allowed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .constraint_impact import assess_constraint_impacts_with_llm
from .llm_semantics import (
    FileEvidence,
    FrozenModel,
    LLMSemanticCompilation,
    RepositoryEvidencePacket,
    SemanticAnalysis,
    _SYSTEM_PROMPT,
    _aggregate_metadata,
    _analysis_prompt,
    _repair_prompt,
    audit_llm_evidence,
    extract_json_object,
)
from .providers.base import ModelProvider, ModelRequest, ModelResponse, TokenUsage
from .semantic_knowledge import SchedulingKnowledgeBase
from .storage import HierarchicalMemory


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
_PAPER_MARKERS = {
    "paper",
    "papers",
    "article",
    "articles",
    "literature",
    "survey",
    "review",
    "reviews",
}
_SECRET_NAMES = {".env", "credentials.json", "secrets.json"}


def _model_json(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_object(text: str) -> dict[str, Any]:
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
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((len(json.dumps(value)), value))
    if not candidates:
        raise ValueError("model output does not contain a JSON object")
    return max(candidates, key=lambda item: item[0])[1]


def _is_paper_like(relative: Path) -> bool:
    lowered_parts = {part.lower() for part in relative.parts}
    stem_tokens = {
        token
        for token in relative.stem.lower().replace("-", "_").split("_")
        if token
    }
    return relative.suffix.lower() == ".pdf" or bool(
        (lowered_parts | stem_tokens) & _PAPER_MARKERS
    )


class FileRole(StrEnum):
    ENTRYPOINT = "entrypoint"
    SOURCE = "source"
    CONFIG = "config"
    TEST = "test"
    PROJECT_DOCUMENT = "project_document"
    DATA_SCHEMA = "data_schema"
    VALIDATOR = "validator"
    SOLVER = "solver"
    OTHER = "other"


class InventoryFile(FrozenModel):
    path: str
    role_hint: FileRole
    size_bytes: int = Field(ge=0)
    sha256: str
    symbols: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    excerpt: str


class NavigationInventory(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    project_name: str
    files: tuple[InventoryFile, ...]
    excluded_paper_count: int = Field(default=0, ge=0)
    omitted_file_count: int = Field(default=0, ge=0)


def _role_hint(relative: Path, symbols: tuple[str, ...]) -> FileRole:
    text = relative.as_posix().lower()
    name = relative.name.lower()
    if "test" in relative.parts or name.startswith("test_"):
        return FileRole.TEST
    if any(term in text for term in ("validator", "validation", "oracle")):
        return FileRole.VALIDATOR
    if any(term in text for term in ("solver", "optimiz", "search", "repair")):
        return FileRole.SOLVER
    if name in {"main.py", "cli.py", "__main__.py", "app.py"}:
        return FileRole.ENTRYPOINT
    if name in {"pyproject.toml", "package.json"} or any(
        term in text for term in ("config", "manifest")
    ):
        return FileRole.CONFIG
    if relative.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
        return FileRole.DATA_SCHEMA
    if relative.suffix.lower() == ".py" or symbols:
        return FileRole.SOURCE
    if relative.suffix.lower() == ".md":
        return FileRole.PROJECT_DOCUMENT
    return FileRole.OTHER


def _python_inventory(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return (), ()
    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append(module)
    return tuple(symbols[:80]), tuple(dict.fromkeys(imports[:80]))


def _navigation_excerpt(path: Path, text: str, max_chars: int) -> str:
    lines = text.splitlines()
    indexes = set(range(min(36, len(lines))))
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(text)
            for node in tree.body:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    indexes.update(
                        range(
                            max(0, node.lineno - 2),
                            min(len(lines), node.lineno + 8),
                        )
                    )
        except SyntaxError:
            pass
    chosen = sorted(indexes)
    return "\n".join(f"{i + 1:04d}: {lines[i]}" for i in chosen)[:max_chars]


def build_navigation_inventory(
    root: str | Path,
    *,
    max_files: int = 240,
    max_characters_per_file: int = 2600,
) -> NavigationInventory:
    """Index non-paper project material for the low-cost navigation stage."""

    base = Path(root).expanduser().resolve()
    candidates: list[Path] = []
    excluded_papers = 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if any(part in _SKIP_PARTS for part in relative.parts):
            continue
        if path.name in _SECRET_NAMES or path.name.startswith(".env."):
            continue
        if _is_paper_like(relative):
            excluded_papers += 1
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        candidates.append(path)
    candidates.sort(
        key=lambda item: (
            0 if item.name.lower() in {"readme.md", "pyproject.toml"} else 1,
            item.relative_to(base).as_posix(),
        )
    )
    files: list[InventoryFile] = []
    for path in candidates[:max_files]:
        text = path.read_text(encoding="utf-8", errors="replace")
        symbols, imports = (
            _python_inventory(text) if path.suffix.lower() == ".py" else ((), ())
        )
        relative = path.relative_to(base)
        files.append(
            InventoryFile(
                path=relative.as_posix(),
                role_hint=_role_hint(relative, symbols),
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                symbols=symbols,
                imports=imports,
                excerpt=_navigation_excerpt(
                    path, text, max_chars=max_characters_per_file
                ),
            )
        )
    return NavigationInventory(
        project_name=base.name,
        files=tuple(files),
        excluded_paper_count=excluded_papers,
        omitted_file_count=max(0, len(candidates) - len(files)),
    )


class NavigationModule(FrozenModel):
    id: str = Field(pattern=r"^module_[0-9]+$")
    name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=800)
    files: tuple[str, ...] = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    entry_symbols: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"]


class ProjectNavigation(FrozenModel):
    schema_version: Literal["1.0"]
    language: Literal["zh-CN"]
    project_summary: str = Field(min_length=1, max_length=1800)
    modules: tuple[NavigationModule, ...] = Field(min_length=1)
    entrypoints: tuple[str, ...] = ()
    configs: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    validators_or_oracles: tuple[str, ...] = ()
    priority_reading_paths: tuple[str, ...] = Field(min_length=1)
    uncertainties: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_module_ids(self) -> "ProjectNavigation":
        ids = [item.id for item in self.modules]
        if len(ids) != len(set(ids)):
            raise ValueError("navigation module IDs must be unique")
        return self


_NAV_SYSTEM = """\
你是低成本“代码仓库导航 Agent”。你只建立地图，不判断业务约束是否成立。
阅读给定的非论文项目文件，回答：文件和模块做什么、入口在哪里、模块如何依赖、
配置/测试/验证器在哪里、强模型下一步优先读什么。

规则：
1. 只能引用清单中存在的相对路径，不得发明文件、符号或依赖。
2. 论文、文章和外部知识不在本阶段输入，也不得据此补充事实。
3. 不输出目标、约束或优化结论；不确定就写入 uncertainties。
4. 所有模块 id 使用 module_1、module_2……，depends_on 只能引用这些 id。
5. 返回一个 JSON 对象，不要 Markdown 或解释。
"""


def _navigation_prompt(inventory: NavigationInventory) -> str:
    return (
        "请根据仓库清单建立强模型可执行的项目导航。role_hint 只是程序先验，"
        "请结合摘录校正。\n\n输出 JSON Schema：\n"
        + json.dumps(ProjectNavigation.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n\n仓库清单：\n"
        + inventory.model_dump_json(indent=2)
    )


def _split_navigation_inventory(
    inventory: NavigationInventory,
    *,
    max_chunk_characters: int = 70000,
) -> tuple[NavigationInventory, ...]:
    chunks: list[list[InventoryFile]] = []
    current: list[InventoryFile] = []
    current_size = 0
    for item in inventory.files:
        size = len(item.model_dump_json())
        if current and current_size + size > max_chunk_characters:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += size
    if current:
        chunks.append(current)
    return tuple(
        NavigationInventory(
            project_name=inventory.project_name,
            files=tuple(items),
            excluded_paper_count=inventory.excluded_paper_count,
            omitted_file_count=inventory.omitted_file_count,
        )
        for items in chunks
    )


def _navigation_consolidation_prompt(
    inventory: NavigationInventory,
    shards: tuple[ProjectNavigation, ...],
) -> str:
    file_catalog = [
        {
            "path": item.path,
            "role_hint": item.role_hint.value,
            "symbols": item.symbols,
            "imports": item.imports,
        }
        for item in inventory.files
    ]
    return (
        "下面是同一仓库不同分片的导航结果。请合并重复模块，重新连续编号 module_1…，"
        "修复跨分片依赖，并按强模型最有效的阅读顺序给出统一导航。不得增加文件目录中"
        "不存在的路径。\n\n输出 JSON Schema：\n"
        + json.dumps(ProjectNavigation.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n\n完整文件目录（不含正文）：\n"
        + json.dumps(file_catalog, ensure_ascii=False, indent=2)
        + "\n\n分片导航：\n"
        + json.dumps(
            [item.model_dump(mode="json") for item in shards],
            ensure_ascii=False,
            indent=2,
        )
    )


def _audit_navigation(
    navigation: ProjectNavigation, inventory: NavigationInventory
) -> ProjectNavigation:
    allowed = {item.path for item in inventory.files}
    referenced = {
        *navigation.entrypoints,
        *navigation.configs,
        *navigation.tests,
        *navigation.validators_or_oracles,
        *navigation.priority_reading_paths,
        *(path for module in navigation.modules for path in module.files),
    }
    invalid = sorted(referenced - allowed)
    if invalid:
        raise ValueError(f"navigation references files outside inventory: {invalid}")
    module_ids = {item.id for item in navigation.modules}
    bad_edges = sorted(
        {
            dep
            for item in navigation.modules
            for dep in item.depends_on
            if dep not in module_ids
        }
    )
    if bad_edges:
        raise ValueError(f"navigation references unknown module IDs: {bad_edges}")
    return navigation


def build_project_navigation_with_llm(
    root: str | Path,
    *,
    provider: ModelProvider,
    max_output_tokens: int = 5000,
) -> tuple[NavigationInventory, ProjectNavigation, tuple[ModelResponse, ...]]:
    inventory = build_navigation_inventory(root)
    inventory_chunks = _split_navigation_inventory(inventory)
    responses: list[ModelResponse] = []
    shard_maps: list[ProjectNavigation] = []
    for index, chunk in enumerate(inventory_chunks):
        response = provider.complete(
            ModelRequest(
                system=_NAV_SYSTEM,
                user=_navigation_prompt(chunk),
                temperature=0.0,
                max_output_tokens=max_output_tokens,
                require_json=True,
                thinking_mode="enabled",
                reasoning_effort="medium",
                metadata={
                    "task": "project_navigation_shard",
                    "shard": index,
                    "shard_count": len(inventory_chunks),
                },
            )
        )
        responses.append(response)
        try:
            shard = ProjectNavigation.model_validate(_extract_object(response.content))
        except (ValueError, ValidationError) as error:
            raise ValueError(
                f"project navigation shard {index} failed schema validation: {error}"
            ) from error
        _audit_navigation(shard, chunk)
        shard_maps.append(shard)
    if len(shard_maps) == 1:
        navigation = shard_maps[0]
    else:
        response = provider.complete(
            ModelRequest(
                system=_NAV_SYSTEM,
                user=_navigation_consolidation_prompt(
                    inventory, tuple(shard_maps)
                ),
                temperature=0.0,
                max_output_tokens=max_output_tokens,
                require_json=True,
                thinking_mode="enabled",
                reasoning_effort="medium",
                metadata={
                    "task": "project_navigation_consolidation",
                    "shard_count": len(shard_maps),
                },
            )
        )
        responses.append(response)
        try:
            navigation = ProjectNavigation.model_validate(
                _extract_object(response.content)
            )
        except (ValueError, ValidationError) as error:
            raise ValueError(
                f"project navigation consolidation failed schema validation: {error}"
            ) from error
    return inventory, _audit_navigation(navigation, inventory), tuple(responses)


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"
    NOT_APPLICABLE = "not_applicable"


class ReadReason(StrEnum):
    CROSS_FILE_DEPENDENCY = "cross_file_dependency"
    CONSTRAINT_IMPLEMENTATION = "constraint_implementation"
    CONFIG_RESOLUTION = "config_resolution"
    VALIDATOR_OR_ORACLE = "validator_or_oracle"
    RUNTIME_STATE_TRANSITION = "runtime_state_transition"
    UNKNOWN_SYMBOL = "unknown_symbol"


class ReadPriority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceAssessment(FrozenModel):
    question_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,80}$")
    sufficiency: EvidenceSufficiency
    evidence: tuple[str, ...] = ()
    missing_information: str | None = Field(default=None, max_length=800)
    requested_read_ids: tuple[str, ...] = ()


class EvidenceReadRequest(FrozenModel):
    id: str = Field(pattern=r"^read_[a-z0-9_]+$")
    source_file: str
    source_symbol: str | None = None
    target_file: str
    target_symbol: str | None = None
    reason: ReadReason
    priority: ReadPriority
    expected_evidence: str = Field(min_length=8, max_length=800)


class BatchFact(FrozenModel):
    id: str = Field(pattern=r"^fact_[a-z0-9_]+$")
    category: Literal[
        "project_type",
        "problem_family",
        "objective",
        "environment",
        "constraint",
        "decision",
        "oracle",
        "data",
        "unknown",
    ]
    statement: str = Field(min_length=1, max_length=1000)
    evidence: tuple[str, ...] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class BatchAnalysis(FrozenModel):
    schema_version: Literal["1.0"]
    batch_id: str
    facts: tuple[BatchFact, ...] = ()
    assessments: tuple[EvidenceAssessment, ...] = Field(min_length=1)
    read_requests: tuple[EvidenceReadRequest, ...] = ()
    contradictions: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    compressed_summary: str = Field(min_length=1, max_length=2200)
    ready_for_synthesis: bool

    @model_validator(mode="after")
    def requests_are_assessed(self) -> "BatchAnalysis":
        request_ids = {item.id for item in self.read_requests}
        referenced = {
            request_id
            for item in self.assessments
            for request_id in item.requested_read_ids
        }
        if request_ids - referenced:
            raise ValueError("every read request must be referenced by an assessment")
        return self


class ReadDecision(FrozenModel):
    request_id: str
    approved: bool
    reason_code: Literal[
        "approved",
        "evidence_already_sufficient",
        "unlinked_assessment",
        "path_not_allowed",
        "paper_or_secret",
        "duplicate",
        "unrelated_target",
        "reason_mismatch",
        "budget_exhausted",
        "low_priority",
        "invalid_symbol",
    ]
    detail: str


class ToolEvidence(FrozenModel):
    request_id: str
    path: str
    symbol: str | None = None
    sha256: str
    excerpt: str
    related_symbols: tuple[str, ...] = ()


class PythonSymbol(FrozenModel):
    file: str
    qualname: str
    name: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    calls: tuple[str, ...] = ()


class ProjectCodeIndex:
    """AST-backed allow-list and relation index for the code re-read tool."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.allowed_files: set[str] = set()
        self.symbols: dict[tuple[str, str], PythonSymbol] = {}
        self.by_terminal_name: dict[str, list[PythonSymbol]] = defaultdict(list)
        self.imports: dict[str, set[str]] = defaultdict(set)
        self._build()

    def _build(self) -> None:
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            if (
                any(part in _SKIP_PARTS for part in relative.parts)
                or path.name in _SECRET_NAMES
                or path.name.startswith(".env.")
                or _is_paper_like(relative)
                or path.suffix.lower() not in _TEXT_SUFFIXES
            ):
                continue
            relative_name = relative.as_posix()
            self.allowed_files.add(relative_name)
            if path.suffix.lower() != ".py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.imports[relative_name].update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.imports[relative_name].add(
                        "." * node.level + (node.module or "")
                    )
            self._index_symbols(relative_name, tree)

    def _index_symbols(self, file: str, tree: ast.Module) -> None:
        def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
            for node in nodes:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue
                qualname = f"{prefix}.{node.name}" if prefix else node.name
                calls = tuple(
                    dict.fromkeys(
                        _call_name(call.func)
                        for call in ast.walk(node)
                        if isinstance(call, ast.Call) and _call_name(call.func)
                    )
                )
                symbol = PythonSymbol(
                    file=file,
                    qualname=qualname,
                    name=node.name,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    calls=calls,
                )
                self.symbols[(file, qualname)] = symbol
                self.symbols.setdefault((file, node.name), symbol)
                self.by_terminal_name[node.name].append(symbol)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, qualname)

        visit(tree.body)

    def symbol(self, file: str, name: str | None) -> PythonSymbol | None:
        if name is None:
            return None
        return self.symbols.get((file, name)) or self.symbols.get(
            (file, name.rsplit(".", 1)[-1])
        )

    def related(self, request: EvidenceReadRequest) -> bool:
        if request.source_file == request.target_file:
            return True
        source = self.symbol(request.source_file, request.source_symbol)
        target = self.symbol(request.target_file, request.target_symbol)
        if source and target and (
            target.name in source.calls or target.qualname in source.calls
        ):
            return True
        target_module = request.target_file.removesuffix(".py").replace("/", ".")
        imported = self.imports.get(request.source_file, set())
        if any(
            target_module.endswith(item.lstrip("."))
            or item.lstrip(".").endswith(target_module.rsplit(".", 1)[-1])
            for item in imported
            if item
        ):
            return True
        if request.reason in {
            ReadReason.CONFIG_RESOLUTION,
            ReadReason.VALIDATOR_OR_ORACLE,
        }:
            lowered = request.target_file.lower()
            return any(
                marker in lowered
                for marker in ("config", "manifest", "valid", "oracle", "test")
            )
        return False

    def reason_is_plausible(self, request: EvidenceReadRequest) -> bool:
        target = f"{request.target_file} {request.target_symbol or ''}".lower()
        source_symbol = self.symbol(request.source_file, request.source_symbol)
        target_symbol = self.symbol(request.target_file, request.target_symbol)
        call_related = bool(
            source_symbol
            and target_symbol
            and (
                target_symbol.name in source_symbol.calls
                or target_symbol.qualname in source_symbol.calls
            )
        )
        imported = self.imports.get(request.source_file, set())
        target_module = request.target_file.removesuffix(".py").replace("/", ".")
        import_related = any(
            target_module.endswith(item.lstrip("."))
            or item.lstrip(".").endswith(target_module.rsplit(".", 1)[-1])
            for item in imported
            if item
        )
        if request.reason == ReadReason.CONFIG_RESOLUTION:
            return any(term in target for term in ("config", "manifest", "setting"))
        if request.reason == ReadReason.VALIDATOR_OR_ORACLE:
            return call_related or any(
                term in target for term in ("valid", "oracle", "check", "test")
            )
        if request.reason == ReadReason.CONSTRAINT_IMPLEMENTATION:
            return call_related or any(
                term in target
                for term in ("constraint", "valid", "feasib", "capacity", "overlap")
            )
        if request.reason == ReadReason.RUNTIME_STATE_TRANSITION:
            return call_related or import_related or any(
                term in target for term in ("step", "transition", "state", "update")
            )
        if request.reason == ReadReason.UNKNOWN_SYMBOL:
            return request.target_symbol is not None and (call_related or import_related)
        return call_related or import_related

    def read(
        self,
        request: EvidenceReadRequest,
        *,
        max_characters: int = 9000,
        max_related_symbols: int = 3,
    ) -> ToolEvidence:
        path = (self.root / request.target_file).resolve()
        if (
            request.target_file not in self.allowed_files
            or not path.is_relative_to(self.root)
            or not path.is_file()
        ):
            raise ValueError("target is outside the controlled project index")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        symbol = self.symbol(request.target_file, request.target_symbol)
        if request.target_symbol and symbol is None:
            raise ValueError("requested symbol does not exist")
        related: list[PythonSymbol] = []
        if symbol:
            indexes = range(symbol.start_line - 1, min(symbol.end_line, len(lines)))
            for called in symbol.calls:
                candidates = self.by_terminal_name.get(called.rsplit(".", 1)[-1], [])
                for candidate in candidates:
                    if candidate.file == request.target_file and candidate != symbol:
                        related.append(candidate)
                        break
                if len(related) >= max_related_symbols:
                    break
        else:
            indexes = range(min(len(lines), 220))
        chosen = set(indexes)
        for item in related:
            chosen.update(range(item.start_line - 1, min(item.end_line, len(lines))))
        excerpt = "\n".join(
            f"{index + 1:04d}: {lines[index]}" for index in sorted(chosen)
        )[:max_characters]
        return ToolEvidence(
            request_id=request.id,
            path=request.target_file,
            symbol=request.target_symbol,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            excerpt=excerpt,
            related_symbols=tuple(item.qualname for item in related),
        )


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class ControlledReadGate:
    def __init__(
        self,
        index: ProjectCodeIndex,
        *,
        max_total_reads: int = 18,
        max_reads_per_round: int = 4,
    ) -> None:
        self.index = index
        self.max_total_reads = max_total_reads
        self.max_reads_per_round = max_reads_per_round
        self.used: set[tuple[str, str | None]] = set()
        self.total_reads = 0

    def decide(
        self, analysis: BatchAnalysis
    ) -> tuple[tuple[ReadDecision, ...], tuple[EvidenceReadRequest, ...]]:
        assessments = {
            request_id: assessment
            for assessment in analysis.assessments
            for request_id in assessment.requested_read_ids
        }
        decisions: list[ReadDecision] = []
        approved: list[EvidenceReadRequest] = []
        for request in analysis.read_requests:
            assessment = assessments.get(request.id)
            reason_code = "approved"
            detail = "request passed deterministic evidence and relation gates"
            if assessment is None:
                reason_code, detail = "unlinked_assessment", "request has no assessment"
            elif assessment.sufficiency in {
                EvidenceSufficiency.SUFFICIENT,
                EvidenceSufficiency.NOT_APPLICABLE,
            }:
                reason_code, detail = (
                    "evidence_already_sufficient",
                    "assessment says no additional evidence is needed",
                )
            elif request.target_file not in self.index.allowed_files:
                reason_code, detail = "path_not_allowed", "target is not indexed"
            elif _is_paper_like(Path(request.target_file)) or Path(
                request.target_file
            ).name in _SECRET_NAMES:
                reason_code, detail = "paper_or_secret", "target type is not readable"
            elif (request.target_file, request.target_symbol) in self.used:
                reason_code, detail = "duplicate", "target was already read"
            elif request.target_symbol and self.index.symbol(
                request.target_file, request.target_symbol
            ) is None:
                reason_code, detail = "invalid_symbol", "target symbol is absent"
            elif not self.index.related(request):
                reason_code, detail = (
                    "unrelated_target",
                    "no call/import/config/validator relation was found",
                )
            elif not self.index.reason_is_plausible(request):
                reason_code, detail = (
                    "reason_mismatch",
                    "selected reason is not supported by target semantics or code relation",
                )
            elif request.priority == ReadPriority.LOW:
                reason_code, detail = "low_priority", "low-priority reads are deferred"
            elif (
                self.total_reads + len(approved) >= self.max_total_reads
                or len(approved) >= self.max_reads_per_round
            ):
                reason_code, detail = "budget_exhausted", "read budget is exhausted"
            if reason_code == "approved":
                approved.append(request)
                self.used.add((request.target_file, request.target_symbol))
            decisions.append(
                ReadDecision(
                    request_id=request.id,
                    approved=reason_code == "approved",
                    reason_code=reason_code,
                    detail=detail,
                )
            )
        self.total_reads += len(approved)
        return tuple(decisions), tuple(approved)


class ShortTermMemory(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    confirmed_facts: tuple[BatchFact, ...] = ()
    contradictions: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    batch_summaries: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class LongTermMemoryContext(FrozenModel):
    """Retrieved prior knowledge; never accepted as current-project evidence."""

    query: str
    node_ids: tuple[str, ...] = ()
    chunks: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


def retrieve_long_term_context(
    database: str | Path | None,
    query: str,
    *,
    limit: int = 8,
) -> LongTermMemoryContext | None:
    if database is None:
        return None
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        return None
    result = HierarchicalMemory.open(path).retrieve(query, limit=limit)
    return LongTermMemoryContext(
        query=query,
        node_ids=tuple(item.canonical_id for item in result.nodes),
        chunks=tuple(
            f"{item.chunk.chunk_id} [L{item.chunk.layer}] "
            f"{item.chunk.title}: {item.chunk.content[:1200]}"
            for item in result.evidence
        ),
        conflicts=result.conflicts,
    )


def update_short_term_memory(
    memory: ShortTermMemory,
    batch: BatchAnalysis,
    evidence: Iterable[ToolEvidence] = (),
    *,
    max_facts: int = 80,
    max_items: int = 80,
) -> ShortTermMemory:
    facts = {item.id: item for item in memory.confirmed_facts}
    facts.update({item.id: item for item in batch.facts})
    return ShortTermMemory(
        confirmed_facts=tuple(facts.values())[-max_facts:],
        contradictions=tuple(
            dict.fromkeys((*memory.contradictions, *batch.contradictions))
        )[-max_items:],
        unresolved=tuple(dict.fromkeys((*memory.unresolved, *batch.unresolved)))[
            -max_items:
        ],
        batch_summaries=(*memory.batch_summaries, batch.compressed_summary)[-12:],
        evidence_ids=tuple(
            dict.fromkeys(
                (*memory.evidence_ids, *(item.request_id for item in evidence))
            )
        )[-max_items:],
    )


class SemanticBatchSpec(FrozenModel):
    id: Literal[
        "project_and_environment",
        "objectives_and_constraints",
        "decisions_oracles_and_unknowns",
    ]
    questions: tuple[str, ...]
    file_roles: tuple[FileRole, ...]


BATCH_SPECS = (
    SemanticBatchSpec(
        id="project_and_environment",
        questions=(
            "project_type_and_problem_family",
            "environment_and_runtime_mode",
            "data_and_instance_semantics",
        ),
        file_roles=(
            FileRole.ENTRYPOINT,
            FileRole.PROJECT_DOCUMENT,
            FileRole.CONFIG,
            FileRole.DATA_SCHEMA,
        ),
    ),
    SemanticBatchSpec(
        id="objectives_and_constraints",
        questions=(
            "optimization_objectives",
            "hard_constraints",
            "constraint_implementation_paths",
        ),
        file_roles=(
            FileRole.SOURCE,
            FileRole.SOLVER,
            FileRole.VALIDATOR,
            FileRole.TEST,
        ),
    ),
    SemanticBatchSpec(
        id="decisions_oracles_and_unknowns",
        questions=(
            "modifiable_decisions",
            "validators_solvers_and_oracles",
            "unresolved_or_conflicting_semantics",
        ),
        file_roles=(
            FileRole.ENTRYPOINT,
            FileRole.SOLVER,
            FileRole.VALIDATOR,
            FileRole.TEST,
            FileRole.CONFIG,
        ),
    ),
)


_BATCH_SYSTEM = """\
你是高能力“项目语义分析 Agent”，但每次只处理一个任务批次。
你收到项目导航、压缩短期记忆和本轮证据。先回答每个问题的证据充分性选择题，
再给出事实或受控复读请求。

严格规则：
1. sufficiency 只能选 sufficient/partial/insufficient/contradictory/not_applicable。
2. 事实 evidence 使用 path::symbol 或 path::line-range，必须在本轮材料中出现。
3. 只有 partial、insufficient 或 contradictory 可以申请 read_request。
4. 复读请求只能指向项目内具体文件/符号，必须说明预期找到什么；模型无权直接读。
5. 不得要求读取 .env、秘密、二进制、论文或项目外路径。
6. 不重复请求已经出现在 tool evidence 或短期记忆 evidence_ids 中的内容。
7. compressed_summary 只保留后续批次需要的事实、证据 ID、矛盾和未知，不保留推理过程。
8. 返回一个 JSON 对象，不要 Markdown 或额外解释。
"""


def _select_initial_files(
    inventory: NavigationInventory,
    navigation: ProjectNavigation,
    spec: SemanticBatchSpec,
    *,
    limit: int = 18,
) -> tuple[InventoryFile, ...]:
    priorities = {path: index for index, path in enumerate(navigation.priority_reading_paths)}
    selected = [
        item
        for item in inventory.files
        if item.role_hint in set(spec.file_roles)
        or item.path in priorities
    ]
    selected.sort(
        key=lambda item: (
            priorities.get(item.path, 10_000),
            0 if item.role_hint in spec.file_roles else 1,
            item.path,
        )
    )
    return tuple(selected[:limit])


def _batch_prompt(
    spec: SemanticBatchSpec,
    navigation: ProjectNavigation,
    memory: ShortTermMemory,
    files: tuple[InventoryFile, ...],
    tool_evidence: tuple[ToolEvidence, ...],
    long_term_context: LongTermMemoryContext | None,
    *,
    round_index: int,
) -> str:
    return (
        f"批次：{spec.id}；轮次：{round_index}\n"
        "问题清单：\n"
        + json.dumps(spec.questions, ensure_ascii=False, indent=2)
        + "\n\n选项库：\n"
        + json.dumps(
            {
                "sufficiency": [item.value for item in EvidenceSufficiency],
                "read_reason": [item.value for item in ReadReason],
                "read_priority": [item.value for item in ReadPriority],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n输出 JSON Schema：\n"
        + json.dumps(BatchAnalysis.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n\n项目导航：\n"
        + navigation.model_dump_json(indent=2)
        + "\n\n压缩短期记忆（先验，不替代本批证据）：\n"
        + memory.model_dump_json(indent=2)
        + "\n\n图谱/分层长期记忆检索（先验，不可作为当前项目 evidence）：\n"
        + (
            long_term_context.model_dump_json(indent=2)
            if long_term_context is not None
            else '{"status":"not_configured"}'
        )
        + "\n\n初始文件证据：\n"
        + json.dumps(
            [item.model_dump(mode="json") for item in files],
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n本批受控工具新增证据：\n"
        + json.dumps(
            [item.model_dump(mode="json") for item in tool_evidence],
            ensure_ascii=False,
            indent=2,
        )
    )


class BatchRound(FrozenModel):
    round_index: int = Field(ge=0)
    analysis: BatchAnalysis
    read_decisions: tuple[ReadDecision, ...] = ()
    tool_evidence: tuple[ToolEvidence, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchRun(FrozenModel):
    batch_id: str
    rounds: tuple[BatchRound, ...] = Field(min_length=1)


class StagedCompilationMetadata(FrozenModel):
    navigator_provider: str
    navigator_model: str
    analyst_provider: str
    analyst_model: str
    navigator_calls: int = Field(ge=1)
    analyst_calls: int = Field(ge=1)
    approved_reads: int = Field(ge=0)
    rejected_reads: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class StagedSemanticCompilation(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    review_status: Literal["pending_human_review"] = "pending_human_review"
    inventory: NavigationInventory
    navigation: ProjectNavigation
    batches: tuple[BatchRun, ...]
    short_term_memory: ShortTermMemory
    long_term_memory_contexts: tuple[LongTermMemoryContext, ...] = ()
    final: LLMSemanticCompilation
    metadata: StagedCompilationMetadata


def _packet_from_ledger(
    root: Path,
    inventory_files: Iterable[InventoryFile],
    tool_evidence: Iterable[ToolEvidence],
) -> RepositoryEvidencePacket:
    excerpts: dict[str, list[str]] = defaultdict(list)
    for item in inventory_files:
        excerpts[item.path].append(item.excerpt)
    for item in tool_evidence:
        excerpts[item.path].append(item.excerpt)
    files: list[FileEvidence] = []
    total = 0
    for path_name in sorted(excerpts):
        path = root / path_name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        excerpt = "\n".join(dict.fromkeys(excerpts[path_name]))[:16000]
        total += len(excerpt)
        files.append(
            FileEvidence(
                path=path_name,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                excerpt=excerpt,
            )
        )
    normalized = json.dumps(
        [item.model_dump(mode="json") for item in files],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return RepositoryEvidencePacket(
        project_name=root.name,
        files=tuple(files),
        character_count=total,
        packet_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def compile_project_semantics_staged(
    root: str | Path,
    *,
    navigator_provider: ModelProvider,
    analyst_provider: ModelProvider,
    max_rounds_per_batch: int = 3,
    max_reads: int = 18,
    max_output_tokens: int = 7000,
    max_schema_repairs: int = 1,
    knowledge_base_path: str | Path | None = None,
    impact_provider: ModelProvider | None = None,
    impact_max_output_tokens: int = 12000,
    memory_database: str | Path | None = None,
) -> StagedSemanticCompilation:
    """Run the navigator, batched analyst, controlled reader and final synthesis."""

    base = Path(root).expanduser().resolve()
    inventory, navigation, navigator_responses = build_project_navigation_with_llm(
        base, provider=navigator_provider
    )
    index = ProjectCodeIndex(base)
    gate = ControlledReadGate(index, max_total_reads=max_reads)
    memory = ShortTermMemory()
    batch_runs: list[BatchRun] = []
    all_tool_evidence: list[ToolEvidence] = []
    all_initial_files: dict[str, InventoryFile] = {}
    analyst_responses: list[ModelResponse] = []
    approved_count = 0
    rejected_count = 0
    long_term_contexts: list[LongTermMemoryContext] = []

    for spec in BATCH_SPECS:
        files = _select_initial_files(inventory, navigation, spec)
        all_initial_files.update({item.path: item for item in files})
        new_evidence: tuple[ToolEvidence, ...] = ()
        rounds: list[BatchRound] = []
        long_term = retrieve_long_term_context(
            memory_database,
            f"{navigation.project_summary} {' '.join(spec.questions)}",
        )
        if long_term is not None:
            long_term_contexts.append(long_term)
        for round_index in range(max_rounds_per_batch):
            response = analyst_provider.complete(
                ModelRequest(
                    system=_BATCH_SYSTEM,
                    user=_batch_prompt(
                        spec,
                        navigation,
                        memory,
                        files,
                        new_evidence,
                        long_term,
                        round_index=round_index,
                    ),
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    require_json=True,
                    thinking_mode="enabled",
                    reasoning_effort="max",
                    metadata={
                        "task": "staged_semantic_batch",
                        "batch": spec.id,
                        "round": round_index,
                    },
                )
            )
            analyst_responses.append(response)
            analysis = BatchAnalysis.model_validate(_extract_object(response.content))
            if analysis.batch_id != spec.id:
                raise ValueError(
                    f"batch response id {analysis.batch_id!r} does not match {spec.id!r}"
                )
            decisions, approved = gate.decide(analysis)
            approved_count += len(approved)
            rejected_count += sum(not item.approved for item in decisions)
            evidence = tuple(index.read(request) for request in approved)
            all_tool_evidence.extend(evidence)
            rounds.append(
                BatchRound(
                    round_index=round_index,
                    analysis=analysis,
                    read_decisions=decisions,
                    tool_evidence=evidence,
                    metadata={
                        "provider": response.provider,
                        "model": response.response_model or response.requested_model,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                )
            )
            memory = update_short_term_memory(memory, analysis, evidence)
            new_evidence = evidence
            # Any approved read must be shown to the analyst in a later round;
            # ``ready_for_synthesis`` cannot bypass evaluation of new evidence.
            if not approved:
                break
        batch_runs.append(BatchRun(batch_id=spec.id, rounds=tuple(rounds)))

    packet = _packet_from_ledger(
        base, all_initial_files.values(), all_tool_evidence
    )
    if not packet.files:
        raise ValueError("staged evidence packet is empty")
    knowledge = SchedulingKnowledgeBase.load(knowledge_base_path)
    packet_text = "\n".join(f"{item.path}\n{item.excerpt}" for item in packet.files)
    initial_hits = knowledge.retrieve(packet_text, top_k=3)
    engineering_hits = knowledge.retrieve_engineering_patterns(packet_text, top_k=8)
    synthesis_response = analyst_provider.complete(
        ModelRequest(
            system=_SYSTEM_PROMPT
            + "\n本次输入还包括经程序批准的分批短期记忆。记忆只用于导航；"
            "最终判断仍必须引用最终证据包。",
            user=(
                _analysis_prompt(packet, initial_hits, engineering_hits)
                + "\n\n分批压缩短期记忆：\n"
                + memory.model_dump_json(indent=2)
                + "\n\n项目导航（只用于定位，不可作为 evidence）：\n"
                + navigation.model_dump_json(indent=2)
            ),
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            require_json=True,
            thinking_mode="enabled",
            reasoning_effort="max",
            metadata={"task": "staged_semantic_final_synthesis"},
        )
    )
    analyst_responses.append(synthesis_response)
    schema_repairs = 0
    while True:
        try:
            semantic_analysis = SemanticAnalysis.model_validate(
                extract_json_object(analyst_responses[-1].content)
            )
            break
        except (ValueError, ValidationError) as error:
            if schema_repairs >= max_schema_repairs:
                raise ValueError(
                    "staged final synthesis failed schema validation after "
                    f"{schema_repairs} repair(s): {error}"
                ) from error
            schema_repairs += 1
            repair = analyst_provider.complete(
                ModelRequest(
                    system=_SYSTEM_PROMPT,
                    user=_repair_prompt(analyst_responses[-1].content, error),
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    require_json=True,
                    thinking_mode="enabled",
                    reasoning_effort="max",
                    metadata={
                        "task": "staged_semantic_schema_repair",
                        "repair": schema_repairs,
                    },
                )
            )
            analyst_responses.append(repair)

    final_hits = knowledge.retrieve(
        packet_text,
        family_hints=tuple(item.value for item in semantic_analysis.problem_families),
        top_k=4,
    )
    final_engineering_hits = knowledge.retrieve_engineering_patterns(
        packet_text,
        family_hints=tuple(item.value for item in semantic_analysis.problem_families),
        top_k=8,
    )
    impact = (
        assess_constraint_impacts_with_llm(
            semantic_analysis,
            final_hits,
            provider=impact_provider,
            engineering_hits=final_engineering_hits,
            max_output_tokens=impact_max_output_tokens,
            thinking_mode="enabled",
            reasoning_effort="max",
        )
        if impact_provider is not None
        else None
    )
    final = LLMSemanticCompilation(
        packet=packet,
        knowledge_hits=final_hits,
        engineering_pattern_hits=final_engineering_hits,
        analysis=semantic_analysis,
        constraint_impact=impact,
        evidence_checks=audit_llm_evidence(base, packet, semantic_analysis),
        metadata=_aggregate_metadata(
            analyst_responses[-(schema_repairs + 1) :],
            schema_repairs=schema_repairs,
        ),
    )
    responses = [*navigator_responses, *analyst_responses]
    usage = TokenUsage(
        input_tokens=sum(item.usage.input_tokens for item in responses),
        output_tokens=sum(item.usage.output_tokens for item in responses),
        total_tokens=sum(item.usage.total_tokens for item in responses),
    )
    return StagedSemanticCompilation(
        inventory=inventory,
        navigation=navigation,
        batches=tuple(batch_runs),
        short_term_memory=memory,
        long_term_memory_contexts=tuple(long_term_contexts),
        final=final,
        metadata=StagedCompilationMetadata(
            navigator_provider=navigator_responses[-1].provider,
            navigator_model=(
                navigator_responses[-1].response_model
                or navigator_responses[-1].requested_model
            ),
            analyst_provider=analyst_responses[-1].provider,
            analyst_model=(
                analyst_responses[-1].response_model
                or analyst_responses[-1].requested_model
            ),
            navigator_calls=len(navigator_responses),
            analyst_calls=len(analyst_responses),
            approved_reads=approved_count,
            rejected_reads=rejected_count,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        ),
    )
