"""Evidence-grounded project semantic compiler.

The compiler deliberately separates *proposal* from *verification*.  An LLM
may propose a semantic claim, but only code, test, document, or witness
evidence can move it to ``verified``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import ClaimStatus, EvidenceRef, SemanticClaim


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class RepositorySymbol(FrozenModel):
    file: str
    symbol: str
    kind: str
    line: int
    digest: str
    tags: tuple[str, ...] = ()


class SemanticProposal(FrozenModel):
    id: str
    claim: str
    query_terms: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "rule"


class SemanticCompilation(FrozenModel):
    root: str
    symbols: tuple[RepositorySymbol, ...]
    claims: tuple[SemanticClaim, ...]
    unknowns: tuple[str, ...]
    ignored_files: tuple[str, ...] = ()


class ProposalProvider(Protocol):
    def propose(
        self,
        *,
        symbols: tuple[RepositorySymbol, ...],
        documents: tuple[str, ...],
    ) -> Iterable[SemanticProposal]: ...


_TAGS = {
    "objective": ("objective", "makespan", "tardiness", "cost", "fitness"),
    "constraint": ("constraint", "validate", "feasible", "overlap", "precedence"),
    "resource": ("machine", "resource", "capacity", "vehicle", "worker"),
    "solver": ("solve", "solver", "cp_sat", "dispatch", "optimize", "search"),
    "oracle": ("oracle", "collision", "simulation", "replay", "validator"),
    "schedule": ("schedule", "assignment", "operation", "job", "task"),
}


def _symbol_tags(name: str, source: str) -> tuple[str, ...]:
    haystack = f"{name} {source[:1200]}".lower()
    return tuple(
        tag
        for tag, terms in _TAGS.items()
        if any(term in haystack for term in terms)
    )


def index_repository(
    root: str | Path,
    *,
    suffixes: tuple[str, ...] = (".py", ".md", ".json", ".yaml", ".yml"),
) -> tuple[RepositorySymbol, ...]:
    base = Path(root).expanduser().resolve()
    symbols: list[RepositorySymbol] = []
    for path in sorted(base.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in suffixes
            or any(part.startswith(".") for part in path.relative_to(base).parts)
            or any(part in {"build", "dist", "__pycache__"} for part in path.parts)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(base).as_posix()
        if path.suffix == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    segment = ast.get_source_segment(text, node) or node.name
                    symbols.append(
                        RepositorySymbol(
                            file=relative,
                            symbol=node.name,
                            kind=type(node).__name__,
                            line=node.lineno,
                            digest=hashlib.sha256(segment.encode()).hexdigest()[:16],
                            tags=_symbol_tags(node.name, segment),
                        )
                    )
        else:
            symbols.append(
                RepositorySymbol(
                    file=relative,
                    symbol=path.stem,
                    kind="document",
                    line=1,
                    digest=hashlib.sha256(text.encode()).hexdigest()[:16],
                    tags=_symbol_tags(path.stem, text),
                )
            )
    return tuple(symbols)


def verify_proposals(
    root: str | Path,
    symbols: tuple[RepositorySymbol, ...],
    proposals: Iterable[SemanticProposal],
) -> tuple[SemanticClaim, ...]:
    base = Path(root).expanduser().resolve()
    claims = []
    for proposal in proposals:
        matches = [
            symbol
            for symbol in symbols
            if all(
                term.lower()
                in f"{symbol.symbol} {' '.join(symbol.tags)}".lower()
                for term in proposal.query_terms
            )
        ]
        evidence = tuple(
            EvidenceRef(
                kind="code" if item.kind != "document" else "document",
                file=item.file,
                symbol=item.symbol,
                detail=f"line={item.line}; sha256={item.digest}",
            )
            for item in matches
            if (base / item.file).exists()
        )
        claims.append(
            SemanticClaim(
                id=proposal.id,
                claim=proposal.claim,
                status=ClaimStatus.VERIFIED if evidence else ClaimStatus.UNKNOWN,
                confidence=proposal.confidence if evidence else min(0.49, proposal.confidence),
                evidence=evidence,
            )
        )
    return tuple(claims)


def default_proposals(symbols: tuple[RepositorySymbol, ...]) -> tuple[SemanticProposal, ...]:
    present_tags = {tag for symbol in symbols for tag in symbol.tags}
    return tuple(
        SemanticProposal(
            id=f"repository-{tag}",
            claim=f"The repository contains executable evidence for {tag} semantics.",
            query_terms=(tag,),
            confidence=0.8,
        )
        for tag in sorted(present_tags)
    )


def compile_project_semantics(
    root: str | Path,
    *,
    provider: ProposalProvider | None = None,
) -> SemanticCompilation:
    base = Path(root).expanduser().resolve()
    symbols = index_repository(base)
    documents = tuple(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(base.rglob("*.md"))
        if ".venv" not in path.parts
    )
    proposals = (
        tuple(provider.propose(symbols=symbols, documents=documents))
        if provider is not None
        else default_proposals(symbols)
    )
    claims = verify_proposals(base, symbols, proposals)
    return SemanticCompilation(
        root=str(base),
        symbols=symbols,
        claims=claims,
        unknowns=tuple(
            item.claim for item in claims if item.status == ClaimStatus.UNKNOWN
        ),
    )
