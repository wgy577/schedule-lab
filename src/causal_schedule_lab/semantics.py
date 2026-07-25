from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ClaimStatus, FailureLabel, ProjectSemantics


class SemanticAuditError(RuntimeError):
    pass


def load_semantics(path: str | Path) -> ProjectSemantics:
    return ProjectSemantics.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_semantics(semantics: ProjectSemantics, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(semantics.model_dump_json(indent=2) + "\n", encoding="utf-8")


def audit_semantic_evidence(
    semantics: ProjectSemantics,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Verify evidence mechanically; language-model claims never self-certify."""

    root = Path(project_root).expanduser().resolve()
    claims: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for claim in semantics.claims:
        evidence_results = []
        for evidence in claim.evidence:
            source = Path(evidence.file).expanduser()
            if not source.is_absolute():
                source = root / source
            exists = source.exists()
            symbol_found = None
            if exists and evidence.symbol:
                try:
                    symbol_found = evidence.symbol in source.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                except OSError:
                    symbol_found = False
            passed = exists and (symbol_found is not False)
            result = {
                "kind": evidence.kind,
                "file": str(source),
                "symbol": evidence.symbol,
                "exists": exists,
                "symbolFound": symbol_found,
                "passed": passed,
            }
            evidence_results.append(result)
        mechanically_verified = bool(evidence_results) and all(
            result["passed"] for result in evidence_results
        )
        claim_result = {
            "id": claim.id,
            "declaredStatus": claim.status.value,
            "mechanicallyVerified": mechanically_verified,
            "evidence": evidence_results,
        }
        claims.append(claim_result)
        if claim.status == ClaimStatus.VERIFIED and not mechanically_verified:
            failures.append(
                {
                    "claim": claim.id,
                    "label": FailureLabel.SEMANTIC_EVIDENCE_MISSING.value,
                }
            )
    return {
        "projectId": semantics.project_id,
        "projectRoot": str(root),
        "passed": not failures,
        "claims": claims,
        "failures": failures,
    }


def require_semantic_evidence(
    semantics: ProjectSemantics,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    audit = audit_semantic_evidence(semantics, project_root=project_root)
    if not audit["passed"]:
        failed = ", ".join(item["claim"] for item in audit["failures"])
        raise SemanticAuditError(f"verified semantic claims lack evidence: {failed}")
    return audit


def semantics_digest(semantics: ProjectSemantics) -> str:
    import hashlib

    payload = json.dumps(
        semantics.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
