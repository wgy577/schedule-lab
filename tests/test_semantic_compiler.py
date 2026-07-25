from pathlib import Path

from causal_schedule_lab.models import ClaimStatus
from causal_schedule_lab.semantic_compiler import compile_project_semantics


ROOT = Path(__file__).resolve().parents[1]


def test_compiler_keeps_evidence_and_unknowns_separate() -> None:
    compilation = compile_project_semantics(ROOT)
    assert compilation.symbols
    assert any(claim.status == ClaimStatus.VERIFIED for claim in compilation.claims)
    for claim in compilation.claims:
        if claim.status == ClaimStatus.VERIFIED:
            assert claim.evidence
            assert all((ROOT / evidence.file).exists() for evidence in claim.evidence)
