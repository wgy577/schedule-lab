import json
from pathlib import Path

from causal_schedule_lab.llm_semantics import (
    build_repository_evidence_packet,
    compile_project_semantics_with_llm,
    extract_json_object,
)
from causal_schedule_lab.providers.base import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
)


def valid_analysis() -> dict:
    evidence = [
        {
            "file": "README.md",
            "symbol": None,
            "detail": "README describes the scheduling laboratory.",
        }
    ]
    return {
        "schema_version": "1.0",
        "language": "zh-CN",
        "summary": "这是一个调度优化研究平台。",
        "project_type": "scheduling_optimization",
        "problem_families": ["JSP", "FSP", "FJSP", "HFSP"],
        "environments": [
            {
                "kind": "offline",
                "statement": "当前系统执行离线调度改进。",
                "evidence": evidence,
                "confidence": "high",
            }
        ],
        "objectives": [
            {
                "id": "objective_1",
                "kind": "makespan",
                "sense": "minimize",
                "priority": 1,
                "statement": "主要目标包含缩短完工时间。",
                "evidence": evidence,
                "confidence": "medium",
            }
        ],
        "constraints": [
            {
                "id": "constraint_1",
                "kind": "precedence",
                "scope": "operation",
                "hard": True,
                "statement": "工序必须满足前置关系。",
                "evidence": evidence,
                "confidence": "high",
            }
        ],
        "decisions": [
            {
                "id": "decision_1",
                "kind": "sequence",
                "modifiable": True,
                "statement": "允许局部调整工序顺序。",
                "evidence": evidence,
                "confidence": "high",
            }
        ],
        "oracles": [
            {
                "id": "oracle_1",
                "kind": "static_validator",
                "required": True,
                "statement": "候选需要约束验证。",
                "evidence": evidence,
                "confidence": "medium",
            }
        ],
        "allowed_interventions": ["sequence"],
        "unknowns": [],
        "overall_confidence": "medium",
    }


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, contents):
        self.contents = list(contents)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=self.contents.pop(0),
            provider=self.name,
            requested_model=self.model,
            response_model=self.model,
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
        )


def test_packet_excludes_env_and_llm_output_is_evidence_audited(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "# Scheduling project\n", encoding="utf-8"
    )
    (tmp_path / ".env").write_text("SECRET=do-not-send\n", encoding="utf-8")
    packet = build_repository_evidence_packet(tmp_path)
    assert [item.path for item in packet.files] == ["README.md"]
    assert "do-not-send" not in packet.model_dump_json()

    provider = FakeProvider([json.dumps(valid_analysis(), ensure_ascii=False)])
    result = compile_project_semantics_with_llm(
        tmp_path,
        provider=provider,
        max_input_characters=5000,
    )
    assert result.review_status == "pending_human_review"
    assert result.evidence_pass_rate == 1.0
    assert result.analysis.problem_families[0].value == "JSP"
    assert "JSP" in {item.family for item in result.knowledge_hits}
    assert len(provider.requests) == 1


def test_family_prior_is_retrieved_but_not_accepted_as_project_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "# Classical JSSP\nA static job shop implementation.\n",
        encoding="utf-8",
    )
    provider = FakeProvider([json.dumps(valid_analysis(), ensure_ascii=False)])
    result = compile_project_semantics_with_llm(tmp_path, provider=provider)
    assert result.knowledge_hits[0].family == "JSP"
    assert "检索到的经典调度知识先验" in provider.requests[0].user
    assert "不能作为当前项目证据" in provider.requests[0].system
    assert all(
        citation.file == "README.md"
        for finding in result.analysis.constraints
        for citation in finding.evidence
    )


def test_invalid_choice_gets_one_schema_repair(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    invalid = valid_analysis()
    invalid["project_type"] = "invented-project-type"
    provider = FakeProvider(
        [
            json.dumps(invalid, ensure_ascii=False),
            "```json\n"
            + json.dumps(valid_analysis(), ensure_ascii=False)
            + "\n```",
        ]
    )
    result = compile_project_semantics_with_llm(
        tmp_path,
        provider=provider,
        max_input_characters=5000,
    )
    assert result.metadata.schema_repairs == 1
    assert len(provider.requests) == 2


def test_json_extractor_handles_surrounding_text() -> None:
    payload = valid_analysis()
    assert extract_json_object(
        "reasoning\\n" + json.dumps(payload, ensure_ascii=False) + "\\nend"
    ) == payload


def test_json_extractor_rejects_nested_finding() -> None:
    nested = valid_analysis()["environments"][0]
    try:
        extract_json_object(json.dumps(nested))
    except ValueError as error:
        assert "complete SemanticAnalysis root" in str(error)
    else:
        raise AssertionError("nested finding must not be accepted as root")


def test_evidence_audit_accepts_qualified_class_method(tmp_path: Path) -> None:
    (tmp_path / "engine.py").write_text(
        "class Engine:\n    def step(self):\n        return 1\n",
        encoding="utf-8",
    )
    payload = valid_analysis()
    for collection in (
        payload["environments"],
        payload["objectives"],
        payload["constraints"],
        payload["decisions"],
        payload["oracles"],
    ):
        for item in collection:
            item["evidence"] = [
                {
                    "file": "engine.py",
                    "symbol": "Engine.step",
                    "detail": "Engine.step contains the executable behavior.",
                }
            ]
    provider = FakeProvider([json.dumps(payload, ensure_ascii=False)])
    result = compile_project_semantics_with_llm(tmp_path, provider=provider)
    assert result.evidence_pass_rate == 1.0
