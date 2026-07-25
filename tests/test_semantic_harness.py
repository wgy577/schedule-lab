import json
from pathlib import Path

from pypdf import PdfWriter

from causal_schedule_lab.providers.base import ModelRequest, ModelResponse, TokenUsage
from causal_schedule_lab.semantic_harness import (
    _extract_generic_json_object,
    repository_inventory,
    run_blind_harness,
    run_code_only_harness,
)


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, contents: list[str]):
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
                output_tokens=5,
                total_tokens=15,
            ),
        )


def analysis_payload(evidence_file: str) -> dict:
    evidence = [
        {
            "file": evidence_file,
            "symbol": None,
            "detail": "该证据说明这是最小化 makespan 的 JSP。",
        }
    ]
    return {
        "schema_version": "1.0",
        "language": "zh-CN",
        "summary": "使用学习策略解决确定性离线作业车间调度。",
        "project_type": "scheduling_optimization",
        "problem_families": ["JSP"],
        "environments": [
            {
                "kind": "deterministic",
                "statement": "处理时间固定。",
                "evidence": evidence,
                "confidence": "high",
            },
            {
                "kind": "offline",
                "statement": "实例在调度开始前给定。",
                "evidence": evidence,
                "confidence": "high",
            },
        ],
        "objectives": [
            {
                "id": "objective_1",
                "kind": "makespan",
                "sense": "minimize",
                "priority": 1,
                "statement": "最小化最大完工时间。",
                "evidence": evidence,
                "confidence": "high",
            }
        ],
        "constraints": [
            {
                "id": "constraint_1",
                "kind": "precedence",
                "scope": "operation",
                "hard": True,
                "statement": "同一作业的工序必须满足前置关系。",
                "evidence": evidence,
                "confidence": "high",
            }
        ],
        "decisions": [
            {
                "id": "decision_1",
                "kind": "sequence",
                "modifiable": True,
                "statement": "策略选择可调度工序以形成机器顺序。",
                "evidence": evidence,
                "confidence": "high",
            }
        ],
        "oracles": [],
        "allowed_interventions": ["sequence"],
        "unknowns": [],
        "overall_confidence": "high",
    }


def test_inventory_and_blind_harness_exclude_paper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "code.py").write_text(
        "def makespan(schedule):\n    return max(schedule)\n",
        encoding="utf-8",
    )
    (repository / "paper").mkdir()
    (repository / "paper" / "leak.md").write_text(
        "hidden ground truth", encoding="utf-8"
    )
    paper = tmp_path / "paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with paper.open("wb") as handle:
        writer.write(handle)
    manifest = tmp_path / "case.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "case_id": "test-case",
                "title": "Test",
                "paper_url": "https://example.test/paper.pdf",
                "repository_url": "https://example.test/repo.git",
                "repository_ref": "abc123",
                "paper_path": "paper.pdf",
                "repository_path": "repository",
                "excluded_code_paths": ["paper"],
                "label_status": "draft_llm_label",
                "notes": "blind test",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "causal_schedule_lab.semantic_harness.extract_paper_text",
        lambda *_args, **_kwargs: "paper-only ground truth",
    )
    read_plan = {
        "schema_version": "1.0",
        "focus_questions": ["problem_family", "objective", "constraints"],
        "selected_files": ["code.py"],
        "rationale": "实现文件包含调度目标和核心逻辑。",
        "expected_gaps": [],
        "paper_used": False,
    }
    provider = FakeProvider(
        [
            json.dumps(analysis_payload("paper.pdf"), ensure_ascii=False),
            "```json\n" + json.dumps(read_plan, ensure_ascii=False) + "\n```",
            json.dumps(analysis_payload("code.py"), ensure_ascii=False),
        ]
    )

    inventory = repository_inventory(repository, excluded_prefixes=("paper",))
    assert [item.path for item in inventory] == ["code.py"]

    result = run_blind_harness(manifest, provider=provider)
    assert result.evaluation.blind is True
    assert result.evaluation.macro_f1 == 1.0
    assert result.label.evidence_pass_rate == 1.0
    assert result.prediction.evidence_pass_rate == 1.0
    assert result.planner_metadata["calls"] == 1
    assert "paper-only ground truth" not in provider.requests[1].user
    assert "paper-only ground truth" not in provider.requests[2].user
    assert "hidden ground truth" not in provider.requests[2].user

    code_provider = FakeProvider(
        [
            json.dumps(read_plan, ensure_ascii=False),
            json.dumps(analysis_payload("code.py"), ensure_ascii=False),
        ]
    )
    code_result = run_code_only_harness(
        manifest,
        label=result.label,
        provider=code_provider,
    )
    assert code_result.evaluation.macro_f1 == 1.0
    assert len(code_provider.requests) == 2
    assert all(
        "paper-only ground truth" not in request.user
        for request in code_provider.requests
    )


def test_generic_json_extractor_accepts_fenced_json() -> None:
    assert _extract_generic_json_object("```json\n{\"ok\": true}\n```") == {
        "ok": True
    }
