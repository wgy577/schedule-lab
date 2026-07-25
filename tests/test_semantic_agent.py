import json
from pathlib import Path

from causal_schedule_lab.providers.base import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from causal_schedule_lab.semantic_agent import (
    BatchAnalysis,
    ControlledReadGate,
    EvidenceReadRequest,
    ProjectCodeIndex,
    build_navigation_inventory,
    compile_project_semantics_staged,
)


class FakeProvider:
    def __init__(self, name: str, contents: list[dict]) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.contents = [json.dumps(item, ensure_ascii=False) for item in contents]
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=self.contents.pop(0),
            provider=self.name,
            requested_model=self.model,
            response_model=self.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def navigation() -> dict:
    return {
        "schema_version": "1.0",
        "language": "zh-CN",
        "project_summary": "一个最小调度项目。",
        "modules": [
            {
                "id": "module_1",
                "name": "入口",
                "purpose": "运行调度与验证。",
                "files": ["main.py"],
                "depends_on": [],
                "entry_symbols": ["run"],
                "confidence": "high",
            }
        ],
        "entrypoints": ["main.py"],
        "configs": [],
        "tests": [],
        "validators_or_oracles": ["validator.py"],
        "priority_reading_paths": ["main.py", "validator.py"],
        "uncertainties": [],
    }


def batch(batch_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "facts": [
            {
                "id": f"fact_{batch_id}",
                "category": "constraint",
                "statement": "当前证据足以完成本批次。",
                "evidence": ["main.py::run"],
                "confidence": "medium",
            }
        ],
        "assessments": [
            {
                "question_id": "batch_evidence",
                "sufficiency": "sufficient",
                "evidence": ["main.py::run"],
                "missing_information": None,
                "requested_read_ids": [],
            }
        ],
        "read_requests": [],
        "contradictions": [],
        "unresolved": [],
        "compressed_summary": f"{batch_id} 已完成并保留证据引用。",
        "ready_for_synthesis": True,
    }


def requesting_batch() -> dict:
    return {
        "schema_version": "1.0",
        "batch_id": "project_and_environment",
        "facts": [],
        "assessments": [
            {
                "question_id": "runtime_validator",
                "sufficiency": "insufficient",
                "evidence": ["main.py::run"],
                "missing_information": "需要确认外部验证函数。",
                "requested_read_ids": ["read_runtime_validator"],
            }
        ],
        "read_requests": [
            {
                "id": "read_runtime_validator",
                "source_file": "main.py",
                "source_symbol": "run",
                "target_file": "validator.py",
                "target_symbol": "validate",
                "reason": "validator_or_oracle",
                "priority": "high",
                "expected_evidence": "确认验证函数是否影响运行时可行性。",
            }
        ],
        "contradictions": [],
        "unresolved": ["验证函数尚未读取"],
        "compressed_summary": "等待受控复读 validator.validate。",
        "ready_for_synthesis": True,
    }


def final_analysis() -> dict:
    evidence = [
        {
            "file": "main.py",
            "symbol": "run",
            "detail": "run 是项目执行入口。",
        }
    ]
    return {
        "schema_version": "1.0",
        "language": "zh-CN",
        "summary": "这是一个最小调度项目。",
        "project_type": "scheduling_optimization",
        "problem_families": ["JSP"],
        "environments": [
            {
                "kind": "offline",
                "statement": "离线运行。",
                "evidence": evidence,
                "confidence": "medium",
            }
        ],
        "objectives": [
            {
                "id": "objective_1",
                "kind": "makespan",
                "sense": "minimize",
                "priority": 1,
                "statement": "缩短最大完工时间。",
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
                "statement": "保持工序优先关系。",
                "evidence": evidence,
                "confidence": "medium",
            }
        ],
        "decisions": [
            {
                "id": "decision_1",
                "kind": "sequence",
                "modifiable": True,
                "statement": "决定工序顺序。",
                "evidence": evidence,
                "confidence": "medium",
            }
        ],
        "oracles": [],
        "allowed_interventions": ["sequence"],
        "unknowns": [],
        "overall_confidence": "medium",
    }


def test_navigation_inventory_excludes_papers_and_secrets(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=x\n", encoding="utf-8")
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "survey.md").write_text("paper", encoding="utf-8")
    inventory = build_navigation_inventory(tmp_path)
    assert [item.path for item in inventory.files] == ["main.py"]
    assert inventory.excluded_paper_count == 1
    assert "SECRET" not in inventory.model_dump_json()


def test_controlled_reader_approves_related_insufficient_request(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        "from validator import validate\n\n"
        "def run(schedule):\n"
        "    return validate(schedule)\n",
        encoding="utf-8",
    )
    (tmp_path / "validator.py").write_text(
        "def validate(schedule):\n"
        "    return bool(schedule)\n",
        encoding="utf-8",
    )
    index = ProjectCodeIndex(tmp_path)
    request = EvidenceReadRequest(
        id="read_validator",
        source_file="main.py",
        source_symbol="run",
        target_file="validator.py",
        target_symbol="validate",
        reason="validator_or_oracle",
        priority="high",
        expected_evidence="确认 validate 是否真正执行硬约束验证。",
    )
    analysis = BatchAnalysis.model_validate(
        {
            "schema_version": "1.0",
            "batch_id": "objectives_and_constraints",
            "facts": [],
            "assessments": [
                {
                    "question_id": "hard_constraint_check",
                    "sufficiency": "insufficient",
                    "evidence": ["main.py::run"],
                    "missing_information": "validate 实现尚未读取",
                    "requested_read_ids": ["read_validator"],
                }
            ],
            "read_requests": [request.model_dump(mode="json")],
            "contradictions": [],
            "unresolved": [],
            "compressed_summary": "需要验证器实现证据。",
            "ready_for_synthesis": False,
        }
    )
    decisions, approved = ControlledReadGate(index).decide(analysis)
    assert decisions[0].approved
    evidence = index.read(approved[0])
    assert "def validate" in evidence.excerpt


def test_staged_pipeline_separates_navigator_and_analyst(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "def run():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "validator.py").write_text(
        "def validate():\n    return True\n", encoding="utf-8"
    )
    navigator_provider = FakeProvider("cheap", [navigation()])
    analyst_provider = FakeProvider(
        "strong",
        [
            batch("project_and_environment"),
            batch("objectives_and_constraints"),
            batch("decisions_oracles_and_unknowns"),
            final_analysis(),
        ],
    )
    result = compile_project_semantics_staged(
        tmp_path,
        navigator_provider=navigator_provider,
        analyst_provider=analyst_provider,
        max_rounds_per_batch=2,
    )
    assert len(navigator_provider.requests) == 1
    assert navigator_provider.requests[0].reasoning_effort == "medium"
    assert len(analyst_provider.requests) == 4
    assert all(
        request.reasoning_effort == "max" for request in analyst_provider.requests
    )
    assert result.final.evidence_pass_rate == 1.0
    assert result.metadata.approved_reads == 0
    assert len(result.short_term_memory.batch_summaries) == 3


def test_staged_pipeline_reasks_after_an_approved_read(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "from validator import validate\n\n"
        "def run(schedule):\n"
        "    return validate(schedule)\n",
        encoding="utf-8",
    )
    (tmp_path / "validator.py").write_text(
        "def validate(schedule):\n"
        "    return bool(schedule)\n",
        encoding="utf-8",
    )
    nav = navigation()
    nav["priority_reading_paths"] = ["main.py"]
    navigator_provider = FakeProvider("cheap", [nav])
    analyst_provider = FakeProvider(
        "strong",
        [
            requesting_batch(),
            batch("project_and_environment"),
            batch("objectives_and_constraints"),
            batch("decisions_oracles_and_unknowns"),
            final_analysis(),
        ],
    )
    result = compile_project_semantics_staged(
        tmp_path,
        navigator_provider=navigator_provider,
        analyst_provider=analyst_provider,
        max_rounds_per_batch=2,
    )
    assert result.metadata.approved_reads == 1
    assert len(result.batches[0].rounds) == 2
    assert "def validate" in analyst_provider.requests[1].user
