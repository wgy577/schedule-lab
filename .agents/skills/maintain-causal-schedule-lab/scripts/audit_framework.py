#!/usr/bin/env python3
"""Read-only architecture/document consistency audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def audit(root: Path) -> dict[str, object]:
    required_documents = (
        "README.md",
        "EXPERIMENTS.md",
        "PROJECT_STATUS_AND_ROADMAP.md",
        "PROJECT_MODULE_GRAPH.md",
        "PROJECT_MODULE_GRAPH_INTERACTIVE.html",
        "docs/README.md",
        "docs/architecture/SYSTEM_ARCHITECTURE.md",
        "docs/architecture/AGENT_PLATFORM_WORKPLAN.md",
        "docs/architecture/CAUSAL_MODULE_WORKPLAN.md",
        "docs/architecture/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md",
        "docs/architecture/FORMULA_IMPLEMENTATION_MATRIX.md",
        "docs/architecture/TRACEABILITY.md",
        "docs/architecture/FRAMEWORK_REQUIREMENTS.md",
        "docs/semantics/LLM_SEMANTIC_COMPILER.md",
        "docs/semantics/LLM_SEMANTIC_HARNESS.md",
        "docs/semantics/L2D_MODEL_COMPARISON_REPORT.md",
        "docs/semantics/L2D_PAPER_CODE_HUMAN_VERIFICATION.md",
        "docs/semantics/CODE_ONLY_SEMANTIC_LEARNING.md",
        "docs/semantics/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md",
        "docs/experiments/EXPERIMENT_PROTOCOL.md",
        "docs/guides/ADAPTER_GUIDE.md",
        ".agents/skills/maintain-causal-schedule-lab/SKILL.md",
        ".agents/skills/maintain-causal-schedule-lab/agents/openai.yaml",
        ".agents/skills/maintain-causal-schedule-lab/references/maintenance-contract.md",
    )
    errors: list[str] = []
    warnings: list[str] = []
    for relative in required_documents:
        if not (root / relative).is_file():
            errors.append(f"missing required document: {relative}")

    architecture_path = root / "docs/architecture/SYSTEM_ARCHITECTURE.md"
    formula_path = root / "docs/architecture/FORMULA_IMPLEMENTATION_MATRIX.md"
    readme_path = root / "README.md"
    architecture = (
        architecture_path.read_text(encoding="utf-8")
        if architecture_path.exists()
        else ""
    )
    formulas = (
        formula_path.read_text(encoding="utf-8")
        if formula_path.exists()
        else ""
    )
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    module_graph_path = root / "PROJECT_MODULE_GRAPH.md"
    module_graph = (
        module_graph_path.read_text(encoding="utf-8")
        if module_graph_path.exists()
        else ""
    )
    interactive_graph_path = root / "PROJECT_MODULE_GRAPH_INTERACTIVE.html"
    interactive_graph = (
        interactive_graph_path.read_text(encoding="utf-8")
        if interactive_graph_path.exists()
        else ""
    )

    required_headings = (
        "## 3. 总体架构",
        "## 4. 当前实际运行链",
        "## 5. 分层模块现状",
        "## 6. 核心数据契约",
        "## 7. 数学框架实现概况",
        "## 8. 训练阶段",
        "## 9. 验证、Oracle 与接受机制",
        "## 10. 搜索、停止与成本模型",
        "## 12. 人与 Agent 的职责边界",
        "## 13. 当前能力与关键缺口",
        "## 15. 文档持续维护规则",
        "### 15.4 变更记录",
    )
    for heading in required_headings:
        if heading not in architecture:
            errors.append(f"architecture missing heading: {heading}")

    source_root = root / "src/causal_schedule_lab"
    ignored_modules = {"__init__.py", "domain_plugins/__init__.py", "solvers/__init__.py"}
    modules = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        if (
            relative in ignored_modules
            or relative.endswith("/__init__.py")
            or "__pycache__" in path.parts
        ):
            continue
        modules.append(relative)
        if f"`src/causal_schedule_lab/{relative}`" not in architecture:
            errors.append(f"source module absent from architecture inventory: {relative}")

    expected_formula_ids = [f"F{index:02d}" for index in range(1, 26)]
    table_ids = re.findall(r"(?m)^\| (F\d{2}) \|", formulas)
    for formula_id in expected_formula_ids:
        count = table_ids.count(formula_id)
        if count != 1:
            errors.append(
                f"formula summary must contain {formula_id} exactly once; found {count}"
            )
    unexpected = sorted(set(table_ids) - set(expected_formula_ids))
    if unexpected:
        errors.append(f"unexpected formula IDs: {unexpected}")

    for link in (
        "docs/architecture/SYSTEM_ARCHITECTURE.md",
        "PROJECT_MODULE_GRAPH.md",
        "PROJECT_STATUS_AND_ROADMAP.md",
        "docs/architecture/FORMULA_IMPLEMENTATION_MATRIX.md",
        ".agents/skills/maintain-causal-schedule-lab/SKILL.md",
    ):
        if link not in readme:
            errors.append(f"README missing maintenance link: {link}")

    if re.search(r"(?i)!\[[^\]]*\]\([^)]*\.png(?:[?#][^)]*)?\)", module_graph):
        errors.append("PROJECT_MODULE_GRAPH.md must not embed PNG snapshots")
    for required_graph_text in (
        "第一层：大模块图",
        "第二层：大模块内部实现图",
        "PROJECT_MODULE_GRAPH_INTERACTIVE.html",
    ):
        if required_graph_text not in module_graph:
            errors.append(
                f"project module graph missing synchronization marker: {required_graph_text}"
            )
    for required_interactive_text in (
        "第一层：大模块",
        "第二层：实现细节",
        "SQLite Property Graph",
        "Candidate Variation Gate",
    ):
        if required_interactive_text not in interactive_graph:
            errors.append(
                f"interactive graph missing current node/control: {required_interactive_text}"
            )

    status_terms = (
        "主闭环已运行",
        "代码已实现但未接入",
        "接口已预留",
        "尚未实现",
        "外部责任",
    )
    for term in status_terms:
        if term not in architecture:
            warnings.append(f"architecture does not use status term: {term}")

    return {
        "project_root": str(root),
        "passed": not errors,
        "source_modules_checked": len(modules),
        "formula_ids_checked": len(expected_formula_ids),
        "documents_checked": len(required_documents),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(project_root())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status}: {result['source_modules_checked']} modules, "
            f"{result['formula_ids_checked']} formulas, "
            f"{result['documents_checked']} documents"
        )
        for warning in result["warnings"]:
            print(f"WARN: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
