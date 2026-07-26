---
name: maintain-causal-schedule-lab
description: Maintain the Causal Schedule Lab architecture, formula implementation matrix, module status, training stages, Oracle boundaries, cost model, tests, and roadmap as the code evolves. Use when changing scheduling IR, causal graphs, CIP discovery, learning losses, Agentic RL, repair/search, validation, experiment pipelines, Headroom integration, project documentation, or when auditing whether mathematical claims are truly implemented and connected to the runtime loop.
---

# Maintain Causal Schedule Lab

Keep code, mathematical claims, runtime behavior, tests, and project documents consistent.
Never upgrade a status from the existence of a class alone.

## Canonical documents

Read these files before changing architectural claims:

1. `PROJECT_STATUS_AND_ROADMAP.md`
2. `PROJECT_MODULE_GRAPH.md`
3. `docs/README.md`
4. `docs/intersections/README.md`
5. `tests/README.md`
6. `docs/cross_module/SYSTEM_ARCHITECTURE.md`
7. `docs/modules/module_b_semantics_memory/AGENT_PLATFORM_WORKPLAN.md`
8. `docs/modules/module_d_diagnosis_causality/CAUSAL_MODULE_WORKPLAN.md`
9. `docs/modules/module_b_semantics_memory/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md`
10. `docs/cross_module/FORMULA_IMPLEMENTATION_MATRIX.md`
11. `docs/cross_module/TRACEABILITY.md`
12. `docs/modules/module_a_input_evidence/FRAMEWORK_REQUIREMENTS.md`
13. `docs/modules/module_b_semantics_memory/LLM_SEMANTIC_COMPILER.md`
14. `docs/modules/module_b_semantics_memory/CODE_ONLY_SEMANTIC_LEARNING.md`
15. `docs/modules/module_b_semantics_memory/LLM_SEMANTIC_HARNESS.md`
16. `docs/modules/module_b_semantics_memory/L2D_MODEL_COMPARISON_REPORT.md`
17. `docs/modules/module_a_input_evidence/L2D_PAPER_CODE_HUMAN_VERIFICATION.md`
18. `docs/modules/module_b_semantics_memory/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md`
19. `docs/modules/module_h_experiments_statistics/EXPERIMENT_PROTOCOL.md`
20. `docs/modules/module_d_diagnosis_causality/SECONDARY_METRIC_AND_DIAGNOSTIC_EXPANSION_PLAN.md`
21. `EXPERIMENTS.md`
22. `README.md`

Read `references/maintenance-contract.md` for the status vocabulary, evidence gates, and
change-to-document mapping.

## Workflow

### Documentation ownership

- Every topical Markdown document must have one primary A–H module and live under the
  matching `docs/modules/module_*` directory.
- Put system-wide architecture, formula matrices and traceability documents under
  `docs/cross_module/`.
- Immediately after the title, include `所属模块` (or `所属范围` for cross-module files),
  document responsibility and any associated module.
- A document spanning two or more modules must declare a stable `交叉分类` such as
  `A × B`, then be linked from the matching `docs/intersections/*/README.md`. Keep one
  canonical body; intersection directories are indexes, never duplicated copies.
- New document types do not create new top-level content buckets such as `architecture`,
  `semantics`, `guides` or `experiments`; route them by system responsibility instead.
- When a file moves, update `docs/README.md`, its module README, root README, maintenance
  skill paths and every relative Markdown link in the same change.

### Reusable test assets

- Before writing a test, search `tests/test_registry.json` by module and capability, then
  reuse or minimally extend the closest registered file.
- Do not create ad-hoc root-level `test.py`, `tmp_test.py`, or one-off test scripts.
- A genuinely new test file must be named `tests/test_<capability>.py`, added to
  `tests/test_registry.json`, and start with a visible `# TEST-TAGS:` header matching the
  registry.
- Preserve stable fixtures and helpers when a future case differs only by parameters.
- Run targeted selections with Pytest markers before the complete suite; marker examples
  and the inventory are maintained in `tests/README.md`.

### 1. Establish scope

- Inspect `git status` when the directory is a Git repository.
- Use `rg` to identify changed or requested modules.
- Preserve unrelated user changes.
- Determine whether the change affects data contracts, formulas, runtime flow,
  training, validation, experiments, costs, or external responsibilities.

### 2. Read implementation evidence

- Read the actual call path, not only type definitions.
- Find where the module is instantiated and called.
- Find tests and generated artifacts.
- Distinguish a deterministic fallback from a learned model.
- Distinguish smoke success from trained or statistically validated performance.

### 3. Assign the truthful status

Use exactly:

- `主闭环已运行`
- `代码已实现但未接入`
- `接口已预留`
- `尚未实现`
- `外部责任`

Use `部分实现` only in the formula matrix when one mathematical term is implemented and
another is not.

Do not claim:

- trained without dataset hash, configuration, checkpoint, and metrics;
- validated improvement without incumbent, candidate, Full Oracle, and audit record;
- cross-family generalization without held-out family experiments;
- convergence when only a fixed iteration budget stopped;
- global optimality from local search;
- causal identification from a rule-derived graph.

### 4. Update every affected surface

- Architecture/module/data flow: update `docs/cross_module/SYSTEM_ARCHITECTURE.md`.
- Formula, reward, loss, posterior, or stopping rule: update
  `docs/cross_module/FORMULA_IMPLEMENTATION_MATRIX.md`.
- Literature discovery, normalization, promotion, removal or reclassification of a
  secondary metric or diagnostic target: update
  `docs/modules/module_d_diagnosis_causality/SECONDARY_METRIC_AND_DIAGNOSTIC_EXPANSION_PLAN.md`.
- Causal provenance, causal data, graph, CIP, path, closure, or causal-model stage:
  update `docs/modules/module_d_diagnosis_causality/CAUSAL_MODULE_WORKPLAN.md`.
- Runtime, provider, tool, budget, artifact, checkpoint, replay, approval, or platform
  priority: update `docs/modules/module_b_semantics_memory/AGENT_PLATFORM_WORKPLAN.md`.
- LLM semantic choices, evidence packet, JSON contract, provider behavior, or review flow:
  update `docs/modules/module_b_semantics_memory/LLM_SEMANTIC_COMPILER.md`.
- Navigator file roles/sharding, semantic batch definitions, evidence-sufficiency choices,
  controlled read approval rules, short-term context compression, long-term-memory injection,
  synthesis order, or prompt timing must update the explicit end-to-end chain in
  `docs/modules/module_b_semantics_memory/LLM_SEMANTIC_COMPILER.md`, all three layers of `PROJECT_MODULE_GRAPH.md`,
  `PROJECT_MODULE_GRAPH_INTERACTIVE.html`, and the truthful milestone in
  `PROJECT_STATUS_AND_ROADMAP.md` in the same change.
- Paper/code alignment, missing-modality training, semantic distillation, code-only
  evaluation, or leakage controls: update `docs/modules/module_b_semantics_memory/CODE_ONLY_SEMANTIC_LEARNING.md`.
- Blind paper-label evaluation, online benchmark cases, read planning, semantic scoring,
  or stage cost reporting: update `docs/modules/module_b_semantics_memory/LLM_SEMANTIC_HARNESS.md`.
- Multi-model semantic benchmark results or routing recommendations: update
  `docs/modules/module_b_semantics_memory/L2D_MODEL_COMPARISON_REPORT.md` and `EXPERIMENTS.md`.
- Paper claims, code facts, paper-code conflicts, semantic equivalence, or label promotion:
  update `docs/modules/module_a_input_evidence/L2D_PAPER_CODE_HUMAN_VERIFICATION.md` before changing formal scores.
- Scheduling-family knowledge, retrieval, variant priors, constraint roles, or impact weights:
  update `docs/modules/module_b_semantics_memory/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md`; never let a low
  optimization weight remove a hard constraint from validation.
- Long-term memory, graph truth, hierarchical retrieval, causal mechanism targets,
  candidate-variation gates, or direct/indirect controllability: update
  `docs/modules/module_b_semantics_memory/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md`.
- Any module, data-flow, implementation-detail, current-status, milestone, or roadmap
  change: update `PROJECT_MODULE_GRAPH.md` and
  `PROJECT_STATUS_AND_ROADMAP.md`.
- Any change that alters a module, status, edge, data flow, file mapping, milestone, or
  user-visible architecture must update every still-current visual in the same change:
  all three Mermaid layers in `PROJECT_MODULE_GRAPH.md`,
  `PROJECT_MODULE_GRAPH_INTERACTIVE.html`, README/document-index entries, and any
  screenshot, SVG, PNG, diagram, or showcase image that still represents the current
  project. Regenerate it, explicitly mark it historical, or remove it. Never leave a
  stale visual labeled as current.
- Keep `PROJECT_MODULE_GRAPH.md` text/Mermaid based; do not embed static PNG snapshots
  there. The interactive HTML is the zoomable current view.
- Project graphs show only implemented project components and actual current edges. Keep
  unimplemented retrievers, agents, validators, causal stages, and other future work in
  `PROJECT_STATUS_AND_ROADMAP.md`, not as graph nodes. Third-layer nodes must visibly cite
  their parent second-layer IDs.
- Original framework coverage: update `docs/cross_module/TRACEABILITY.md`.
- Training configuration/stage: update `configs/training.json` and architecture stage table.
- Adapter/Oracle contract: update `docs/modules/module_c_ir_adapters/ADAPTER_GUIDE.md`.
- Experiment design: update `docs/modules/module_h_experiments_statistics/EXPERIMENT_PROTOCOL.md`.
- Actually executed result: update `EXPERIMENTS.md`.
- User-facing entry point or command: update `README.md`.
- For a material framework change, update the architecture's verification date and append
  one concise entry to section 15.4. Do not create a framework version for wording-only edits.

Keep planned work separate from measured results.

### 5. Validate

Run:

```bash
python3 .agents/skills/maintain-causal-schedule-lab/scripts/audit_framework.py
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m compileall -q src
```

For scoped changes, select registered assets instead of inventing a new test harness:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "module_b and module_d" -q
PYTHONPATH=src .venv/bin/python -m pytest -m cap_secondary_metric_catalog -q
```

When architecture visuals are affected, also verify:

- all three Markdown graph layers contain only implemented components and actual edges;
- every third-layer node visibly maps to one or more second-layer IDs;
- the interactive graph contains the same current structure;
- README and `docs/README.md` point to the current visualization;
- no current screenshot or exported diagram contradicts the canonical Markdown graph;
- `PROJECT_MODULE_GRAPH.md` contains no embedded PNG snapshot.

Run `bash scripts/reproduce_all.sh` when runtime flow, solver, validation, objective, CLI,
or accepted schedule behavior changed.

### 6. Report

Report:

- code behavior changed;
- documents updated;
- formula statuses changed;
- tests and smoke runs executed;
- remaining partial or missing elements.

Never hide a failed audit by weakening the document.

## Headroom

Use Headroom only for agent-facing logs, JSON, and tool output. Keep original training data,
schedule instances, and Oracle inputs uncompressed. Review `headroom learn` output before
using `--apply`. When Token accounting is added, record both original and compressed counts.
