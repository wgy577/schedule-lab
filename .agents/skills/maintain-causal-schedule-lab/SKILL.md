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
4. `docs/architecture/SYSTEM_ARCHITECTURE.md`
5. `docs/architecture/AGENT_PLATFORM_WORKPLAN.md`
6. `docs/architecture/CAUSAL_MODULE_WORKPLAN.md`
7. `docs/architecture/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md`
8. `docs/architecture/FORMULA_IMPLEMENTATION_MATRIX.md`
9. `docs/architecture/TRACEABILITY.md`
10. `docs/architecture/FRAMEWORK_REQUIREMENTS.md`
11. `docs/semantics/LLM_SEMANTIC_COMPILER.md`
12. `docs/semantics/CODE_ONLY_SEMANTIC_LEARNING.md`
13. `docs/semantics/LLM_SEMANTIC_HARNESS.md`
14. `docs/semantics/L2D_MODEL_COMPARISON_REPORT.md`
15. `docs/semantics/L2D_PAPER_CODE_HUMAN_VERIFICATION.md`
16. `docs/semantics/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md`
17. `docs/experiments/EXPERIMENT_PROTOCOL.md`
18. `EXPERIMENTS.md`
19. `README.md`

Read `references/maintenance-contract.md` for the status vocabulary, evidence gates, and
change-to-document mapping.

## Workflow

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

- Architecture/module/data flow: update `docs/architecture/SYSTEM_ARCHITECTURE.md`.
- Formula, reward, loss, posterior, or stopping rule: update
  `docs/architecture/FORMULA_IMPLEMENTATION_MATRIX.md`.
- Causal provenance, causal data, graph, CIP, path, closure, or causal-model stage:
  update `docs/architecture/CAUSAL_MODULE_WORKPLAN.md`.
- Runtime, provider, tool, budget, artifact, checkpoint, replay, approval, or platform
  priority: update `docs/architecture/AGENT_PLATFORM_WORKPLAN.md`.
- LLM semantic choices, evidence packet, JSON contract, provider behavior, or review flow:
  update `docs/semantics/LLM_SEMANTIC_COMPILER.md`.
- Navigator file roles/sharding, semantic batch definitions, evidence-sufficiency choices,
  controlled read approval rules, short-term context compression, long-term-memory injection,
  synthesis order, or prompt timing must update the explicit end-to-end chain in
  `docs/semantics/LLM_SEMANTIC_COMPILER.md`, both layers of `PROJECT_MODULE_GRAPH.md`,
  `PROJECT_MODULE_GRAPH_INTERACTIVE.html`, and the truthful milestone in
  `PROJECT_STATUS_AND_ROADMAP.md` in the same change.
- Paper/code alignment, missing-modality training, semantic distillation, code-only
  evaluation, or leakage controls: update `docs/semantics/CODE_ONLY_SEMANTIC_LEARNING.md`.
- Blind paper-label evaluation, online benchmark cases, read planning, semantic scoring,
  or stage cost reporting: update `docs/semantics/LLM_SEMANTIC_HARNESS.md`.
- Multi-model semantic benchmark results or routing recommendations: update
  `docs/semantics/L2D_MODEL_COMPARISON_REPORT.md` and `EXPERIMENTS.md`.
- Paper claims, code facts, paper-code conflicts, semantic equivalence, or label promotion:
  update `docs/semantics/L2D_PAPER_CODE_HUMAN_VERIFICATION.md` before changing formal scores.
- Scheduling-family knowledge, retrieval, variant priors, constraint roles, or impact weights:
  update `docs/semantics/SCHEDULING_SEMANTIC_KNOWLEDGE_AND_IMPACT.md`; never let a low
  optimization weight remove a hard constraint from validation.
- Long-term memory, graph truth, hierarchical retrieval, causal mechanism targets,
  candidate-variation gates, or direct/indirect controllability: update
  `docs/architecture/LONG_TERM_MEMORY_AND_CAUSAL_MECHANISMS.md`.
- Any module, data-flow, implementation-detail, current-status, milestone, or roadmap
  change: update `PROJECT_MODULE_GRAPH.md` and
  `PROJECT_STATUS_AND_ROADMAP.md`.
- Any change that alters a module, status, edge, data flow, file mapping, milestone, or
  user-visible architecture must update every still-current visual in the same change:
  both Mermaid layers in `PROJECT_MODULE_GRAPH.md`,
  `PROJECT_MODULE_GRAPH_INTERACTIVE.html`, README/document-index entries, and any
  screenshot, SVG, PNG, diagram, or showcase image that still represents the current
  project. Regenerate it, explicitly mark it historical, or remove it. Never leave a
  stale visual labeled as current.
- Keep `PROJECT_MODULE_GRAPH.md` text/Mermaid based; do not embed static PNG snapshots
  there. The interactive HTML is the zoomable current view.
- Original framework coverage: update `docs/architecture/TRACEABILITY.md`.
- Training configuration/stage: update `configs/training.json` and architecture stage table.
- Adapter/Oracle contract: update `docs/guides/ADAPTER_GUIDE.md`.
- Experiment design: update `docs/experiments/EXPERIMENT_PROTOCOL.md`.
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

When architecture visuals are affected, also verify:

- both Markdown graph layers contain the changed module/state/edge;
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
