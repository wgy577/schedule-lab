# Maintenance contract

## Canonical truth order

When sources disagree, resolve in this order:

1. Executed tests and reproducible artifacts;
2. Runtime call path;
3. Implementation code;
4. Data models and protocols;
5. Architecture documents;
6. Roadmap statements.

A roadmap never proves an implementation.

## Evidence gates

| Claim | Minimum evidence |
|---|---|
| Code implemented | Importable implementation plus focused test |
| Runtime integrated | Default call path plus end-to-end test |
| Deterministic | Fixed seed/worker/budget and replay hash equality |
| Trained | Dataset hash, split, config, checkpoint, train/validation metrics |
| Oracle validated | Oracle identity/version, input, output and pass record |
| Strict improvement | Incumbent/candidate objectives and Full validation |
| Cross-family | Held-out instances for all claimed families |
| Converged | Declared neighborhood coverage and stopping evidence |

## Formula status checks

For every formula:

1. Identify the mathematical inputs.
2. Identify the code producing each input.
3. Identify the function computing the expression.
4. Identify the runtime caller.
5. Identify the test or experiment.
6. Record missing terms separately.

Example: a reward helper is `代码已实现但未接入` until transitions call it and PPO consumes
those transitions.

## Change mapping

| Changed path/pattern | Documents to inspect |
|---|---|
| `ir.py`, `models.py`, `project.py` | architecture, adapter guide, traceability |
| `graph.py`, `cip.py`, `tensorization.py` | architecture, formula F01–F03 |
| Causal module provenance, data or stage status | architecture, causal workplan, formula matrix |
| Runtime, provider, tools, budget, artifacts, replay or approval | architecture, Agent platform workplan |
| Long-term graph memory, evidence retrieval or mechanism targets | architecture, project module graph, project status, long-term memory contract |
| Module, status, data flow, file mapping or milestone | both Mermaid graph layers, interactive graph, README/index visual entries, and every still-current exported visual |
| `llm_semantics.py`, `providers/` | architecture, Agent platform workplan, LLM semantic compiler |
| Paper/code pairing, distillation, modality dropout or code-only evaluation | code-only semantic learning, experiment protocol |
| Blind paper labels, repository read planning or semantic evaluation | LLM semantic Harness, experiment protocol, experiments |
| Provider comparison, benchmark routing or API stability | model comparison report, experiments, Agent platform workplan |
| `learning.py`, `trainers.py` | architecture, formula F03/F13–F15 |
| `agent.py`, `agentic_rl.py` | architecture, formula F04/F05/F12/F16–F18 |
| `operators.py`, `repair.py`, `search.py` | architecture, formula F06/F10/F20/F25 |
| `conditional_generator.py` | architecture, formula F19/F20 |
| `validation.py`, `core_validation.py` | architecture, formula F07/F08/F21–F23 |
| `posterior.py` | architecture, formula F10/F11/F23 |
| `controller.py` | architecture runtime flow, acceptance, stopping |
| `dataset.py`, `counterfactuals.py` | formula F07/F09/F13–F15 |
| `training_pipeline.py`, `configs/training.json` | architecture stage table |
| `experiment_runner.py`, `statistics.py` | experiment protocol and experiments |
| Headroom or Token integration | architecture cost model and formula F24 |

## Human-maintained decisions

Keep these visibly marked `外部责任` until supplied:

- instance semantics;
- business hard constraints;
- domain Oracle;
- objective priority and tolerance;
- time and Token budgets;
- licensing and publication boundaries;
- final business acceptance.

## Audit behavior

Run `scripts/audit_framework.py` from the project root. It checks:

- canonical documents exist;
- every current source module appears in the detailed architecture or an explicit ignore list;
- formula IDs F01–F25 are present exactly once in the summary table;
- mandatory architecture sections exist;
- README links to the detailed documents and maintenance Skill.

The audit checks consistency, not scientific correctness. Tests and experiments remain required.

## Visual synchronization gate

- `PROJECT_MODULE_GRAPH.md` is the canonical reviewable graph and must not embed PNG
  snapshots.
- `PROJECT_MODULE_GRAPH_INTERACTIVE.html` is the canonical zoomable rendering.
- Architecture-affecting changes must update both in the same change.
- Other screenshots, SVGs, PNGs, videos, or showcase images that claim to show the current
  architecture must be regenerated, marked historical, or removed.
