---
name: improve-schedules-with-oracles
description: Improve an existing JSP, FSP, FJSP, HFSP, carrier, or domain-constrained schedule through deterministic incumbent-preserving search, exact repair, heuristics, decomposition, statistical experiment allocation, and a domain Oracle. Use when diagnosing bottlenecks, reducing makespan/tardiness/idle/blocking, changing machine assignments or sequences safely, comparing validated candidates, or building a reproducible scheduling optimization workflow rather than scheduling from scratch.
---

# Improve Schedules With Oracles

Improve the supplied incumbent monotonically. Let diagnostics choose where to search, exact or bounded algorithms construct candidates, and validators plus the domain Oracle decide acceptance.

## Workflow

1. Locate `schedule_lab`, its canonical adapter, incumbent, validator, and optional simulator.
2. Preserve the incumbent. Record normalized hashes for the problem, schedule, objective, solver settings, and Oracle version.
3. Validate the incumbent before modifying it. Do not infer feasibility from a Gantt chart.
4. Generate a method plan with `schedule-lab improvement-workflow`. Read [references/method-selection.md](references/method-selection.md) when selecting or combining methods.
   Read [references/family-strategy-routing.md](references/family-strategy-routing.md) when the Agent must diagnose and select among multiple JSP/FSP/FJSP/HFSP strategy packs.
   Read [references/advanced-optimization-portfolio.md](references/advanced-optimization-portfolio.md) when neighborhood methods plateau, the user requests alternatives, or the instance needs decomposition, robustness, or an expensive Oracle.
   Read [references/agentic-rl-and-joint-trajectories.md](references/agentic-rl-and-joint-trajectories.md) when the user requests Agentic RL, an encoder, learned search control, or joint scheduling and conflict-free trajectory optimization.
5. Run phases in order:
   - compact timing while fixing modes and resource orders;
   - test the smallest family-specific deterministic operator;
   - use causal-closure or local-branching fix-and-optimize;
   - use rolling-horizon or deterministic ALNS after a declared plateau;
   - use elite path relinking only with two validated elites;
   - use Bayesian allocation only after at least 12 deduplicated validated experiments.
6. Freeze every decision outside the declared release boundary. Rebuild propagation closure from validator or Oracle evidence rather than silently changing outside assignments.
7. Run generic validation, then the original domain Oracle when required. Recompute the full objective after Oracle-inserted waits or delays.
8. Accept only a lexicographically better, reproduced candidate. Otherwise retain the incumbent and record the rejection.
9. After acceptance, recompute diagnostics and restart from the accepted incumbent.

For latency-sensitive work, default to the cost-aware fast controller: shortlist at most three compatible structural methods, inspect at most two bottlenecks, evaluate at most four bounded candidates, stop on the first strict validated improvement, and send at most one survivor to a domain Oracle. Use `balanced` or `thorough` only when the fast shortlist returns the incumbent.

Use the adaptive controller by default when the user prioritizes speed: run `fast` first, escalate once to `balanced` only if no strict improvement or provisional Oracle survivor exists, and exclude every signature already evaluated by `fast`.

## Method boundaries

- Use VNS/ALNS to select a bounded search region, not to certify a schedule.
- Use CP-SAT/MILP/exact enumeration to construct legal released-region assignments.
- Use stable constructive heuristics only as proposal sources. Repair and validate their proposals.
- Use tabu hashes to skip duplicate moves and cycles.
- Use Bayesian statistics to allocate experiment budget, never to waive constraints.
- Use evolutionary search only offline with fixed seeds, deterministic repair, and a cheap pre-Oracle filter.
- Prefer causal closure, local branching, shifting bottleneck, rolling horizon, dual-price release, or path relinking when radius neighborhoods plateau.
- Prefer logic-based Benders with reusable Oracle cuts when machine/sequence decisions are cheap to model but routing, collision, or simulation feasibility is expensive.
- Optimize expected quality and robustness only after hard feasibility; never hide scenario violations inside a weighted scalar objective.

## Project commands

From the repository root (the wrapper is robust even when the editable virtual-environment entry point is stale):

```bash
schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  improvement-workflow problem.json incumbent.json \
  --evidence-count 12 --validated-elite-count 2 \
  --no-improvement-count 4 --oracle-failure-count 2 \
  --output outputs/improvement_workflow.json
schedule_lab/.venv/bin/python schedule_lab/run_schedule_lab.py \
  adaptive-improve problem.json incumbent.json \
  --output outputs/adaptive_improvement.json
schedule_lab/.venv/bin/python -m unittest discover -s schedule_lab/tests -v
```

For the carrier project and comparison-video contract, read [references/carrier-and-video.md](references/carrier-and-video.md). Use `scripts/audit_raw_schedules.py` before comparing arbitrary legacy JSON schedules.

## Exact Oracle-cut memory

After deterministic domain replay, extract exact no-good cuts before launching another batch:

```bash
PYTHONPATH=schedule_lab/src schedule_lab/.venv/bin/python -m schedule_lab.cli \
  carrier-oracle-cuts --history <search-history...> \
  --output schedule_lab/outputs/carrier_oracle_cuts.json
```

Pass the resulting store to `carrier-alns-search --oracle-cuts ...`. An exact cut may skip only the same incumbent hash, neighborhood signature, operator, dispatch swaps, and seed. Treat `requiredExpansionJobs` as evidence for the next causal-closure experiment. Never generalize one replay failure to other neighborhoods, routes, geometry versions, seeds, or incumbents.

## Required evidence

Return incumbent and candidate hashes, diagnosed causal bottleneck, released/frozen sets, selected method and deterministic budget, machine/sequence/timing diff, validation and Oracle results, objective-vector delta, acceptance decision, reproduction command, and artifact paths.
