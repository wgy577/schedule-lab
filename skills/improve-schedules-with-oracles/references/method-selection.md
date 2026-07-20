# Method selection and combination

## Controller layers

| Layer | Method | Best use | Important limit |
|---|---|---|---|
| Timing normalization | Sequence-preserving compaction | Remove avoidable idle while keeping all dispatch choices | Cannot escape a bad sequence or route |
| Family heuristic | NEH, shifting bottleneck, insertion, ATC/SPT, load balancing | Cheap ranked proposals and causal clues | Must be repaired and validated |
| Small exact change | CP-SAT/MILP local branching, fix-and-optimize | Stable improvement near a good incumbent | Requires a bounded change budget |
| Graph-guided change | Causal-closure repair | Radius neighborhoods expand unpredictably | Requires complete constraint/provenance arcs |
| Time decomposition | Rolling horizon, relax-and-fix | Long schedules and large suffixes | Use overlapping windows and prefix freezing |
| Resource decomposition | Shifting bottleneck, Benders/Lagrangian guidance | One stage/resource family dominates | Relaxation is guidance, not feasibility |
| Plateau escape | Deterministic ALNS | Irregular multi-block or multi-gap repairs | Bound destroy size and change cost |
| Diversification | Elite path relinking | Two structurally different validated elites exist | Repair every intermediate checkpoint |
| Experiment allocation | Bayesian posterior/UCB | Oracle calls are expensive and history exists | Select configurations, not schedules |
| Final authority | Generic validator + domain Oracle | Every candidate | Never bypass this layer |

## Family-specific first choices

- JSP: critical-path and critical-block diagnosis; shifting bottleneck; N5/N6 or adjacent block repair; CP-SAT local branching.
- FSP: preserve or explicitly alter the permutation; NEH insertion as a proposal; sink-stage gap repair; downstream recomputation.
- FJSP: alternate eligible-machine assignment and local sequence repair; use congestion or dual-price guidance; validate route/setup consequences.
- HFSP: diagnose starvation/blocking by stage; repair one parallel-machine stage and propagate downstream; focus on a single-capacity sink when present.
- Domain hybrid: use the closest family only for proposals; replay paths, continuity, time windows, collisions, and state-dependent delays in the Oracle.

## Causal-closure fix-and-optimize

This project-specific contribution replaces “take radius 3 around a gap” with a constraint-graph rule:

1. Start at the objective bottleneck: critical block, sink gap, late tail, starvation, or blocking interval.
2. Traverse backward through job precedence, resource-order, setup, route-binding, vehicle-continuity, and time-window arcs.
3. Estimate which predecessor arcs actually prevent the bottleneck from moving earlier.
4. Select the smallest responsible operation set and add every directly conflicting operation required for closure.
5. Freeze the complement; solve the released set with a local-branching/change budget.
6. If replay changes an outside decision, reject the open boundary and add only the evidence-backed expansion nodes.
7. Accept after full validation and restart diagnosis.

This remains local and auditable but is not tied to a geometric radius or operation number, so it transfers across JSP, FSP, FJSP, and HFSP.

## Recommended combination

Use a staged portfolio instead of mixing every method simultaneously:

```text
compaction
  → family heuristic or shifting bottleneck diagnosis
  → causal closure / local branching exact repair
  → tabu duplicate filter
  → validator / Oracle
  → accept and restart

plateau
  → rolling horizon or bounded deterministic ALNS
  → path relinking when two elites exist
  → Bayesian budget allocation after sufficient evidence
```

Do not use independent random multi-start as the production controller. A fixed-seed heuristic may remain a benchmark or proposal generator.
